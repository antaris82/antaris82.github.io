#!/usr/bin/env python3
"""
Effective simplicial geometry closure test for dynamic CNNA/NGF provenance growth.

This test follows the dynamic birth/backreaction model from the previous sprint,
but now extracts several effective simplicial complexes and asks:

1. Does the response monodromy appear as topological H1?
2. If H1 is killed by filled 2-simplices, does the response survive as
   local 2-form curvature delta A on faces?
3. Is the effect present only in a selected response layer, or also in the
   extracted effective geometry?
4. Which extraction modes are closure-capable, and which only create
   contractible/discrete book-keeping structures?

Important:
- This is a numerical/model diagnostic, not a Lean theorem.
- Conductance/response weights are NOT geometric edge lengths.
- Equal geometric simplex edge lengths remain compatible with changing
  response/cochain data.
"""

from __future__ import annotations

import argparse
import csv
import math
import cmath
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np


EPS = 1e-12


@dataclass
class Node:
    id: int
    parent: Optional[int]
    level: int
    birth_order: int
    birth_time: int
    birth_g: float
    g: float
    children: List[int] = field(default_factory=list)


class DynamicBirthModel:
    def __init__(
        self,
        mode: str = "linear",
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
        eps: float = EPS,
    ):
        if branching != 3:
            raise ValueError("This diagnostic currently assumes ternary sibling triples.")
        self.mode = mode
        self.branching = branching
        self.base = base
        self.alpha_env = alpha_env
        self.ancestor_decay = ancestor_decay
        self.br_ancestor = br_ancestor
        self.br_sibling = br_sibling
        self.eps = eps
        self.nodes: Dict[int, Node] = {}
        self.t = 0
        self.next_id = 0
        self.directed_edges: Dict[Tuple[int, int], float] = defaultdict(float)
        self.triple_completion_time: Dict[int, int] = {}
        root = self._new_node(parent=None, level=0, birth_order=0, birth_g=1.0)
        self.root = root.id

    def _new_node(self, parent: Optional[int], level: int, birth_order: int, birth_g: float) -> Node:
        n = Node(
            id=self.next_id,
            parent=parent,
            level=level,
            birth_order=birth_order,
            birth_time=self.t,
            birth_g=birth_g,
            g=birth_g,
        )
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def parent_line(self, parent: int) -> List[int]:
        line: List[int] = []
        cur: Optional[int] = parent
        while cur is not None:
            line.append(cur)
            cur = self.nodes[cur].parent
        return line

    def birth_environment_load(self, parent: int, older_siblings: List[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_decay ** (d - 1))
        for s in older_siblings:
            env += self.nodes[s].g
        return env

    def child_conductance_from_env(self, env_load: float) -> float:
        if self.mode == "linear":
            return self.base + self.alpha_env * env_load
        if self.mode == "log":
            return self.base + self.alpha_env * math.log1p(env_load)
        if self.mode == "saturating":
            return self.base + self.alpha_env * (env_load / (1.0 + env_load))
        raise ValueError(f"unknown mode: {self.mode}")

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env_load = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env_load)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g)
        c = child.id

        total_env = env_load + self.eps

        # Environment influence: parent line and older siblings -> newborn.
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(a, c)] += weight
        for s in older:
            contrib = self.nodes[s].g
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(s, c)] += weight

        # Newborn acts as UV-tail/backreaction for parent line.
        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * birth_g / (d * d)
            self.nodes[a].g += delta
            self.directed_edges[(c, a)] += delta

        # Newborn backreacts on older siblings.
        for s in older:
            delta = self.br_sibling * birth_g
            self.nodes[s].g += delta
            self.directed_edges[(c, s)] += delta

        if len(self.nodes[parent].children) == 3:
            self.triple_completion_time[parent] = self.t

        return c

    def grow_level(self, frontier: List[int]) -> List[int]:
        out: List[int] = []
        for p in frontier:
            for k in range(1, 4):
                out.append(self.add_child(p, k))
        return out

    def run(self, max_level: int) -> None:
        frontier = [self.root]
        for _level in range(1, max_level + 1):
            frontier = self.grow_level(frontier)

    def completed_parents(self) -> List[int]:
        return [p for p, t in self.triple_completion_time.items()]

    def w(self, u: int, v: int) -> float:
        return self.directed_edges.get((u, v), 0.0)


def e2(a: int, b: int) -> Tuple[int, int]:
    if a == b:
        raise ValueError("no loops")
    return (a, b) if a < b else (b, a)


def f3(a: int, b: int, c: int) -> Tuple[int, int, int]:
    xs = tuple(sorted((a, b, c)))
    if len(set(xs)) != 3:
        raise ValueError("degenerate face")
    return xs


@dataclass
class SimplicialComplex:
    vertices: Set[int]
    edges: Set[Tuple[int, int]]
    faces: Set[Tuple[int, int, int]]
    face_sources: Dict[Tuple[int, int, int], str] = field(default_factory=dict)

    def close_faces_edges(self) -> None:
        for a, b, c in list(self.faces):
            self.edges.add(e2(a, b))
            self.edges.add(e2(b, c))
            self.edges.add(e2(a, c))


def extract_complex(model: DynamicBirthModel, mode: str) -> SimplicialComplex:
    V = set(model.nodes.keys())
    E: Set[Tuple[int, int]] = set()
    F: Set[Tuple[int, int, int]] = set()
    face_sources: Dict[Tuple[int, int, int], str] = {}

    if mode in ("radial_tree", "sibling_cycle_unfilled", "sibling_triangle_filled",
                "parent_fan_filled", "full_local_surface"):
        for n in model.nodes.values():
            if n.parent is not None:
                E.add(e2(n.parent, n.id))

    for p in model.completed_parents():
        ch = model.nodes[p].children
        if len(ch) != 3:
            continue
        c1, c2, c3 = ch

        if mode in ("sibling_cycle_unfilled", "sibling_triangle_filled",
                    "parent_fan_filled", "full_local_surface"):
            E.add(e2(c1, c2))
            E.add(e2(c2, c3))
            E.add(e2(c3, c1))

        if mode in ("sibling_triangle_filled", "full_local_surface"):
            face = f3(c1, c2, c3)
            F.add(face)
            face_sources[face] = "sibling_face"

        if mode in ("parent_fan_filled", "full_local_surface"):
            for a, b in [(c1, c2), (c2, c3), (c3, c1)]:
                face = f3(p, a, b)
                F.add(face)
                face_sources[face] = "parent_fan"

    K = SimplicialComplex(V, E, F, face_sources)
    K.close_faces_edges()
    return K


def gf2_rank(rows: Iterable[int]) -> int:
    basis: Dict[int, int] = {}
    rank = 0
    for x in rows:
        y = x
        while y:
            p = y.bit_length() - 1
            if p in basis:
                y ^= basis[p]
            else:
                basis[p] = y
                rank += 1
                break
    return rank


def betti_numbers(K: SimplicialComplex) -> Tuple[int, int, int, int, int]:
    vertices = sorted(K.vertices)
    edges = sorted(K.edges)
    faces = sorted(K.faces)
    vi = {v: i for i, v in enumerate(vertices)}
    ei = {e: i for i, e in enumerate(edges)}

    # Boundary d1: C1 -> C0 over F2. Each edge column has two endpoints.
    d1_rows_by_edge_cols: List[int] = []
    # rank of d1 as matrix rows? Easier use columns as row bitsets under transpose; rank invariant.
    d1_cols = []
    for a, b in edges:
        d1_cols.append((1 << vi[a]) ^ (1 << vi[b]))
    r1 = gf2_rank(d1_cols)

    # Boundary d2: C2 -> C1. Each face column has three boundary edges.
    d2_cols = []
    for a, b, c in faces:
        bits = (1 << ei[e2(a, b)]) ^ (1 << ei[e2(a, c)]) ^ (1 << ei[e2(b, c)])
        d2_cols.append(bits)
    r2 = gf2_rank(d2_cols)

    n0, n1, n2 = len(vertices), len(edges), len(faces)
    b0 = n0 - r1
    b1 = n1 - r1 - r2
    b2 = n2 - r2
    return b0, b1, b2, r1, r2


def edge_cochain_A(model: DynamicBirthModel, u: int, v: int) -> float:
    """Antisymmetric log-ratio cochain on oriented edge u->v."""
    return math.log((model.w(u, v) + EPS) / (model.w(v, u) + EPS))


def face_curvature(model: DynamicBirthModel, face: Tuple[int, int, int]) -> float:
    """Orientation by sorted vertex order: a->b->c->a."""
    a, b, c = face
    return edge_cochain_A(model, a, b) + edge_cochain_A(model, b, c) + edge_cochain_A(model, c, a)


def sibling_face_curvature_by_parent(model: DynamicBirthModel, parent: int) -> Optional[float]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return None
    c1, c2, c3 = ch
    return edge_cochain_A(model, c1, c2) + edge_cochain_A(model, c2, c3) + edge_cochain_A(model, c3, c1)


def neutral_phasor(model: DynamicBirthModel, parent: int) -> Optional[complex]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return None
    vals = [model.nodes[c].g for c in ch]
    omega = cmath.exp(2j * math.pi / 3)
    return vals[0] + vals[1] * omega + vals[2] * omega**2


def local_full_markov_eig_class(model: DynamicBirthModel, parent: int) -> Tuple[str, str]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return "none", ""
    c1, c2, c3 = ch
    A = np.zeros((3, 3), dtype=float)
    # column-source: A[target, source]
    for i, u in enumerate(ch):
        for j, v in enumerate(ch):
            if u == v:
                continue
            A[j, i] = model.w(u, v)
    # column stochastic
    P = A.copy()
    for j in range(3):
        s = P[:, j].sum()
        if s > 0:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    ev = np.linalg.eigvals(P)
    cls = "complex_pair" if np.max(np.abs(np.imag(ev))) > 1e-9 else "real_or_degenerate"
    return cls, " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in ev)


def summarize_model(model: DynamicBirthModel, level: int, geom_mode: str) -> dict:
    K = extract_complex(model, geom_mode)
    b0, b1, b2, r1, r2 = betti_numbers(K)

    face_curvs = [face_curvature(model, face) for face in K.faces]
    sibling_curvs = []
    neutral_norms = []
    full_complex = 0
    count_triples = 0

    for p in model.completed_parents():
        curv = sibling_face_curvature_by_parent(model, p)
        if curv is not None:
            sibling_curvs.append(curv)
        z = neutral_phasor(model, p)
        if z is not None:
            mean_g = sum(model.nodes[c].g for c in model.nodes[p].children) / 3
            neutral_norms.append(abs(z) / mean_g)
        cls, _ = local_full_markov_eig_class(model, p)
        if cls == "complex_pair":
            full_complex += 1
        count_triples += 1

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def frac_nonzero(xs: List[float], tol: float = 1e-9) -> float:
        return sum(1 for x in xs if abs(x) > tol) / len(xs) if xs else 0.0

    return {
        "level": level,
        "geom_mode": geom_mode,
        "nodes": len(model.nodes),
        "edges": len(K.edges),
        "faces": len(K.faces),
        "completed_triples": len(model.completed_parents()),
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "rank_d1": r1,
        "rank_d2": r2,
        "mean_abs_face_curvature": mean([abs(x) for x in face_curvs]),
        "frac_nonzero_face_curvature": frac_nonzero(face_curvs),
        "mean_abs_sibling_curvature": mean([abs(x) for x in sibling_curvs]),
        "frac_nonzero_sibling_curvature": frac_nonzero(sibling_curvs),
        "mean_neutral_norm": mean(neutral_norms),
        "frac_full_local_complex": full_complex / count_triples if count_triples else 0.0,
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_suite(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    modes = [
        ("linear", 0.22),
        ("log", 0.22),
        ("saturating", 0.90),
    ]
    geom_modes = [
        "radial_tree",
        "sibling_cycle_unfilled",
        "sibling_triangle_filled",
        "parent_fan_filled",
        "full_local_surface",
    ]

    report_lines: List[str] = []
    for dyn_mode, alpha in modes:
        model = DynamicBirthModel(mode=dyn_mode, alpha_env=alpha)
        frontier = [model.root]
        # level 0
        for gm in geom_modes:
            row = summarize_model(model, 0, gm)
            row["dynamic_mode"] = dyn_mode
            rows.append(row)
        for level in range(1, max_level + 1):
            frontier = model.grow_level(frontier)
            for gm in geom_modes:
                row = summarize_model(model, level, gm)
                row["dynamic_mode"] = dyn_mode
                rows.append(row)

        report_lines.append(f"MODE {dyn_mode}")
        for gm in geom_modes:
            r = [x for x in rows if x["dynamic_mode"] == dyn_mode and x["level"] == max_level and x["geom_mode"] == gm][0]
            report_lines.append(
                f"  {gm}: V={r['nodes']} E={r['edges']} F={r['faces']} "
                f"b1={r['b1']} b2={r['b2']} "
                f"mean|face_curv|={r['mean_abs_face_curvature']:.6f} "
                f"frac face curv!=0={r['frac_nonzero_face_curvature']:.3f} "
                f"mean|sibling_curv|={r['mean_abs_sibling_curvature']:.6f} "
                f"full-local complex={r['frac_full_local_complex']:.3f}"
            )
        # Root diagnostics
        cls, evs = local_full_markov_eig_class(model, model.root)
        z = neutral_phasor(model, model.root)
        curv = sibling_face_curvature_by_parent(model, model.root)
        root_g = " ".join(f"{model.nodes[c].g:.9g}" for c in model.nodes[model.root].children)
        report_lines.append(f"  root g={root_g}")
        report_lines.append(f"  root sibling curvature={curv:.6f}")
        report_lines.append(f"  root neutral |Z|={abs(z):.6f}, phase={math.degrees(cmath.phase(z)):.3f} deg")
        report_lines.append(f"  root full-local Markov={cls}, eigs={evs}")
        report_lines.append("")

    write_csv(outdir / "geometry_level_summaries.csv", rows)
    summary = "\n".join(report_lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=6)
    ap.add_argument("--outdir", type=Path, default=Path("effective_geometry_closure_out"))
    args = ap.parse_args()
    print(run_suite(args.max_level, args.outdir))


if __name__ == "__main__":
    main()
