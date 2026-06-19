#!/usr/bin/env python3
"""
Effective operator J-candidate test for dynamic CNNA/NGF provenance growth.

This test is the next step after effective_geometry_closure:

- It stops treating topology/cochains as sufficient.
- It constructs local directed response transport operators M on each completed
  sibling triple.
- It decomposes M into symmetric and skew parts:
      S = (M + M^T)/2
      A = (M - M^T)/2
- It checks whether the skew part acts as a complex structure on the relevant
  2D sibling sector orthogonal to the constant vector (1,1,1).

Important guardrails
--------------------
This is a numerical diagnostic, not a Lean theorem.

A positive local J residual does NOT prove physical i, Type III, modular flow,
or a vN-algebra. It only shows that the response operator contains a stable
rotational 2D sector in this surrogate model.

The metric issue is NOT solved here:
- Euclidean/sum-zero sector test is used as a first invariant screen.
- The symmetric part S is diagnosed, but not declared to be the physical metric.
- A later operator/algebra tower must supply the correct G/weight/state.
"""

from __future__ import annotations

import argparse
import csv
import math
import cmath
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
            raise ValueError("This diagnostic assumes ternary sibling triples.")
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
        self.completed_parent_time: Dict[int, int] = {}
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
        out: List[int] = []
        cur: Optional[int] = parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

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
        raise ValueError(self.mode)

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g)
        c = child.id

        total_env = env + self.eps

        # Environment influence: parent line + older siblings -> newborn.
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(a, c)] += weight
        for s in older:
            contrib = self.nodes[s].g
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(s, c)] += weight

        # Newborn as UV-tail/backreaction for parent line.
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
            self.completed_parent_time[parent] = self.t

        return c

    def grow_level(self, frontier: List[int]) -> List[int]:
        out: List[int] = []
        for p in frontier:
            for k in range(1, 4):
                out.append(self.add_child(p, k))
        return out

    def run(self, max_level: int) -> None:
        frontier = [self.root]
        for _ in range(max_level):
            frontier = self.grow_level(frontier)

    def completed_parents(self) -> List[int]:
        return list(self.completed_parent_time.keys())

    def w(self, u: int, v: int) -> float:
        return self.directed_edges.get((u, v), 0.0)


def standard_sector_basis() -> np.ndarray:
    u1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
    u2 = np.array([1.0, 1.0, -2.0]) / math.sqrt(6)
    return np.vstack([u1, u2]).T


def kappa_matrix() -> np.ndarray:
    # swaps birth-order 1 and 3, fixes 2
    return np.array([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0]])


def local_raw_matrix(model: DynamicBirthModel, parent: int) -> Optional[np.ndarray]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return None
    M = np.zeros((3, 3), dtype=float)
    # column-source convention: M[target, source] = weight(source -> target)
    for j, u in enumerate(ch):
        for i, v in enumerate(ch):
            if u == v:
                continue
            M[i, j] = model.w(u, v)
    return M


def column_stochastic(M: np.ndarray) -> np.ndarray:
    P = M.copy().astype(float)
    for j in range(P.shape[1]):
        s = P[:, j].sum()
        if s > EPS:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    return P


def neutral_phasor(model: DynamicBirthModel, parent: int) -> complex:
    ch = model.nodes[parent].children
    vals = [model.nodes[c].g for c in ch]
    omega = cmath.exp(2j * math.pi / 3)
    return vals[0] + vals[1] * omega + vals[2] * omega**2


def skew_axis(A: np.ndarray) -> np.ndarray:
    # For skew matrix A = [[0,-z,y],[z,0,-x],[-y,x,0]]
    # vector a=(x,y,z) gives A v = a × v.
    return np.array([
        A[2, 1],
        A[0, 2],
        A[1, 0],
    ], dtype=float)


def eig_class(vals: np.ndarray, tol: float = 1e-9) -> str:
    return "complex_pair" if np.max(np.abs(np.imag(vals))) > tol else "real_or_degenerate"


def operator_diagnostics(M: np.ndarray) -> dict:
    B = standard_sector_basis()
    c = np.ones(3) / math.sqrt(3)
    K = kappa_matrix()
    K2 = B.T @ K @ B

    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)

    S2 = B.T @ S @ B
    A2 = B.T @ A @ B
    eig_M = np.linalg.eigvals(M)
    eig_S2 = np.linalg.eigvalsh(S2)

    axis = skew_axis(A)
    axis_norm = float(np.linalg.norm(axis))
    axis_align_const = 0.0
    if axis_norm > EPS:
        axis_align_const = float(abs(np.dot(axis / axis_norm, c)))

    # Invariance of the sum-zero sector under A.
    # A maps the sum-zero sector into itself iff c^T A B = 0.
    leakage = float(np.linalg.norm(c.T @ A @ B))
    A2_sq = A2 @ A2
    alpha2 = max(0.0, float(-np.trace(A2_sq) / 2.0))
    alpha = math.sqrt(alpha2)
    if alpha > EPS:
        J = A2 / alpha
        J2_resid = float(np.linalg.norm(J @ J + np.eye(2)))
        kappa_flip_resid = float(np.linalg.norm(K2 @ J @ K2.T + J))
        j_ok = bool(J2_resid < 1e-8)
        kappa_flip_ok = bool(kappa_flip_resid < 1e-8)
    else:
        J2_resid = float("inf")
        kappa_flip_resid = float("inf")
        j_ok = False
        kappa_flip_ok = False

    # S-based metric is only diagnosed. It is not accepted as physical G here.
    min_abs_S2 = float(np.min(np.abs(eig_S2))) if eig_S2.size else 0.0
    S2_pos = bool(np.all(eig_S2 > 1e-10))
    S2_neg = bool(np.all(eig_S2 < -1e-10))
    S2_definite = S2_pos or S2_neg

    return {
        "eig_class": eig_class(eig_M),
        "eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_M),
        "S2_eigvals": " ".join(f"{x:.9g}" for x in eig_S2),
        "S2_positive_definite": int(S2_pos),
        "S2_negative_definite": int(S2_neg),
        "S2_sign_definite": int(S2_definite),
        "S2_min_abs_eig": min_abs_S2,
        "skew_norm": float(np.linalg.norm(A)),
        "skew_axis_norm": axis_norm,
        "skew_axis_align_const": axis_align_const,
        "sumzero_leakage": leakage,
        "alpha": alpha,
        "J2_resid": J2_resid,
        "J2_ok": int(j_ok),
        "kappa_flip_resid": kappa_flip_resid,
        "kappa_flip_ok": int(kappa_flip_ok),
    }


def selected_forward_cycle_matrix(model: DynamicBirthModel, parent: int) -> Optional[np.ndarray]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return None
    c1, c2, c3 = ch
    M = np.zeros((3, 3), dtype=float)
    # use actual forward edge weights, with eps closure guard
    M[1, 0] = model.w(c1, c2) + EPS
    M[2, 1] = model.w(c2, c3) + EPS
    M[0, 2] = model.w(c3, c1) + EPS
    return M


def path_without_closure_matrix(model: DynamicBirthModel, parent: int) -> Optional[np.ndarray]:
    ch = model.nodes[parent].children
    if len(ch) != 3:
        return None
    c1, c2, c3 = ch
    M = np.zeros((3, 3), dtype=float)
    M[1, 0] = model.w(c1, c2) + EPS
    M[2, 1] = model.w(c2, c3) + EPS
    # no 3 -> 1 closure
    return M


def symmetrized_matrix(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def parent_row(model: DynamicBirthModel, parent: int, level: int) -> dict:
    raw = local_raw_matrix(model, parent)
    assert raw is not None
    P = column_stochastic(raw)
    fwd = column_stochastic(selected_forward_cycle_matrix(model, parent))
    path = path_without_closure_matrix(model, parent)
    sym_raw = symmetrized_matrix(raw)

    d_raw = operator_diagnostics(raw)
    d_P = operator_diagnostics(P)
    d_fwd = operator_diagnostics(fwd)
    d_path = operator_diagnostics(path)
    d_sym = operator_diagnostics(sym_raw)

    ch = model.nodes[parent].children
    z = neutral_phasor(model, parent)
    mean_g = sum(model.nodes[c].g for c in ch) / 3.0
    birth_times = [model.nodes[c].birth_time for c in ch]
    K = kappa_matrix()
    birth_vec = np.array(birth_times, dtype=float)
    birth_kappa = K @ birth_vec
    kappa_preserves_birth_order = bool(np.all(np.diff(birth_kappa) > 0))

    def prefix(prefix: str, d: dict) -> dict:
        return {f"{prefix}_{k}": v for k, v in d.items()}

    row = {
        "level": level,
        "mode": model.mode,
        "parent": parent,
        "parent_level": model.nodes[parent].level,
        "children": " ".join(map(str, ch)),
        "child_g": " ".join(f"{model.nodes[c].g:.12g}" for c in ch),
        "child_birth_g": " ".join(f"{model.nodes[c].birth_g:.12g}" for c in ch),
        "child_birth_times": " ".join(map(str, birth_times)),
        "neutral_abs": abs(z),
        "neutral_norm": abs(z) / mean_g,
        "neutral_phase_deg": math.degrees(cmath.phase(z)),
        "kappa_preserves_birth_order": int(kappa_preserves_birth_order),
    }
    row.update(prefix("raw", d_raw))
    row.update(prefix("markov", d_P))
    row.update(prefix("selected_fwd", d_fwd))
    row.update(prefix("path_no_closure", d_path))
    row.update(prefix("sym_raw", d_sym))
    return row


def summarize(rows: List[dict], model: DynamicBirthModel, level: int) -> dict:
    if not rows:
        return {
            "mode": model.mode,
            "level": level,
            "nodes": len(model.nodes),
            "completed_triples": 0,
        }

    def mean_key(k: str) -> float:
        return float(np.mean([float(r[k]) for r in rows]))

    def frac_key(k: str) -> float:
        return float(np.mean([float(r[k]) for r in rows]))

    def frac_class(prefix: str) -> float:
        return float(np.mean([1.0 if r[f"{prefix}_eig_class"] == "complex_pair" else 0.0 for r in rows]))

    return {
        "mode": model.mode,
        "level": level,
        "nodes": len(model.nodes),
        "completed_triples": len(rows),
        "mean_neutral_norm": mean_key("neutral_norm"),
        "mean_markov_alpha": mean_key("markov_alpha"),
        "mean_markov_J2_resid": mean_key("markov_J2_resid"),
        "mean_markov_kappa_flip_resid": mean_key("markov_kappa_flip_resid"),
        "mean_markov_sumzero_leakage": mean_key("markov_sumzero_leakage"),
        "mean_markov_axis_align_const": mean_key("markov_skew_axis_align_const"),
        "frac_markov_complex": frac_class("markov"),
        "frac_markov_J2_ok": frac_key("markov_J2_ok"),
        "frac_markov_kappa_flip_ok": frac_key("markov_kappa_flip_ok"),
        "frac_markov_S2_sign_definite": frac_key("markov_S2_sign_definite"),
        "frac_selected_fwd_J2_ok": frac_key("selected_fwd_J2_ok"),
        "frac_selected_fwd_kappa_flip_ok": frac_key("selected_fwd_kappa_flip_ok"),
        "frac_path_no_closure_complex": frac_class("path_no_closure"),
        "frac_sym_raw_complex": frac_class("sym_raw"),
        "frac_kappa_preserves_birth_order": frac_key("kappa_preserves_birth_order"),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_suite(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("linear", 0.22),
        ("log", 0.22),
        ("saturating", 0.90),
    ]
    all_level_rows: List[dict] = []
    report: List[str] = []
    for mode, alpha in configs:
        model = DynamicBirthModel(mode=mode, alpha_env=alpha)
        frontier = [model.root]
        level_rows: List[dict] = []
        parent_rows_all: List[dict] = []

        # level 0 summary
        all_level_rows.append(summarize([], model, 0))

        for level in range(1, max_level + 1):
            frontier = model.grow_level(frontier)
            rows = [parent_row(model, p, level) for p in model.completed_parents()]
            # record final parent rows only to keep CSV reasonable
            if level == max_level:
                parent_rows_all = rows
            s = summarize(rows, model, level)
            all_level_rows.append(s)

        write_csv(outdir / f"operator_parent_rows_{mode}.csv", parent_rows_all)

        final = [r for r in all_level_rows if r.get("mode") == mode and r.get("level") == max_level][0]
        root = parent_row(model, model.root, max_level)

        report.append(f"MODE {mode}")
        report.append(f"  final nodes={final['nodes']}, completed triples={final['completed_triples']}")
        report.append(f"  mean neutral norm={final['mean_neutral_norm']:.6f}")
        report.append(f"  frac Markov complex={final['frac_markov_complex']:.3f}")
        report.append(f"  frac Markov J2 ok={final['frac_markov_J2_ok']:.3f}")
        report.append(f"  mean Markov J2 residual={final['mean_markov_J2_resid']:.6e}")
        report.append(f"  mean Markov sum-zero leakage={final['mean_markov_sumzero_leakage']:.6e}")
        report.append(f"  mean Markov axis alignment with constant={final['mean_markov_axis_align_const']:.6f}")
        report.append(f"  frac Markov kappa flip ok={final['frac_markov_kappa_flip_ok']:.3f}")
        report.append(f"  control: frac selected forward J2 ok={final['frac_selected_fwd_J2_ok']:.3f}")
        report.append(f"  control: path no-closure complex={final['frac_path_no_closure_complex']:.3f}")
        report.append(f"  control: sym raw complex={final['frac_sym_raw_complex']:.3f}")
        report.append(f"  root child g={root['child_g']}")
        report.append(f"  root Markov eigs={root['markov_eigvals']}")
        report.append(f"  root Markov alpha={float(root['markov_alpha']):.9f}")
        report.append(f"  root Markov J2 residual={float(root['markov_J2_resid']):.6e}")
        report.append(f"  root Markov axis alignment={float(root['markov_skew_axis_align_const']):.9f}")
        report.append(f"  root Markov kappa flip residual={float(root['markov_kappa_flip_resid']):.6e}")
        report.append(f"  root Markov S2 eigs={root['markov_S2_eigvals']}")
        report.append(f"  root kappa preserves birth order={bool(root['kappa_preserves_birth_order'])}")
        report.append("")

    write_csv(outdir / "operator_level_summaries.csv", all_level_rows)
    summary = "\n".join(report)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=Path("effective_operator_j_out"))
    args = ap.parse_args()
    print(run_suite(args.max_level, args.outdir))


if __name__ == "__main__":
    main()
