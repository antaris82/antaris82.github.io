#!/usr/bin/env python3
"""
Dynamic birth monodromy test for the CNNA/NGF provenance-growth picture.

Purpose
-------
This is a Python diagnostic surrogate, not a Lean theorem and not a physics claim.

It extends test_dynamic_birth_conductance.py by separating three questions:

1. Conductance/neutral imbalance:
   Do sequential births create unequal sibling response weights?

2. Log-circulation / non-coboundary surrogate:
   For a completed sibling triple (1,2,3), compare the directed forward cycle
       1->2, 2->3, 3->1
   with the reverse cycle
       1->3, 3->2, 2->1.
   The log-ratio is a gauge-invariant circulation around the local triangle.
   Nonzero log-ratio is a necessary "not pure gradient on this cycle" test.

3. Monodromy classification:
   a) The selected forward-cycle transport operator is a weighted Z3-cycle.
      If the closure 3->1 is actually derived, this operator carries a 2D
      rotation sector.
   b) The full local Markov transport operator uses all directed influence
      weights among the siblings. If this full operator already has a complex
      pair, then the response dynamics itself carries a rotational sector.
      If it is real/degenerate, the selected Z3 cycle is only a candidate
      closure, not a conclusion from the full local response.

Important guardrail
-------------------
Conductance/response data are not geometric edge lengths. Equal effective
geometry lengths remain compatible with changing response weights.
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


class DynamicBirthMonodromyModel:
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
        self.event_rows: List[dict] = []
        self.level_rows: List[dict] = []
        self.triple_rows: List[dict] = []

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

    def neutral_phasor_values(self, vals: List[float]) -> complex:
        omega = cmath.exp(2j * math.pi / 3)
        return vals[0] + vals[1] * omega + vals[2] * (omega ** 2)

    def neutral_for_parent(self, parent: int, current: bool = True) -> Optional[complex]:
        ch = self.nodes[parent].children
        if len(ch) != 3:
            return None
        vals = [self.nodes[c].g if current else self.nodes[c].birth_g for c in ch]
        return self.neutral_phasor_values(vals)

    def w(self, u: int, v: int) -> float:
        return self.directed_edges.get((u, v), 0.0)

    def local_pair_weights(self, parent: int) -> Optional[dict]:
        ch = self.nodes[parent].children
        if len(ch) != 3:
            return None
        c1, c2, c3 = ch
        return {
            "w12": self.w(c1, c2),
            "w23": self.w(c2, c3),
            "w31": self.w(c3, c1),
            "w13": self.w(c1, c3),
            "w32": self.w(c3, c2),
            "w21": self.w(c2, c1),
        }

    @staticmethod
    def transport_matrix_from_weights(
        w12: float, w23: float, w31: float, w13: float = 0.0, w32: float = 0.0, w21: float = 0.0
    ) -> np.ndarray:
        # Column-source convention: A[target, source] = weight(source -> target).
        A = np.zeros((3, 3), dtype=float)
        A[1, 0] = w12
        A[2, 1] = w23
        A[0, 2] = w31
        A[2, 0] = w13
        A[1, 2] = w32
        A[0, 1] = w21
        return A

    @staticmethod
    def column_stochastic(A: np.ndarray) -> np.ndarray:
        B = A.copy().astype(float)
        for j in range(B.shape[1]):
            s = B[:, j].sum()
            if s > 0:
                B[:, j] /= s
            else:
                B[j, j] = 1.0
        return B

    @staticmethod
    def eig_class(vals: np.ndarray, tol: float = 1e-9) -> str:
        im = np.max(np.abs(np.imag(vals)))
        if im > tol:
            return "complex_pair"
        return "real_or_degenerate"

    @staticmethod
    def standard_sector_basis() -> np.ndarray:
        u1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
        u2 = np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)
        return np.vstack([u1, u2]).T

    @staticmethod
    def sector_skew_J(R2: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        A = R2 - R2.T
        n = np.linalg.norm(A)
        if n < 1e-12:
            return A, 0.0, False
        J = A / n * math.sqrt(2.0)
        ok = np.allclose(J @ J, -np.eye(2), atol=1e-8)
        return J, n, ok

    def triple_monodromy(self, parent: int) -> Optional[dict]:
        ch = self.nodes[parent].children
        if len(ch) != 3:
            return None
        c1, c2, c3 = ch
        weights = self.local_pair_weights(parent)
        assert weights is not None

        e = self.eps
        w12 = weights["w12"] + e
        w23 = weights["w23"] + e
        w31 = weights["w31"] + e
        w13 = weights["w13"] + e
        w32 = weights["w32"] + e
        w21 = weights["w21"] + e

        forward_product = w12 * w23 * w31
        reverse_product = w13 * w32 * w21
        log_circ = math.log(forward_product / reverse_product)

        A_fwd = self.transport_matrix_from_weights(w12, w23, w31)
        A_rev = self.transport_matrix_from_weights(0.0, 0.0, 0.0, w13, w32, w21)
        A_full = self.transport_matrix_from_weights(w12, w23, w31, w13, w32, w21)
        A_sym = 0.5 * (A_full + A_full.T)

        # Birth-order path without the closure edge 3->1.
        A_path = self.transport_matrix_from_weights(w12, w23, 0.0, 0.0, 0.0, 0.0)

        P_fwd = self.column_stochastic(A_fwd)
        P_full = self.column_stochastic(A_full)
        P_sym = self.column_stochastic(A_sym)
        P_path = self.column_stochastic(A_path)

        eig_fwd = np.linalg.eigvals(P_fwd)
        eig_full = np.linalg.eigvals(P_full)
        eig_A_full = np.linalg.eigvals(A_full)
        eig_sym = np.linalg.eigvals(A_sym)
        eig_P_sym = np.linalg.eigvals(P_sym)
        eig_path = np.linalg.eigvals(A_path)
        eig_P_path = np.linalg.eigvals(P_path)

        B = self.standard_sector_basis()
        R2_fwd = B.T @ P_fwd @ B
        J_fwd, skew_norm_fwd, j_ok_fwd = self.sector_skew_J(R2_fwd)

        # κ swaps child 1 and child 3, child 2 fixed.
        K = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        P_kappa = K @ P_fwd @ K.T
        R2_kappa = B.T @ P_kappa @ B
        J_kappa, skew_norm_kappa, j_ok_kappa = self.sector_skew_J(R2_kappa)
        kappa_flips_J = bool(j_ok_fwd and j_ok_kappa and np.allclose(J_kappa, -J_fwd, atol=1e-8))

        # Birth history preservation by κ.
        birth = np.array([self.nodes[c1].birth_time, self.nodes[c2].birth_time, self.nodes[c3].birth_time], dtype=float)
        birth_kappa = K @ birth
        kappa_preserves_birth_order = bool(np.all(np.diff(birth_kappa) > 0))

        z_current = self.neutral_for_parent(parent, current=True)
        z_birth = self.neutral_for_parent(parent, current=False)
        mean_g = sum(self.nodes[x].g for x in ch) / 3.0
        mean_bg = sum(self.nodes[x].birth_g for x in ch) / 3.0

        row = {
            "mode": self.mode,
            "time": self.t,
            "parent": parent,
            "parent_level": self.nodes[parent].level,
            "children": f"{c1} {c2} {c3}",
            "child_birth_times": f"{int(birth[0])} {int(birth[1])} {int(birth[2])}",
            "child_birth_g": " ".join(f"{self.nodes[x].birth_g:.12g}" for x in ch),
            "child_current_g": " ".join(f"{self.nodes[x].g:.12g}" for x in ch),
            "w12": weights["w12"],
            "w23": weights["w23"],
            "w31": weights["w31"],
            "w13": weights["w13"],
            "w32": weights["w32"],
            "w21": weights["w21"],
            "forward_product": forward_product,
            "reverse_product": reverse_product,
            "log_circulation_forward_vs_reverse": log_circ,
            "neutral_abs_current": abs(z_current) if z_current is not None else 0.0,
            "neutral_norm_current": abs(z_current) / mean_g if z_current is not None else 0.0,
            "neutral_phase_current_deg": math.degrees(cmath.phase(z_current)) if z_current is not None else 0.0,
            "neutral_abs_birth": abs(z_birth) if z_birth is not None else 0.0,
            "neutral_norm_birth": abs(z_birth) / mean_bg if z_birth is not None else 0.0,
            "fwd_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_fwd),
            "full_markov_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_full),
            "full_raw_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_A_full),
            "sym_raw_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_sym),
            "sym_markov_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_P_sym),
            "path_raw_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_path),
            "path_markov_eigvals": " ".join(f"{z.real:.9g}{z.imag:+.9g}j" for z in eig_P_path),
            "fwd_eig_class": self.eig_class(eig_fwd),
            "full_markov_eig_class": self.eig_class(eig_full),
            "full_raw_eig_class": self.eig_class(eig_A_full),
            "sym_raw_eig_class": self.eig_class(eig_sym),
            "sym_markov_eig_class": self.eig_class(eig_P_sym),
            "path_raw_eig_class": self.eig_class(eig_path),
            "path_markov_eig_class": self.eig_class(eig_P_path),
            "fwd_sector_skew_norm": skew_norm_fwd,
            "fwd_J_squared_minus_I": int(j_ok_fwd),
            "kappa_flips_fwd_J": int(kappa_flips_J),
            "kappa_preserves_birth_order": int(kappa_preserves_birth_order),
        }
        return row

    def completed_parents(self) -> List[int]:
        return [n.id for n in self.nodes.values() if len(n.children) == 3]

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env_load = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env_load)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g)
        c = child.id

        # Environment influence: existing parent-line and older siblings -> newborn.
        total_env = env_load + self.eps
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(a, c)] += weight
        for s in older:
            contrib = self.nodes[s].g
            weight = self.alpha_env * contrib / total_env * birth_g
            self.directed_edges[(s, c)] += weight

        # Backreaction: newborn acts as UV-tail for parent line.
        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * birth_g / (d * d)
            self.nodes[a].g += delta
            self.directed_edges[(c, a)] += delta

        # Backreaction to older siblings.
        for s in older:
            delta = self.br_sibling * birth_g
            self.nodes[s].g += delta
            self.directed_edges[(c, s)] += delta

        # Event snapshot.
        children = self.nodes[parent].children
        event = {
            "mode": self.mode,
            "t": self.t,
            "parent": parent,
            "child": c,
            "child_level": child.level,
            "child_order": order,
            "older_siblings": " ".join(map(str, older)),
            "env_load": env_load,
            "child_birth_g": birth_g,
            "parent_g_after": self.nodes[parent].g,
            "sibling_current_g": " ".join(f"{self.nodes[x].g:.12g}" for x in children),
            "triple_completed": int(len(children) == 3),
        }
        if len(children) == 3:
            m = self.triple_monodromy(parent)
            assert m is not None
            self.triple_rows.append(m)
            event.update(
                {
                    "triple_log_circulation": m["log_circulation_forward_vs_reverse"],
                    "triple_neutral_norm_current": m["neutral_norm_current"],
                    "triple_full_markov_eig_class": m["full_markov_eig_class"],
                    "triple_kappa_flips_fwd_J": m["kappa_flips_fwd_J"],
                }
            )
        self.event_rows.append(event)
        return c

    def grow_level(self, frontier: List[int]) -> List[int]:
        next_frontier: List[int] = []
        for p in frontier:
            for k in range(1, 4):
                next_frontier.append(self.add_child(p, k))
        return next_frontier

    def global_undirected_h1_rank(self) -> Tuple[int, int, int]:
        und = set()
        for (u, v), w in self.directed_edges.items():
            if u != v and w > 0:
                und.add(tuple(sorted((u, v))))
        V = len(self.nodes)
        E = len(und)
        adj: Dict[int, List[int]] = defaultdict(list)
        for u, v in und:
            adj[u].append(v)
            adj[v].append(u)
        seen = set()
        comps = 0
        for n in self.nodes:
            if n in seen:
                continue
            comps += 1
            stack = [n]
            seen.add(n)
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
        return E - V + comps, E, comps

    def level_summary(self, level: int) -> dict:
        triples = [r for r in self.triple_rows if self.nodes[int(r["parent"])].level < level]
        # Since completed triples remain active and conductances keep changing, recompute current monodromy
        # for all completed parents at level summary time.
        current_rows = []
        for p in self.completed_parents():
            m = self.triple_monodromy(p)
            if m is not None:
                current_rows.append(m)

        def mean(vals: List[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        def frac_complex(key: str) -> float:
            if not current_rows:
                return 0.0
            return sum(1 for r in current_rows if r[key] == "complex_pair") / len(current_rows)

        h1, und_e, comps = self.global_undirected_h1_rank()
        gs = [n.g for n in self.nodes.values()]
        logcs = [float(r["log_circulation_forward_vs_reverse"]) for r in current_rows]
        neutral_norms = [float(r["neutral_norm_current"]) for r in current_rows]
        row = {
            "mode": self.mode,
            "level": level,
            "time": self.t,
            "nodes": len(self.nodes),
            "completed_triples": len(current_rows),
            "directed_edges": len(self.directed_edges),
            "undirected_edges": und_e,
            "undirected_H1_support_rank": h1,
            "mean_log_circulation": mean(logcs),
            "mean_abs_log_circulation": mean([abs(x) for x in logcs]),
            "min_log_circulation": min(logcs) if logcs else 0.0,
            "max_log_circulation": max(logcs) if logcs else 0.0,
            "mean_neutral_norm_current": mean(neutral_norms),
            "frac_forward_cycle_complex": frac_complex("fwd_eig_class"),
            "frac_full_markov_complex": frac_complex("full_markov_eig_class"),
            "frac_full_raw_complex": frac_complex("full_raw_eig_class"),
            "frac_sym_raw_complex": frac_complex("sym_raw_eig_class"),
            "frac_sym_markov_complex": frac_complex("sym_markov_eig_class"),
            "frac_path_raw_complex": frac_complex("path_raw_eig_class"),
            "frac_path_markov_complex": frac_complex("path_markov_eig_class"),
            "frac_kappa_flips_forward_J": mean([float(r["kappa_flips_fwd_J"]) for r in current_rows]),
            "frac_kappa_preserves_birth_order": mean([float(r["kappa_preserves_birth_order"]) for r in current_rows]),
            "min_g": min(gs),
            "max_g": max(gs),
            "mean_g": float(np.mean(gs)),
        }
        self.level_rows.append(row)
        return row

    def run(self, max_level: int) -> None:
        frontier = [self.root]
        self.level_summary(0)
        for level in range(1, max_level + 1):
            frontier = self.grow_level(frontier)
            self.level_summary(level)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("linear", 0.22),
        ("log", 0.22),
        ("saturating", 0.90),
    ]

    all_level_rows: List[dict] = []
    lines: List[str] = []
    for mode, alpha in configs:
        model = DynamicBirthMonodromyModel(mode=mode, alpha_env=alpha)
        model.run(max_level)

        write_csv(outdir / f"events_{mode}.csv", model.event_rows)
        write_csv(outdir / f"triples_{mode}.csv", model.triple_rows)
        write_csv(outdir / f"levels_{mode}.csv", model.level_rows)
        all_level_rows.extend(model.level_rows)

        root_m = model.triple_monodromy(model.root)
        final = model.level_rows[-1]
        lines.append(f"MODE {mode}")
        lines.append(f"  final nodes={final['nodes']}, completed triples={final['completed_triples']}")
        lines.append(f"  support H1-rank={final['undirected_H1_support_rank']}, directed edges={final['directed_edges']}")
        lines.append(f"  mean log-circulation={final['mean_log_circulation']:.6f}")
        lines.append(f"  mean neutral norm={final['mean_neutral_norm_current']:.6f}")
        lines.append(f"  frac forward-cycle complex={final['frac_forward_cycle_complex']:.3f}")
        lines.append(f"  frac full-local Markov complex={final['frac_full_markov_complex']:.3f}")
        lines.append(f"  frac full-raw local complex={final['frac_full_raw_complex']:.3f}")
        lines.append(f"  control frac sym-raw complex={final['frac_sym_raw_complex']:.3f}")
        lines.append(f"  control frac path-raw complex={final['frac_path_raw_complex']:.3f}")
        if root_m:
            lines.append("  root:")
            lines.append(f"    current g: {root_m['child_current_g']}")
            lines.append(f"    log-circulation: {root_m['log_circulation_forward_vs_reverse']:.6f}")
            lines.append(f"    neutral norm: {root_m['neutral_norm_current']:.6f}")
            lines.append(f"    forward eig class: {root_m['fwd_eig_class']}")
            lines.append(f"    full Markov eig class: {root_m['full_markov_eig_class']}")
            lines.append(f"    full Markov eigs: {root_m['full_markov_eigvals']}")
            lines.append(f"    sym raw eig class: {root_m['sym_raw_eig_class']}")
            lines.append(f"    path raw eig class: {root_m['path_raw_eig_class']}")
            lines.append(f"    kappa flips selected forward J: {bool(root_m['kappa_flips_fwd_J'])}")
            lines.append(f"    kappa preserves birth order: {bool(root_m['kappa_preserves_birth_order'])}")
        lines.append("")

    write_csv(outdir / "all_level_summaries.csv", all_level_rows)
    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=Path("dynamic_birth_monodromy_out"))
    args = ap.parse_args()
    summary = run_suite(args.max_level, args.outdir)
    print(summary)


if __name__ == "__main__":
    main()
