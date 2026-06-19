#!/usr/bin/env python3
"""
Dynamic conductance scaling / sector-closure extrapolation test.

Question
--------
The user expects:
- Early ancestors receive smaller and smaller changes from later/deeper births.
- The largest conductance changes are local near the newly born child.
- For high-level extrapolation, we need a generalized growth/backreaction law.

This test distinguishes:
1. per-birth attenuation to old ancestors,
2. aggregate per-level backreaction after summing over exponentially many births,
3. local-vs-global backreaction concentration,
4. operator-sector leakage trends by level.

Core warning
------------
A kernel K(d)=1/d^2 makes a single remote birth weak, but it does not beat
the exponential shell count b^L. Aggregate root backreaction per level can
grow. For a finite/stable infinite-limit response, the ancestor kernel must
be normalized/damped strongly enough, e.g. K(d) decays faster than b^{-d}
or is shell-normalized.
"""

from __future__ import annotations

import argparse
import csv
import math
import cmath
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class ScalingModel:
    def __init__(
        self,
        kernel: str,
        max_level: int,
        mode: str = "log",
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_env_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
        exp_rho: float = 0.25,
    ):
        self.kernel = kernel
        self.max_level = max_level
        self.mode = mode
        self.b = branching
        self.base = base
        self.alpha_env = alpha_env
        self.ancestor_env_decay = ancestor_env_decay
        self.br_ancestor = br_ancestor
        self.br_sibling = br_sibling
        self.exp_rho = exp_rho

        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.t = 0

        # Local sibling directed weights only, keyed by parent and child order positions.
        # For a parent p, local_w[p][(i,j)] = weight child_i -> child_j.
        self.local_w: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))

        self.level_event_rows: List[dict] = []
        self.level_operator_rows: List[dict] = []
        root = self._new_node(None, 0, 0, 1.0)
        self.root = root.id

    def _new_node(self, parent: Optional[int], level: int, birth_order: int, birth_g: float) -> Node:
        n = Node(self.next_id, parent, level, birth_order, self.t, birth_g, birth_g)
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def kernel_value(self, d: int) -> float:
        if self.kernel == "inverse_square":
            return 1.0 / (d * d)
        if self.kernel == "exp_0p25":
            return 0.25 ** (d - 1)
        if self.kernel == "exp_0p40":
            return 0.40 ** (d - 1)
        if self.kernel == "critical_exp_1over3":
            return (1.0 / self.b) ** (d - 1)
        if self.kernel == "shell_norm_inverse_square":
            return 1.0 / ((d * d) * (self.b ** (d - 1)))
        raise ValueError(self.kernel)

    def parent_line(self, parent: int) -> List[int]:
        line: List[int] = []
        cur: Optional[int] = parent
        while cur is not None:
            line.append(cur)
            cur = self.nodes[cur].parent
        return line

    def birth_env_load(self, parent: int, older: List[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_env_decay ** (d - 1))
        for s in older:
            env += self.nodes[s].g
        return env

    def child_g_from_env(self, env: float) -> float:
        if self.mode == "linear":
            return self.base + self.alpha_env * env
        if self.mode == "log":
            return self.base + self.alpha_env * math.log1p(env)
        if self.mode == "saturating":
            return self.base + self.alpha_env * env / (1.0 + env)
        raise ValueError(self.mode)

    def add_child(self, parent: int, order: int, current_birth_level: int, accum: dict) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env = self.birth_env_load(parent, older)
        bg = self.child_g_from_env(env)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, bg)
        c = child.id

        # Local older siblings -> new child and new child -> older siblings.
        # Store only child-order indices for local operator diagnostics.
        for s in older:
            i = self.nodes[s].birth_order
            j = order
            # environment-influence share as simple local proportional surrogate
            self.local_w[parent][(i, j)] += self.alpha_env * self.nodes[s].g / (env + EPS) * bg
        for s in older:
            i = order
            j = self.nodes[s].birth_order
            delta = self.br_sibling * bg
            self.nodes[s].g += delta
            self.local_w[parent][(i, j)] += delta

        # Ancestor backreaction: newborn acts as UV-tail for parent line.
        total_ancestor_delta = 0.0
        root_delta = 0.0
        immediate_parent_delta = 0.0

        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * bg * self.kernel_value(d)
            self.nodes[a].g += delta
            total_ancestor_delta += delta
            if d == 1:
                immediate_parent_delta = delta
            if a == self.root:
                root_delta = delta
            # distribution by target level
            key = f"target_level_{self.nodes[a].level}_delta"
            accum[key] = accum.get(key, 0.0) + delta

        accum["births"] += 1
        accum["root_delta_sum"] += root_delta
        accum["root_delta_max"] = max(accum["root_delta_max"], root_delta)
        accum["root_delta_min_nonzero"] = min(accum["root_delta_min_nonzero"], root_delta if root_delta > 0 else accum["root_delta_min_nonzero"])
        accum["immediate_parent_delta_sum"] += immediate_parent_delta
        accum["immediate_parent_delta_max"] = max(accum["immediate_parent_delta_max"], immediate_parent_delta)
        accum["ancestor_delta_sum"] += total_ancestor_delta
        accum["birth_g_sum"] += bg
        accum["birth_g_max"] = max(accum["birth_g_max"], bg)
        return c

    def grow_one_level(self, frontier: List[int], level: int) -> List[int]:
        accum = {
            "births": 0,
            "root_delta_sum": 0.0,
            "root_delta_max": 0.0,
            "root_delta_min_nonzero": float("inf"),
            "immediate_parent_delta_sum": 0.0,
            "immediate_parent_delta_max": 0.0,
            "ancestor_delta_sum": 0.0,
            "birth_g_sum": 0.0,
            "birth_g_max": 0.0,
        }
        new_frontier: List[int] = []
        for p in frontier:
            for k in range(1, self.b + 1):
                new_frontier.append(self.add_child(p, k, level, accum))

        births = max(1, accum["births"])
        root_delta_min = accum["root_delta_min_nonzero"]
        if root_delta_min == float("inf"):
            root_delta_min = 0.0

        row = {
            "kernel": self.kernel,
            "mode": self.mode,
            "level": level,
            "nodes": len(self.nodes),
            "births_in_level": accum["births"],
            "root_g": self.nodes[self.root].g,
            "root_delta_sum": accum["root_delta_sum"],
            "root_delta_mean_per_birth": accum["root_delta_sum"] / births,
            "root_delta_max": accum["root_delta_max"],
            "root_delta_min_nonzero": root_delta_min,
            "immediate_parent_delta_sum": accum["immediate_parent_delta_sum"],
            "immediate_parent_delta_mean_per_birth": accum["immediate_parent_delta_sum"] / births,
            "immediate_parent_delta_max": accum["immediate_parent_delta_max"],
            "root_to_local_mean_ratio": (accum["root_delta_sum"] / births) / (accum["immediate_parent_delta_sum"] / births + EPS),
            "ancestor_delta_sum": accum["ancestor_delta_sum"],
            "local_fraction_of_ancestor_delta": accum["immediate_parent_delta_sum"] / (accum["ancestor_delta_sum"] + EPS),
            "birth_g_mean": accum["birth_g_sum"] / births,
            "birth_g_max": accum["birth_g_max"],
        }
        # Add aggregate target-level fractions for the first few levels.
        for tl in range(0, min(level, 6) + 1):
            row[f"target_level_{tl}_delta"] = accum.get(f"target_level_{tl}_delta", 0.0)
        self.level_event_rows.append(row)
        return new_frontier

    def local_matrix_for_parent(self, parent: int) -> Optional[np.ndarray]:
        if len(self.nodes[parent].children) != self.b:
            return None
        M = np.zeros((3, 3), dtype=float)
        w = self.local_w[parent]
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    continue
                # column-source: source order i -> target order j
                M[j - 1, i - 1] = w.get((i, j), 0.0)
        return M

    @staticmethod
    def column_stochastic(M: np.ndarray) -> np.ndarray:
        P = M.copy()
        for j in range(3):
            s = P[:, j].sum()
            if s > EPS:
                P[:, j] /= s
            else:
                P[j, j] = 1.0
        return P

    @staticmethod
    def standard_basis() -> np.ndarray:
        u1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
        u2 = np.array([1.0, 1.0, -2.0]) / math.sqrt(6)
        return np.vstack([u1, u2]).T

    @staticmethod
    def skew_axis(A: np.ndarray) -> np.ndarray:
        return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)

    def operator_diag_for_parent(self, parent: int) -> Optional[dict]:
        M = self.local_matrix_for_parent(parent)
        if M is None:
            return None
        P = self.column_stochastic(M)
        A = 0.5 * (P - P.T)
        B = self.standard_basis()
        c = np.ones(3) / math.sqrt(3)
        A2 = B.T @ A @ B
        alpha2 = max(0.0, float(-np.trace(A2 @ A2) / 2.0))
        alpha = math.sqrt(alpha2)
        if alpha > EPS:
            J = A2 / alpha
            J2_resid = float(np.linalg.norm(J @ J + np.eye(2)))
        else:
            J2_resid = float("inf")
        leakage = float(np.linalg.norm(c.T @ A @ B))
        axis = self.skew_axis(A)
        axis_norm = float(np.linalg.norm(axis))
        axis_align = float(abs(np.dot(axis / axis_norm, c))) if axis_norm > EPS else 0.0
        ev = np.linalg.eigvals(P)
        complex_pair = float(np.max(np.abs(np.imag(ev))) > 1e-9)
        return {
            "alpha": alpha,
            "J2_resid": J2_resid,
            "leakage": leakage,
            "axis_align_const": axis_align,
            "complex_pair": complex_pair,
        }

    def summarize_operator_by_parent_level(self, level: int) -> List[dict]:
        groups: Dict[int, List[dict]] = defaultdict(list)
        for n in self.nodes.values():
            d = self.operator_diag_for_parent(n.id)
            if d is not None:
                groups[n.level].append(d)
        rows = []
        for pl, ds in sorted(groups.items()):
            rows.append({
                "kernel": self.kernel,
                "mode": self.mode,
                "global_level": level,
                "parent_level": pl,
                "count": len(ds),
                "mean_alpha": float(np.mean([x["alpha"] for x in ds])),
                "mean_J2_resid": float(np.mean([x["J2_resid"] for x in ds])),
                "mean_leakage": float(np.mean([x["leakage"] for x in ds])),
                "mean_axis_align_const": float(np.mean([x["axis_align_const"] for x in ds])),
                "frac_complex_pair": float(np.mean([x["complex_pair"] for x in ds])),
            })
        return rows

    def run(self, max_level: int) -> None:
        frontier = [self.root]
        for level in range(1, max_level + 1):
            frontier = self.grow_one_level(frontier, level)
            # Operator summaries after each level; cheap enough up to L10.
            self.level_operator_rows.extend(self.summarize_operator_by_parent_level(level))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def slope_last(xs: List[float]) -> float:
    # log growth factor from last adjacent nonzero values.
    vals = [x for x in xs if x > 0]
    if len(vals) < 2:
        return float("nan")
    return vals[-1] / vals[-2]


def run_suite(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    kernels = [
        "inverse_square",
        "critical_exp_1over3",
        "exp_0p40",
        "exp_0p25",
        "shell_norm_inverse_square",
    ]
    all_events = []
    all_ops = []
    lines = []
    for kernel in kernels:
        m = ScalingModel(kernel=kernel, max_level=max_level, mode="log")
        m.run(max_level)
        all_events.extend(m.level_event_rows)
        all_ops.extend(m.level_operator_rows)
        root_sums = [r["root_delta_sum"] for r in m.level_event_rows]
        root_means = [r["root_delta_mean_per_birth"] for r in m.level_event_rows]
        local_means = [r["immediate_parent_delta_mean_per_birth"] for r in m.level_event_rows]
        final = m.level_event_rows[-1]
        final_op = [r for r in m.level_operator_rows if r["global_level"] == max_level]
        # Use deepest completed parents for local operator trend.
        deepest = max(r["parent_level"] for r in final_op) if final_op else -1
        deep_rows = [r for r in final_op if r["parent_level"] == deepest]
        deep = deep_rows[0] if deep_rows else None
        lines.append(f"KERNEL {kernel}")
        lines.append(f"  final level={max_level}, nodes={final['nodes']}, births={final['births_in_level']}")
        lines.append(f"  root_g={final['root_g']:.6f}")
        lines.append(f"  root_delta_sum={final['root_delta_sum']:.6e}")
        lines.append(f"  root_delta_mean_per_birth={final['root_delta_mean_per_birth']:.6e}")
        lines.append(f"  immediate_parent_delta_mean_per_birth={final['immediate_parent_delta_mean_per_birth']:.6e}")
        lines.append(f"  root/local mean ratio={final['root_to_local_mean_ratio']:.6e}")
        lines.append(f"  local fraction of ancestor delta={final['local_fraction_of_ancestor_delta']:.6f}")
        lines.append(f"  last ratio root_delta_sum L/L-1={slope_last(root_sums):.6f}")
        lines.append(f"  last ratio root_delta_mean_per_birth L/L-1={slope_last(root_means):.6f}")
        lines.append(f"  last ratio local_mean_per_birth L/L-1={slope_last(local_means):.6f}")
        if deep:
            lines.append(
                f"  deepest parent_level={deepest}: mean leakage={deep['mean_leakage']:.6e}, "
                f"axis_align={deep['mean_axis_align_const']:.6f}, frac complex={deep['frac_complex_pair']:.3f}"
            )
        lines.append("")
        write_csv(outdir / f"events_{kernel}.csv", m.level_event_rows)
        write_csv(outdir / f"operators_{kernel}.csv", m.level_operator_rows)
    write_csv(outdir / "all_event_summaries.csv", all_events)
    write_csv(outdir / "all_operator_summaries.csv", all_ops)
    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=10)
    ap.add_argument("--outdir", type=Path, default=Path("conductance_scaling_out"))
    args = ap.parse_args()
    print(run_suite(args.max_level, args.outdir))


if __name__ == "__main__":
    main()
