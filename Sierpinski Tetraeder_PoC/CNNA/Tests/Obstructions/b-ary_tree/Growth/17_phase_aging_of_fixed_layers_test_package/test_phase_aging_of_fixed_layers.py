#!/usr/bin/env python3
"""
Phase aging of fixed layers.

Question:
    If a fixed parent level is observed while the tower grows from L to L+k,
    does its residual continue to change/relax, or is it frozen after local
    completion?

Important semantic split:
1. stored_response phase:
   The local response operator is the historically accumulated local_w matrix.
   Once a parent has all three children, no later descendant birth changes this
   local_w block in the current model. Therefore this phase should freeze.

2. refreshed_conductance phase:
   The current conductances g of the three children can continue to change as
   their subtrees grow, because descendant births backreact along parent lines.
   This is not the same operator as stored_response; it is a current-state
   snapshot diagnostic.

This test reports both.
"""

from __future__ import annotations

import argparse, csv, math, time
from collections import defaultdict
from dataclasses import dataclass, field
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

class ShellGrowth:
    def __init__(self):
        self.b = 3
        self.base = 1.0
        self.alpha_env = 0.22
        self.ancestor_env_decay = 0.55
        self.br_ancestor = 0.045
        self.br_sibling = 0.035
        self.nodes: Dict[int, Node] = {}
        self.local_w: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
        self.next_id = 0
        self.t = 0
        root = self._new_node(None, 0, 0, 1.0)
        self.root = root.id

    def _new_node(self, parent, level, birth_order, birth_g):
        n = Node(self.next_id, parent, level, birth_order, self.t, birth_g, birth_g)
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def kernel(self, d: int) -> float:
        return 1.0 / ((self.b ** (d - 1)) * d * d)

    def parent_line(self, parent: int) -> List[int]:
        out = []
        cur = parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def birth_env(self, parent: int, older: List[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_env_decay ** (d - 1))
        for s in older:
            env += self.nodes[s].g
        return env

    def child_g(self, env: float) -> float:
        return self.base + self.alpha_env * math.log1p(env)

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env = self.birth_env(parent, older)
        bg = self.child_g(env)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, bg)

        for s in older:
            i = self.nodes[s].birth_order
            j = order
            self.local_w[parent][(i, j)] += self.alpha_env * self.nodes[s].g / (env + EPS) * bg

        for d, a in enumerate(self.parent_line(parent), start=1):
            self.nodes[a].g += self.br_ancestor * bg * self.kernel(d)

        for s in older:
            i = order
            j = self.nodes[s].birth_order
            delta = self.br_sibling * bg
            self.nodes[s].g += delta
            self.local_w[parent][(i, j)] += delta

        return child.id

    def grow_one_level(self, frontier: List[int]) -> List[int]:
        out = []
        for p in frontier:
            for k in range(1, 4):
                out.append(self.add_child(p, k))
        return out

    def local_matrix(self, parent: int) -> Optional[np.ndarray]:
        if len(self.nodes[parent].children) != 3:
            return None
        M = np.zeros((3, 3), float)
        w = self.local_w[parent]
        for i in range(1, 4):
            for j in range(1, 4):
                if i != j:
                    M[j - 1, i - 1] = w.get((i, j), 0.0)
        return M

def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0

def angle_diff(a: float, b: float) -> float:
    return wrap_deg(a - b)

def mean(vals):
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")

def std(vals):
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.std(xs)) if xs else float("nan")

def perc(vals, q):
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.percentile(xs, q)) if xs else float("nan")

def column_stochastic(M):
    P = M.copy().astype(float)
    for j in range(3):
        s = P[:, j].sum()
        if s > EPS:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    return P

def skew_axis(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]], float)

def plane_basis(axis):
    a = axis / (np.linalg.norm(axis) + EPS)
    h = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, h))) > 0.9:
        h = np.array([0.0, 1.0, 0.0])
    u = h - np.dot(h, a) * a
    u = u / (np.linalg.norm(u) + EPS)
    v = np.cross(a, u)
    v = v / (np.linalg.norm(v) + EPS)
    return np.vstack([u, v]).T

def polar_so2(A2):
    U, s, Vt = np.linalg.svd(A2)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q

def angle_so2_deg(Q):
    cosv = float((Q[0, 0] + Q[1, 1]) / 2.0)
    sinv = float((Q[1, 0] - Q[0, 1]) / 2.0)
    return math.degrees(math.atan2(sinv, cosv))

def stored_response_phase(model: ShellGrowth, parent: int) -> Optional[float]:
    M = model.local_matrix(parent)
    if M is None:
        return None
    P = column_stochastic(M)
    A = 0.5 * (P - P.T)
    axis = skew_axis(A)
    if np.linalg.norm(axis) <= EPS:
        return None
    c = np.ones(3) / math.sqrt(3)
    if np.dot(axis, c) < 0:
        axis = -axis
    axis = axis / (np.linalg.norm(axis) + EPS)
    B = plane_basis(axis)
    R2 = B.T @ P @ B
    Q = polar_so2(R2)
    return angle_so2_deg(Q)

def refreshed_conductance_snapshot(model: ShellGrowth, parent: int) -> Optional[dict]:
    if len(model.nodes[parent].children) != 3:
        return None
    ch = model.nodes[parent].children
    vals = np.array([model.nodes[c].g for c in ch], float)
    omega = complex(math.cos(2*math.pi/3), math.sin(2*math.pi/3))
    z = vals[0] + vals[1]*omega + vals[2]*(omega**2)
    phase = math.degrees(math.atan2(z.imag, z.real))
    mean_g = float(np.mean(vals))
    norm = abs(z) / (mean_g + EPS)
    rel_std = float(np.std(vals) / (mean_g + EPS))
    spread = float((np.max(vals) - np.min(vals)) / (mean_g + EPS))
    return {
        "neutral_phase": phase,
        "neutral_norm": norm,
        "g_rel_std": rel_std,
        "g_spread": spread,
        "g1": float(vals[0]),
        "g2": float(vals[1]),
        "g3": float(vals[2]),
    }

def completed_parent_ids(model: ShellGrowth) -> List[int]:
    return [n.id for n in model.nodes.values() if len(n.children) == 3]

def build_loops(model: ShellGrowth, phase_map: Dict[int, float]):
    loops = []
    for p in phase_map:
        cs = [c for c in model.nodes[p].children if c in phase_map]
        if len(cs) == 3:
            c1, c2, c3 = cs
            loops.append(("sibling_cycle", model.nodes[p].level + 1, p, [c1, c2, c3]))
            loops.append(("parent_child_ring", model.nodes[p].level, p, [p, c1, c2, c3]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, p, [p, c1, c2]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, p, [p, c2, c3]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, p, [p, c3, c1]))
    return loops

def level_centered_loop_residuals(model: ShellGrowth, phase_map: Dict[int, float]) -> List[dict]:
    bg = defaultdict(list)
    for p, th in phase_map.items():
        bg[model.nodes[p].level].append(th)
    bg = {k: mean(v) for k, v in bg.items()}
    rows = []
    for mode, loop_level, base, loop in build_loops(model, phase_map):
        vals = [phase_map[u] for u in loop]
        bgs = [bg[model.nodes[u].level] for u in loop]
        res = wrap_deg(sum(vals) - sum(bgs))
        rows.append({
            "loop_mode": mode,
            "loop_level": loop_level,
            "base": base,
            "base_level": model.nodes[base].level,
            "loop_len": len(loop),
            "residual": res,
            "abs_residual": abs(res),
        })
    return rows

def summarize_groups(rows, keys, value_key):
    groups = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        d = {keys[i]: k[i] for i in range(len(keys))}
        vals = [r[value_key] for r in rs]
        d.update(count=len(rs), mean=mean(vals), std=std(vals), p95=perc(vals, 95))
        return_count = len(rs)
        out.append(d)
    return out

def run(max_level: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    model = ShellGrowth()
    frontier = [model.root]

    baseline: Dict[int, dict] = {}
    all_parent_rows = []
    all_layer_rows = []
    all_loop_rows = []
    all_loop_summary = []

    t0 = time.time()
    lines = ["PHASE AGING OF FIXED LAYERS", f"  max_level={max_level}", ""]

    for gl in range(1, max_level + 1):
        frontier = model.grow_one_level(frontier)
        stored_map = {}
        neutral_map = {}
        parent_rows = []

        for p in completed_parent_ids(model):
            st = stored_response_phase(model, p)
            snap = refreshed_conductance_snapshot(model, p)
            if st is None or snap is None:
                continue
            if p not in baseline:
                baseline[p] = {
                    "completion_observation_level": gl,
                    "parent_level": model.nodes[p].level,
                    "stored_phase": st,
                    "neutral_phase": snap["neutral_phase"],
                    "neutral_norm": snap["neutral_norm"],
                    "g_rel_std": snap["g_rel_std"],
                    "g_spread": snap["g_spread"],
                }
            b = baseline[p]
            age = gl - b["completion_observation_level"]
            stored_drift = abs(angle_diff(st, b["stored_phase"]))
            neutral_phase_drift = abs(angle_diff(snap["neutral_phase"], b["neutral_phase"]))
            neutral_norm_delta = snap["neutral_norm"] - b["neutral_norm"]
            g_rel_std_delta = snap["g_rel_std"] - b["g_rel_std"]
            g_spread_delta = snap["g_spread"] - b["g_spread"]

            row = {
                "global_level": gl,
                "parent": p,
                "parent_level": model.nodes[p].level,
                "completion_observation_level": b["completion_observation_level"],
                "age": age,
                "stored_phase": st,
                "stored_phase_drift_abs": stored_drift,
                "neutral_phase": snap["neutral_phase"],
                "neutral_phase_drift_abs": neutral_phase_drift,
                "neutral_norm": snap["neutral_norm"],
                "neutral_norm_delta": neutral_norm_delta,
                "g_rel_std": snap["g_rel_std"],
                "g_rel_std_delta": g_rel_std_delta,
                "g_spread": snap["g_spread"],
                "g_spread_delta": g_spread_delta,
            }
            parent_rows.append(row)
            stored_map[p] = st
            neutral_map[p] = snap["neutral_phase"]

        all_parent_rows.extend(parent_rows)

        # Layer/age summaries.
        for (parent_level, age), rs in sorted(defaultdict(list, {}).items()):
            pass
        groups = defaultdict(list)
        for r in parent_rows:
            groups[(r["parent_level"], r["age"])].append(r)
        for (pl, age), rs in sorted(groups.items()):
            all_layer_rows.append({
                "global_level": gl,
                "parent_level": pl,
                "age": age,
                "count": len(rs),
                "stored_phase_drift_abs_mean": mean([r["stored_phase_drift_abs"] for r in rs]),
                "stored_phase_drift_abs_p95": perc([r["stored_phase_drift_abs"] for r in rs], 95),
                "neutral_phase_drift_abs_mean": mean([r["neutral_phase_drift_abs"] for r in rs]),
                "neutral_phase_drift_abs_p95": perc([r["neutral_phase_drift_abs"] for r in rs], 95),
                "neutral_norm_delta_mean": mean([r["neutral_norm_delta"] for r in rs]),
                "g_rel_std_delta_mean": mean([r["g_rel_std_delta"] for r in rs]),
                "g_spread_delta_mean": mean([r["g_spread_delta"] for r in rs]),
            })

        for phase_kind, phase_map in [("stored_response", stored_map), ("refreshed_neutral", neutral_map)]:
            loop_rows = level_centered_loop_residuals(model, phase_map)
            for lr in loop_rows:
                lr["global_level"] = gl
                lr["phase_kind"] = phase_kind
            all_loop_rows.extend(loop_rows)
            by_mode = defaultdict(list)
            for lr in loop_rows:
                by_mode[lr["loop_mode"]].append(lr["abs_residual"])
            for mode, vals in sorted(by_mode.items()):
                all_loop_summary.append({
                    "global_level": gl,
                    "phase_kind": phase_kind,
                    "loop_mode": mode,
                    "count": len(vals),
                    "mean_abs_residual": mean(vals),
                    "p95_abs_residual": perc(vals, 95),
                })

        # Compact log for polar/stored vs refreshed.
        stored_drift_all = mean([r["stored_phase_drift_abs"] for r in parent_rows])
        neutral_drift_all = mean([r["neutral_phase_drift_abs"] for r in parent_rows])
        neutral_norm_delta_all = mean([r["neutral_norm_delta"] for r in parent_rows])
        gspread_delta_all = mean([r["g_spread_delta"] for r in parent_rows])
        stored_loop_mean = mean([r["abs_residual"] for r in all_loop_rows if r["global_level"] == gl and r["phase_kind"] == "stored_response"])
        neutral_loop_mean = mean([r["abs_residual"] for r in all_loop_rows if r["global_level"] == gl and r["phase_kind"] == "refreshed_neutral"])
        lines.append(
            f"  L={gl}: nodes={len(model.nodes)}, completed={len(parent_rows)}, "
            f"stored_drift={stored_drift_all:.9e}, neutral_phase_drift={neutral_drift_all:.9f}, "
            f"neutral_norm_delta={neutral_norm_delta_all:.9f}, gspread_delta={gspread_delta_all:.9f}, "
            f"stored_loop={stored_loop_mean:.9f}, neutral_loop={neutral_loop_mean:.9f}, "
            f"elapsed={time.time()-t0:.2f}s"
        )

    write_csv(outdir / "phase_aging_parent_rows.csv", all_parent_rows)
    write_csv(outdir / "phase_aging_layer_age_summary.csv", all_layer_rows)
    write_csv(outdir / "phase_aging_loop_rows.csv", all_loop_rows)
    write_csv(outdir / "phase_aging_loop_summary.csv", all_loop_summary)

    lines.append("")
    lines.append("FINAL LEVEL AGE SUMMARY")
    final = max_level
    for r in all_layer_rows:
        if r["global_level"] == final and r["parent_level"] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            lines.append(
                f"  parent_level={r['parent_level']}, age={r['age']}, count={r['count']}, "
                f"stored_drift={r['stored_phase_drift_abs_mean']:.3e}, "
                f"neutral_phase_drift={r['neutral_phase_drift_abs_mean']:.9f}, "
                f"neutral_norm_delta={r['neutral_norm_delta_mean']:.9f}, "
                f"gspread_delta={r['g_spread_delta_mean']:.9f}"
            )

    lines.append("")
    lines.append("FINAL LEVEL LOOP SUMMARY")
    for r in all_loop_summary:
        if r["global_level"] == final:
            lines.append(
                f"  {r['phase_kind']} / {r['loop_mode']}: count={r['count']}, "
                f"mean_abs_res={r['mean_abs_residual']:.9f}, p95={r['p95_abs_residual']:.9f}"
            )

    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary

def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=10)
    ap.add_argument("--outdir", type=Path, default=Path("phase_aging_fixed_layers_out"))
    args = ap.parse_args()
    print(run(args.max_level, args.outdir))

if __name__ == "__main__":
    main()
