#!/usr/bin/env python3
"""
CNNA / growing real complement network
Response operator refresh rule stress test.

Purpose
-------
Compare three semantics for completed local sibling triples:

1. record_only
   The historical directed local response W_birth stored at the birth/completion
   event of the parent triple. This is the old local_w record layer.

2. live_only
   A current-state replay of the same local birth/response law, but using the
   current child conductances and current ancestor environment. This is not a
   new free force: it reuses the existing alpha_env, br_sibling,
   ancestor_env_decay, current g-values, and shell-normalized backreaction.

3. record_plus_live
   A two-layer/parallel-channel diagnostic. Keep W_birth as the event record and
   add only the nonnegative later live increment of the replayed current-state
   operator relative to its completion snapshot:

       W_two = W_birth + max(W_live(t) - W_live(t_complete), 0)

   The clipped negative part is reported as audit mass, because a negative
   conductance channel is not silently inserted.

Hard limitation
---------------
This is a numerical provenance/semantics test. It does not prove physical i,
physical time, modular flow, AQFT, or Type III structure.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


class ShellNormGrowth:
    def __init__(
        self,
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_env_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
    ):
        if branching != 3:
            raise ValueError("this local-response test currently assumes b=3")
        self.b = branching
        self.base = base
        self.alpha_env = alpha_env
        self.ancestor_env_decay = ancestor_env_decay
        self.br_ancestor = br_ancestor
        self.br_sibling = br_sibling
        self.kernel_name = "shell_norm_inverse_square"

        self.nodes: Dict[int, Node] = {}
        self.local_w: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
        self.live_completion_matrix: Dict[int, np.ndarray] = {}
        self.record_completion_matrix: Dict[int, np.ndarray] = {}
        self.completion_level: Dict[int, int] = {}
        self.completion_child_g: Dict[int, np.ndarray] = {}
        self.completion_ancestor_env: Dict[int, float] = {}
        self.desc_shell_load_by_node: Dict[int, float] = defaultdict(float)

        self.next_id = 0
        self.t = 0
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
        return 1.0 / ((self.b ** (d - 1)) * d * d)

    def parent_line(self, parent: int) -> List[int]:
        line: List[int] = []
        cur: Optional[int] = parent
        while cur is not None:
            line.append(cur)
            cur = self.nodes[cur].parent
        return line

    def ancestor_env_load(self, parent: int) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_env_decay ** (d - 1))
        return env

    def birth_env_load(self, parent: int, older: List[int]) -> float:
        env = self.ancestor_env_load(parent)
        for s in older:
            env += self.nodes[s].g
        return env

    def child_g_from_env(self, env: float) -> float:
        return self.base + self.alpha_env * math.log1p(env)

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env = self.birth_env_load(parent, older)
        bg = self.child_g_from_env(env)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, bg)

        for s in older:
            i = self.nodes[s].birth_order
            j = order
            self.local_w[parent][(i, j)] += self.alpha_env * self.nodes[s].g / (env + EPS) * bg

        for s in older:
            i = order
            j = self.nodes[s].birth_order
            delta = self.br_sibling * bg
            self.nodes[s].g += delta
            self.local_w[parent][(i, j)] += delta

        for d, a in enumerate(self.parent_line(parent), start=1):
            kd = self.kernel_value(d)
            self.nodes[a].g += self.br_ancestor * bg * kd
            self.desc_shell_load_by_node[a] += bg * kd

        if len(self.nodes[parent].children) == self.b:
            self._snapshot_completion(parent)

        return child.id

    def grow_one_level(self, frontier: List[int]) -> List[int]:
        new_frontier: List[int] = []
        for p in frontier:
            for k in range(1, self.b + 1):
                new_frontier.append(self.add_child(p, k))
        return new_frontier

    def completed_parent_ids(self) -> List[int]:
        return [n.id for n in self.nodes.values() if len(n.children) == self.b]

    def child_ids_ordered(self, parent: int) -> List[int]:
        ch = list(self.nodes[parent].children)
        ch.sort(key=lambda c: self.nodes[c].birth_order)
        return ch

    def child_g_vector(self, parent: int) -> np.ndarray:
        return np.array([self.nodes[c].g for c in self.child_ids_ordered(parent)], dtype=float)

    def local_record_matrix(self, parent: int) -> Optional[np.ndarray]:
        if len(self.nodes[parent].children) != self.b:
            return None
        M = np.zeros((3, 3), dtype=float)
        w = self.local_w[parent]
        for i in range(1, 4):
            for j in range(1, 4):
                if i != j:
                    M[j - 1, i - 1] = w.get((i, j), 0.0)
        return M

    def live_replay_matrix(self, parent: int) -> Optional[np.ndarray]:
        if len(self.nodes[parent].children) != self.b:
            return None
        g = self.child_g_vector(parent)
        ancestor_env = self.ancestor_env_load(parent)
        M = np.zeros((3, 3), dtype=float)
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    continue
                source = g[i - 1]
                target = g[j - 1]
                if i < j:
                    prefix = ancestor_env + float(np.sum(g[: j - 1]))
                    M[j - 1, i - 1] = self.alpha_env * source / (prefix + EPS) * target
                else:
                    M[j - 1, i - 1] = self.br_sibling * source
        return M

    def two_layer_matrix(self, parent: int) -> Optional[Tuple[np.ndarray, dict]]:
        R = self.local_record_matrix(parent)
        L = self.live_replay_matrix(parent)
        if R is None or L is None or parent not in self.live_completion_matrix:
            return None
        dL = L - self.live_completion_matrix[parent]
        pos = np.maximum(dL, 0.0)
        neg = np.maximum(-dL, 0.0)
        M = R + pos
        audit = {
            "live_delta_pos_sum": float(np.sum(pos)),
            "live_delta_neg_sum": float(np.sum(neg)),
            "live_delta_signed_sum": float(np.sum(dL)),
            "live_delta_fro": float(np.linalg.norm(dL, ord="fro")),
        }
        return M, audit

    def descendant_count(self, node: int) -> int:
        count = 0
        stack = list(self.nodes[node].children)
        while stack:
            u = stack.pop()
            count += 1
            stack.extend(self.nodes[u].children)
        return count

    def descendant_shell_birth_load(self, node: int) -> float:
        return float(self.desc_shell_load_by_node.get(node, 0.0))

    def child_descendant_loads(self, parent: int) -> np.ndarray:
        return np.array([self.descendant_shell_birth_load(c) for c in self.child_ids_ordered(parent)], dtype=float)

    def _snapshot_completion(self, parent: int) -> None:
        if parent in self.completion_level:
            return
        R = self.local_record_matrix(parent)
        L = self.live_replay_matrix(parent)
        if R is None or L is None:
            return
        self.record_completion_matrix[parent] = R.copy()
        self.live_completion_matrix[parent] = L.copy()
        self.completion_level[parent] = self.nodes[parent].level + 1
        self.completion_child_g[parent] = self.child_g_vector(parent).copy()
        self.completion_ancestor_env[parent] = self.ancestor_env_load(parent)


def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def angle_diff(a: float, b: float) -> float:
    return wrap_deg(a - b)


def mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def std(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.std(xs)) if xs else float("nan")


def perc(vals: Iterable[float], q: float) -> float:
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.percentile(xs, q)) if xs else float("nan")


def column_stochastic(M: np.ndarray) -> np.ndarray:
    P = M.copy().astype(float)
    for j in range(P.shape[1]):
        s = float(P[:, j].sum())
        if s > EPS:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    return P


def skew_axis(A: np.ndarray) -> np.ndarray:
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def cross_matrix(a: np.ndarray) -> np.ndarray:
    x, y, z = a
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def plane_basis(axis: np.ndarray) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + EPS)
    h = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, h))) > 0.9:
        h = np.array([0.0, 1.0, 0.0])
    u = h - np.dot(h, a) * a
    u = u / (np.linalg.norm(u) + EPS)
    v = np.cross(a, u)
    v = v / (np.linalg.norm(v) + EPS)
    return np.vstack([u, v]).T


def polar_so2(A2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    U, s, Vt = np.linalg.svd(A2)
    Q = U @ Vt
    refl = 0
    if np.linalg.det(Q) < 0:
        refl = 1
        U[:, -1] *= -1
        Q = U @ Vt
    return Q, s, refl


def angle_so2_deg(Q: np.ndarray) -> float:
    cosv = float((Q[0, 0] + Q[1, 1]) / 2.0)
    sinv = float((Q[1, 0] - Q[0, 1]) / 2.0)
    return math.degrees(math.atan2(sinv, cosv))


def matrix_diagnostics(M: np.ndarray) -> Optional[dict]:
    if M is None or not np.all(np.isfinite(M)):
        return None
    if float(np.sum(np.abs(M))) <= EPS:
        return None
    P = column_stochastic(M)
    S = 0.5 * (P + P.T)
    A = 0.5 * (P - P.T)
    axis_raw = skew_axis(A)
    axis_norm = float(np.linalg.norm(axis_raw))
    if axis_norm <= EPS:
        return None
    c = np.ones(3) / math.sqrt(3.0)
    if np.dot(axis_raw, c) < 0:
        axis_raw = -axis_raw
    axis = axis_raw / (np.linalg.norm(axis_raw) + EPS)
    B = plane_basis(axis)
    A2 = B.T @ A @ B
    alpha2 = max(0.0, float(-np.trace(A2 @ A2) / 2.0))
    alpha = math.sqrt(alpha2)
    if alpha > EPS:
        J = A2 / alpha
        j2_resid = float(np.linalg.norm(J @ J + np.eye(2), ord="fro"))
    else:
        j2_resid = float("inf")
    R2 = B.T @ P @ B
    Q, sing, refl = polar_so2(R2)
    theta = angle_so2_deg(Q)
    ev = np.linalg.eigvals(P)
    imag_max = float(np.max(np.abs(np.imag(ev))))
    return {
        "M_sum": float(np.sum(M)),
        "P": P,
        "S_min": float(np.min(np.linalg.eigvalsh(S))),
        "S_max": float(np.max(np.linalg.eigvalsh(S))),
        "axis": axis,
        "axis_norm_raw": axis_norm,
        "axis_align_const": float(np.dot(axis, c)),
        "alpha": alpha,
        "J2_resid": j2_resid,
        "theta_deg": theta,
        "theta_abs_deg": abs(theta),
        "polar_reflection": refl,
        "singular_0": float(sing[0]),
        "singular_1": float(sing[1]),
        "anisotropy": float(max(sing) / (min(sing) + EPS)),
        "complex_pair": float(imag_max > 1e-9),
        "eig_imag_max": imag_max,
    }


def phase_for_matrix(M: np.ndarray) -> Optional[float]:
    d = matrix_diagnostics(M)
    if d is None:
        return None
    return float(d["theta_deg"])


def semantic_matrices(model: ShellNormGrowth, parent: int) -> Dict[str, Tuple[np.ndarray, dict]]:
    out: Dict[str, Tuple[np.ndarray, dict]] = {}
    R = model.local_record_matrix(parent)
    L = model.live_replay_matrix(parent)
    if R is not None:
        out["record_only"] = (R, {})
    if L is not None:
        out["live_only"] = (L, {})
    T = model.two_layer_matrix(parent)
    if T is not None:
        out["record_plus_live"] = T
    return out


def build_loops(model: ShellNormGrowth, phase_map: Dict[int, float]) -> List[dict]:
    loops = []
    for p in phase_map:
        cs = [c for c in model.child_ids_ordered(p) if c in phase_map]
        if len(cs) == 3:
            c1, c2, c3 = cs
            loops.append({"loop_mode": "sibling_cycle", "loop_level": model.nodes[p].level + 1, "base": p, "loop": [c1, c2, c3]})
            loops.append({"loop_mode": "parent_child_ring", "loop_level": model.nodes[p].level, "base": p, "loop": [p, c1, c2, c3]})
            loops.append({"loop_mode": "parent_fan_triangle", "loop_level": model.nodes[p].level, "base": p, "loop": [p, c1, c2]})
            loops.append({"loop_mode": "parent_fan_triangle", "loop_level": model.nodes[p].level, "base": p, "loop": [p, c2, c3]})
            loops.append({"loop_mode": "parent_fan_triangle", "loop_level": model.nodes[p].level, "base": p, "loop": [p, c3, c1]})
    return loops


def level_background(model: ShellNormGrowth, phase_map: Dict[int, float]) -> Dict[int, float]:
    bg: Dict[int, List[float]] = defaultdict(list)
    for p, th in phase_map.items():
        bg[model.nodes[p].level].append(th)
    return {k: mean(v) for k, v in bg.items()}


def loop_residual_rows(model: ShellNormGrowth, semantic: str, global_level: int, phase_map: Dict[int, float]) -> List[dict]:
    bg = level_background(model, phase_map)
    rows = []
    for item in build_loops(model, phase_map):
        loop = item["loop"]
        vals = [phase_map[u] for u in loop]
        bgs = [bg[model.nodes[u].level] for u in loop]
        raw = wrap_deg(sum(vals))
        centered = wrap_deg(sum(vals) - sum(bgs))
        base = item["base"]
        base_level = model.nodes[base].level if base in model.nodes else -1
        rows.append({
            "global_level": global_level,
            "semantic": semantic,
            "loop_mode": item["loop_mode"],
            "loop_level": item["loop_level"],
            "base": base,
            "base_level": base_level,
            "distance_to_frontier": max(0, global_level - 1 - base_level),
            "loop_len": len(loop),
            "raw_sum_wrapped_deg": raw,
            "abs_raw_sum_wrapped_deg": abs(raw),
            "level_centered_residual_deg": centered,
            "abs_level_centered_residual_deg": abs(centered),
        })
    return rows


def pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    dot = float(np.dot(a, b))
    dot_abs = abs(dot)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot_abs))))
    return {
        "axis_dot": dot,
        "axis_abs_dot": dot_abs,
        "axis_angle_deg": angle,
        "plane_distance": math.sqrt(max(0.0, 1.0 - dot_abs * dot_abs)),
        "J_frobenius_mismatch": float(np.linalg.norm(cross_matrix(a) - cross_matrix(b), ord="fro")),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[dict], group_keys: List[str], value_keys: List[str]) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        row = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row["count"] = len(rs)
        for vk in value_keys:
            vals = [r[vk] for r in rs if vk in r]
            row[f"{vk}_mean"] = mean(vals)
            row[f"{vk}_std"] = std(vals)
            row[f"{vk}_p95"] = perc(vals, 95)
        out.append(row)
    return out


def run(max_level: int, outdir: Path, emit_loop_rows: bool = True) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    model = ShellNormGrowth()
    frontier = [model.root]

    completion_phase: Dict[Tuple[str, int], float] = {}
    all_local_rows: List[dict] = []
    all_level_rows: List[dict] = []
    all_aging_rows: List[dict] = []
    all_loop_rows: List[dict] = []
    all_gluing_rows: List[dict] = []
    all_gluing_summary: List[dict] = []
    final_local_rows: List[dict] = []

    lines = [
        "RESPONSE OPERATOR REFRESH RULE TEST",
        f"  max_level={max_level}",
        "  kernel=shell_norm_inverse_square = 1/(3^(d-1) d^2)",
        "  semantics=record_only, live_only, record_plus_live",
        "",
    ]
    t0 = time.time()

    for gl in range(1, max_level + 1):
        frontier = model.grow_one_level(frontier)
        local_rows_this: List[dict] = []
        phase_maps: Dict[str, Dict[int, float]] = {"record_only": {}, "live_only": {}, "record_plus_live": {}}
        axis_maps: Dict[str, Dict[int, np.ndarray]] = {"record_only": {}, "live_only": {}, "record_plus_live": {}}

        for p in model.completed_parent_ids():
            child_g = model.child_g_vector(p)
            child_g0 = model.completion_child_g.get(p, np.full(3, np.nan))
            desc_loads = model.child_descendant_loads(p)
            ancestor_env = model.ancestor_env_load(p)
            ancestor_env0 = model.completion_ancestor_env.get(p, float("nan"))
            completion_gl = model.completion_level.get(p, gl)
            age = gl - completion_gl
            distance_to_frontier = max(0, gl - 1 - model.nodes[p].level)

            for semantic, (M, audit) in semantic_matrices(model, p).items():
                d = matrix_diagnostics(M)
                if d is None:
                    continue
                key = (semantic, p)
                if key not in completion_phase:
                    completion_phase[key] = float(d["theta_deg"])
                phase_drift = abs(angle_diff(float(d["theta_deg"]), completion_phase[key]))
                row = {
                    "global_level": gl,
                    "semantic": semantic,
                    "parent": p,
                    "parent_level": model.nodes[p].level,
                    "completion_level": completion_gl,
                    "age": age,
                    "distance_to_frontier": distance_to_frontier,
                    "theta_deg": float(d["theta_deg"]),
                    "theta_abs_deg": float(d["theta_abs_deg"]),
                    "phase_drift_abs_deg": phase_drift,
                    "J2_resid": float(d["J2_resid"]),
                    "axis_norm_raw": float(d["axis_norm_raw"]),
                    "axis_align_const": float(d["axis_align_const"]),
                    "alpha": float(d["alpha"]),
                    "M_sum": float(d["M_sum"]),
                    "S_min": float(d["S_min"]),
                    "S_max": float(d["S_max"]),
                    "anisotropy": float(d["anisotropy"]),
                    "complex_pair": float(d["complex_pair"]),
                    "eig_imag_max": float(d["eig_imag_max"]),
                    "child_g1": float(child_g[0]),
                    "child_g2": float(child_g[1]),
                    "child_g3": float(child_g[2]),
                    "child_g_delta1": float(child_g[0] - child_g0[0]),
                    "child_g_delta2": float(child_g[1] - child_g0[1]),
                    "child_g_delta3": float(child_g[2] - child_g0[2]),
                    "child_g_rel_std": float(np.std(child_g) / (np.mean(child_g) + EPS)),
                    "desc_shell_load1": float(desc_loads[0]),
                    "desc_shell_load2": float(desc_loads[1]),
                    "desc_shell_load3": float(desc_loads[2]),
                    "ancestor_env": float(ancestor_env),
                    "ancestor_env_delta": float(ancestor_env - ancestor_env0),
                    **audit,
                }
                local_rows_this.append(row)
                phase_maps[semantic][p] = float(d["theta_deg"])
                axis_maps[semantic][p] = d["axis"]

        all_local_rows.extend(local_rows_this)
        if gl == max_level:
            final_local_rows = list(local_rows_this)

        level_summary = summarize(
            local_rows_this,
            ["global_level", "semantic", "parent_level"],
            ["theta_deg", "phase_drift_abs_deg", "J2_resid", "axis_align_const", "child_g_rel_std", "ancestor_env_delta"],
        )
        all_level_rows.extend(level_summary)

        aging_summary = summarize(
            local_rows_this,
            ["global_level", "semantic", "distance_to_frontier"],
            ["phase_drift_abs_deg", "J2_resid", "child_g_rel_std", "ancestor_env_delta"],
        )
        all_aging_rows.extend(aging_summary)

        loop_summaries_this = []
        for semantic, phase_map in phase_maps.items():
            loop_rows = loop_residual_rows(model, semantic, gl, phase_map)
            if emit_loop_rows:
                all_loop_rows.extend(loop_rows)
            loop_summaries_this.extend(summarize(
                loop_rows,
                ["global_level", "semantic", "loop_mode", "distance_to_frontier"],
                ["abs_level_centered_residual_deg", "abs_raw_sum_wrapped_deg"],
            ))
        write_csv(outdir / "refresh_loop_summary_by_distance.csv", loop_summaries_this if gl == max_level else [])

        if gl == max_level:
            all_loop_summary = []
            for semantic, phase_map in phase_maps.items():
                loop_rows = loop_residual_rows(model, semantic, gl, phase_map)
                all_loop_summary.extend(summarize(
                    loop_rows,
                    ["global_level", "semantic", "loop_mode"],
                    ["abs_level_centered_residual_deg", "abs_raw_sum_wrapped_deg"],
                ))
            write_csv(outdir / "refresh_final_loop_summary.csv", all_loop_summary)

        if gl == max_level:
            for semantic, axes in axis_maps.items():
                vertical = []
                for p, a in axes.items():
                    for c in model.child_ids_ordered(p):
                        if c in axes:
                            m = pair_metrics(a, axes[c])
                            vertical.append({
                                "semantic": semantic,
                                "parent": p,
                                "child": c,
                                "parent_level": model.nodes[p].level,
                                "child_level": model.nodes[c].level,
                                **m,
                            })
                all_gluing_rows.extend(vertical)
                all_gluing_summary.extend(summarize(
                    vertical,
                    ["semantic", "parent_level"],
                    ["axis_abs_dot", "axis_angle_deg", "plane_distance", "J_frobenius_mismatch"],
                ))

        compact = summarize(local_rows_this, ["global_level", "semantic"], ["phase_drift_abs_deg", "J2_resid", "axis_align_const"])
        msg_parts = []
        for r in compact:
            msg_parts.append(
                f"{r['semantic']}: drift={r['phase_drift_abs_deg_mean']:.6g}, "
                f"J2={r['J2_resid_mean']:.3e}, axis={r['axis_align_const_mean']:.6f}"
            )
        lines.append(
            f"  L={gl}: nodes={len(model.nodes)}, completed={len(model.completed_parent_ids())}, "
            + " | ".join(msg_parts)
            + f", elapsed={time.time() - t0:.2f}s"
        )

    final_level_summary = [r for r in all_level_rows if r["global_level"] == max_level]
    final_aging_summary = [r for r in all_aging_rows if r["global_level"] == max_level]

    write_csv(outdir / "refresh_local_rows_all_levels.csv", all_local_rows)
    write_csv(outdir / "refresh_local_rows_final.csv", final_local_rows)
    write_csv(outdir / "refresh_level_summary.csv", all_level_rows)
    write_csv(outdir / "refresh_final_level_summary.csv", final_level_summary)
    write_csv(outdir / "refresh_aging_summary.csv", all_aging_rows)
    write_csv(outdir / "refresh_final_aging_summary.csv", final_aging_summary)
    write_csv(outdir / "refresh_vertical_gluing_pairs_final.csv", all_gluing_rows)
    write_csv(outdir / "refresh_vertical_gluing_summary_final.csv", all_gluing_summary)
    if emit_loop_rows:
        write_csv(outdir / "refresh_loop_rows_all_levels.csv", all_loop_rows)

    lines.append("")
    lines.append("FINAL SEMANTIC SUMMARY")
    final_sem = summarize(final_local_rows, ["semantic"], ["theta_deg", "phase_drift_abs_deg", "J2_resid", "axis_align_const", "child_g_rel_std"])
    for r in final_sem:
        lines.append(
            f"  {r['semantic']}: count={r['count']}, "
            f"theta_mean={r['theta_deg_mean']:.9f}, theta_std={r['theta_deg_std']:.9f}, "
            f"drift_mean={r['phase_drift_abs_deg_mean']:.9f}, "
            f"J2_mean={r['J2_resid_mean']:.3e}, axis_align={r['axis_align_const_mean']:.9f}"
        )

    lines.append("")
    lines.append("FINAL OLD-INTERIOR VS FRONTIER")
    by_sem_dist = summarize(final_local_rows, ["semantic", "distance_to_frontier"], ["phase_drift_abs_deg", "J2_resid", "child_g_rel_std"])
    for r in by_sem_dist:
        if r["distance_to_frontier"] in [0, 1, 2, 3, 4, 5, max_level - 1]:
            lines.append(
                f"  {r['semantic']} distance={r['distance_to_frontier']}: count={r['count']}, "
                f"drift={r['phase_drift_abs_deg_mean']:.9f}, J2={r['J2_resid_mean']:.3e}, "
                f"child_rel_std={r['child_g_rel_std_mean']:.9f}"
            )

    if all_gluing_rows:
        lines.append("")
        lines.append("FINAL VERTICAL GLUING ALL")
        for semantic in ["record_only", "live_only", "record_plus_live"]:
            rs = [r for r in all_gluing_rows if r["semantic"] == semantic]
            lines.append(
                f"  {semantic}: pairs={len(rs)}, mean|dot|={mean([r['axis_abs_dot'] for r in rs]):.9f}, "
                f"mean_angle={mean([r['axis_angle_deg'] for r in rs]):.9f}, "
                f"mean_J_mismatch={mean([r['J_frobenius_mismatch'] for r in rs]):.9e}"
            )

    if emit_loop_rows and all_loop_rows:
        final_loop_rows = [r for r in all_loop_rows if r["global_level"] == max_level]
        lines.append("")
        lines.append("FINAL LEVEL-CENTERED LOOP RESIDUALS ALL MODES")
        for semantic in ["record_only", "live_only", "record_plus_live"]:
            rs = [r for r in final_loop_rows if r["semantic"] == semantic]
            lines.append(
                f"  {semantic}: loops={len(rs)}, mean_abs_centered={mean([r['abs_level_centered_residual_deg'] for r in rs]):.9f}, "
                f"p95={perc([r['abs_level_centered_residual_deg'] for r in rs], 95):.9f}"
            )

    two_rows = [r for r in final_local_rows if r["semantic"] == "record_plus_live"]
    if two_rows:
        lines.append("")
        lines.append("TWO-LAYER AUDIT")
        lines.append(
            f"  mean live_delta_pos_sum={mean([r.get('live_delta_pos_sum', float('nan')) for r in two_rows]):.9e}"
        )
        lines.append(
            f"  mean live_delta_neg_sum={mean([r.get('live_delta_neg_sum', float('nan')) for r in two_rows]):.9e}"
        )
        lines.append(
            "  negative live delta is not inserted as conductance; it is reported as audit mass."
        )

    lines.append("")
    lines.append("INTERPRETATION GUARD")
    lines.append("  This test compares response semantics. It does not prove physical i/time/modular flow/Type III.")

    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=9)
    ap.add_argument("--outdir", type=Path, default=Path("response_operator_refresh_rule_out"))
    ap.add_argument("--no-loop-rows", action="store_true")
    args = ap.parse_args()
    print(run(args.max_level, args.outdir, emit_loop_rows=not args.no_loop_rows))


if __name__ == "__main__":
    main()
