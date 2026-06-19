#!/usr/bin/env python3
"""
Phase renormalization / clock subtraction test.

Best current candidate:
    kernel K(d) = 1 / (3^(d-1) d^2)

Idea:
    theta_inf is interpreted as a background response-clock phase density.
    For a loop of length n, subtract n * theta_inf from the accumulated phase.

This test compares:
1. raw wrapped loop phase
2. clock-subtracted loop phase using fixed theta_inf
3. level-mean-subtracted loop phase
4. source-parent-level-subtracted loop phase
5. normalized effective generator h = theta/theta_inf - 1

It also fits the decay of these residuals by level.

This is a numerical diagnostic, not a theorem and not a physical-time claim.
"""

from __future__ import annotations

import argparse, csv, math, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12
METHODS = ["polar", "eigen", "G_weighted"]

# From shell_phase_limit_highL L11 tail fits.
DEFAULT_THETA_INF = {
    "polar": 168.027421321,
    "eigen": 172.552820014,
    "G_weighted": 167.890861027,
}

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
        self.local_w: Dict[int, Dict[Tuple[int,int], float]] = defaultdict(lambda: defaultdict(float))
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
        c = child.id

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

        return c

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
    refl = 0
    if np.linalg.det(Q) < 0:
        refl = 1
        U[:, -1] *= -1
        Q = U @ Vt
    return Q, s, refl

def angle_so2_deg(Q):
    cosv = float((Q[0, 0] + Q[1, 1]) / 2.0)
    sinv = float((Q[1, 0] - Q[0, 1]) / 2.0)
    return math.degrees(math.atan2(sinv, cosv))

def local_phase(model: ShellGrowth, parent: int) -> Optional[dict]:
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

    Q, s, refl = polar_so2(R2)
    theta_polar = angle_so2_deg(Q)

    ev = np.linalg.eigvals(P)
    ces = [z for z in ev if abs(z.imag) > 1e-9]
    theta_eigen = float("nan")
    if ces:
        z = max(ces, key=lambda z: z.imag)
        theta_eigen = math.degrees(math.atan2(z.imag, z.real))

    S2 = 0.5 * (R2 + R2.T)
    theta_G = float("nan")
    for sign, cand in [(-1, -S2), (1, S2)]:
        eig = np.linalg.eigvalsh(cand)
        if np.all(eig > 1e-10):
            try:
                L = np.linalg.cholesky(cand)
                C = L.T
                Rg = C @ R2 @ np.linalg.inv(C)
                Qg, _, _ = polar_so2(Rg)
                theta_G = angle_so2_deg(Qg)
                break
            except np.linalg.LinAlgError:
                pass

    return {
        "parent": parent,
        "level": model.nodes[parent].level,
        "polar": theta_polar,
        "eigen": theta_eigen,
        "G_weighted": theta_G,
    }

def completed_parent_ids(model: ShellGrowth) -> List[int]:
    return [n.id for n in model.nodes.values() if len(n.children) == 3]

def build_loops(model: ShellGrowth, data: Dict[int, dict]):
    loops = []
    for p in data:
        cs = [c for c in model.nodes[p].children if c in data]
        if len(cs) == 3:
            c1, c2, c3 = cs
            loops.append(("sibling_cycle", model.nodes[p].level + 1, [c1, c2, c3]))
            loops.append(("parent_child_ring", model.nodes[p].level, [p, c1, c2, c3]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, [p, c1, c2]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, [p, c2, c3]))
            loops.append(("parent_fan_triangle", model.nodes[p].level, [p, c3, c1]))
    return loops

def summarize_level(model: ShellGrowth, gl: int, theta_inf: Dict[str, float]) -> dict:
    data = {}
    for p in completed_parent_ids(model):
        d = local_phase(model, p)
        if d is not None:
            data[p] = d

    row = {"global_level": gl, "nodes": len(model.nodes), "completed": len(data)}

    by_source_level = {m: defaultdict(list) for m in METHODS}
    for d in data.values():
        for m in METHODS:
            by_source_level[m][d["level"]].append(d[m])

    bg_level = {m: {k: mean(v) for k, v in by_source_level[m].items()} for m in METHODS}
    bg_global = {m: mean([d[m] for d in data.values()]) for m in METHODS}

    for m in METHODS:
        vals = [d[m] for d in data.values()]
        row[f"{m}_mean"] = mean(vals)
        row[f"{m}_std"] = std(vals)
        row[f"{m}_theta_inf"] = theta_inf[m]
        row[f"{m}_mean_minus_theta_inf"] = row[f"{m}_mean"] - theta_inf[m]
        h = [v / theta_inf[m] - 1.0 for v in vals if np.isfinite(v)]
        row[f"{m}_heff_mean"] = mean(h)
        row[f"{m}_heff_std"] = std(h)

    loops = build_loops(model, data)
    row["loop_count"] = len(loops)

    for m in METHODS:
        raw = []
        clock = []
        level_centered = []
        global_centered = []
        hloop = []
        by_mode_clock = defaultdict(list)
        by_mode_level = defaultdict(list)
        for mode, lvl, loop in loops:
            vals = [data[u][m] for u in loop]
            if not all(np.isfinite(vals)):
                continue
            raw_v = abs(wrap_deg(sum(vals)))
            clock_v = abs(wrap_deg(sum(vals) - theta_inf[m] * len(loop)))
            glob_v = abs(wrap_deg(sum(vals) - bg_global[m] * len(loop)))
            lev_v = abs(wrap_deg(sum(vals) - sum(bg_level[m][data[u]["level"]] for u in loop)))
            h_v = abs(sum(v / theta_inf[m] - 1.0 for v in vals))
            raw.append(raw_v)
            clock.append(clock_v)
            global_centered.append(glob_v)
            level_centered.append(lev_v)
            hloop.append(h_v)
            by_mode_clock[mode].append(clock_v)
            by_mode_level[mode].append(lev_v)

        row[f"{m}_raw_loop_mean"] = mean(raw)
        row[f"{m}_clock_residual_mean"] = mean(clock)
        row[f"{m}_clock_residual_p95"] = perc(clock, 95)
        row[f"{m}_global_centered_mean"] = mean(global_centered)
        row[f"{m}_level_centered_mean"] = mean(level_centered)
        row[f"{m}_level_centered_p95"] = perc(level_centered, 95)
        row[f"{m}_heff_loop_abs_mean"] = mean(hloop)
        for mode in ["sibling_cycle", "parent_fan_triangle", "parent_child_ring"]:
            row[f"{m}_{mode}_clock_mean"] = mean(by_mode_clock[mode])
            row[f"{m}_{mode}_level_mean"] = mean(by_mode_level[mode])
    return row

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y-yhat)**2)))

def fit_decay(L, y):
    mask = np.isfinite(y) & (y > 0)
    L = L[mask]
    y = y[mask]
    if len(y) < 4:
        return {"model": "insufficient", "r": float("nan"), "alpha": float("nan"), "rmse": float("nan")}
    logy = np.log(y)
    X = np.vstack([np.ones_like(L), L]).T
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    yhat = np.exp(X @ beta)
    expfit = {"model": "C_r_pow_L", "C": float(math.exp(beta[0])), "r": float(math.exp(beta[1])), "alpha": float("nan"), "rmse": rmse(y, yhat)}
    N = (3.0 ** (L + 1) - 1.0) / 2.0
    X = np.vstack([np.ones_like(N), np.log(N)]).T
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    yhat = np.exp(X @ beta)
    powfit = {"model": "C_N_minus_alpha", "C": float(math.exp(beta[0])), "r": float("nan"), "alpha": float(-beta[1]), "rmse": rmse(y, yhat)}
    return expfit if expfit["rmse"] <= powfit["rmse"] else powfit

def fit_theta_gap(L, y):
    # y is signed gap mean - theta_inf. Fit abs gap decay.
    return fit_decay(L, np.abs(y))

def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def run(max_level: int, outdir: Path, theta_inf: Dict[str, float]):
    outdir.mkdir(parents=True, exist_ok=True)
    model = ShellGrowth()
    frontier = [model.root]
    rows = []
    t0 = time.time()
    for gl in range(1, max_level + 1):
        frontier = model.grow_one_level(frontier)
        row = summarize_level(model, gl, theta_inf)
        row["elapsed_sec"] = time.time() - t0
        rows.append(row)

    write_csv(outdir / "phase_renormalization_clock_levels.csv", rows)

    L = np.array([r["global_level"] for r in rows if r["global_level"] >= 4], float)
    fits = []
    for m in METHODS:
        for col, kind in [
            (f"{m}_mean_minus_theta_inf", "theta_gap_decay"),
            (f"{m}_std", "phase_std_decay"),
            (f"{m}_clock_residual_mean", "clock_residual_decay"),
            (f"{m}_level_centered_mean", "level_centered_decay"),
            (f"{m}_heff_loop_abs_mean", "heff_loop_abs_decay"),
        ]:
            y = np.array([r[col] for r in rows if r["global_level"] >= 4], float)
            fit = fit_theta_gap(L, y) if kind == "theta_gap_decay" else fit_decay(L, y)
            fit.update({"method": m, "kind": kind, "last": float(y[-1]), "column": col})
            fits.append(fit)
    write_csv(outdir / "phase_renormalization_clock_fits.csv", fits)

    lines = [
        "PHASE RENORMALIZATION / CLOCK SUBTRACTION TEST",
        f"  max_level={max_level}",
        f"  final nodes={rows[-1]['nodes']}",
        f"  final completed={rows[-1]['completed']}",
        "  theta_inf:",
    ]
    for m in METHODS:
        lines.append(f"    {m}: {theta_inf[m]:.9f} deg")
    lines.append("")
    for r in rows:
        lines.append(
            f"  L={r['global_level']}: nodes={r['nodes']}, completed={r['completed']}, "
            f"polar_mean={r['polar_mean']:.9f}, gap={r['polar_mean_minus_theta_inf']:.9f}, "
            f"std={r['polar_std']:.9f}, clock_res={r['polar_clock_residual_mean']:.9f}, "
            f"level_res={r['polar_level_centered_mean']:.9f}, heff_loop={r['polar_heff_loop_abs_mean']:.9e}"
        )
    lines += ["", "FITS"]
    for f in fits:
        if f["model"] == "C_r_pow_L":
            spec = f"r={f['r']:.6f}"
        elif f["model"] == "C_N_minus_alpha":
            spec = f"alpha={f['alpha']:.6f}"
        else:
            spec = ""
        lines.append(f"  {f['method']} {f['kind']}: last={f['last']:.9e}, model={f['model']}, {spec}, rmse={f['rmse']:.6e}")

    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=11)
    ap.add_argument("--outdir", type=Path, default=Path("phase_renormalization_clock_out"))
    ap.add_argument("--polar-theta-inf", type=float, default=DEFAULT_THETA_INF["polar"])
    ap.add_argument("--eigen-theta-inf", type=float, default=DEFAULT_THETA_INF["eigen"])
    ap.add_argument("--g-theta-inf", type=float, default=DEFAULT_THETA_INF["G_weighted"])
    args = ap.parse_args()
    theta_inf = {"polar": args.polar_theta_inf, "eigen": args.eigen_theta_inf, "G_weighted": args.g_theta_inf}
    print(run(args.max_level, args.outdir, theta_inf))

if __name__ == "__main__":
    main()
