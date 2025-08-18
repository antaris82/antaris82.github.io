#!/usr/bin/env python3
"""
E2 — ST Level-5 hard check:
- zentraler Ball (Radius 7), Isotropie-Sampling je Distanz-Shell
- J-Skalierung (0.5, 1.0, 2.0)
- first-crossing v_* für mehrere epsilons
- Bootstrap-CI
Schreibt JSON + eine CSV (eps=0.02, J=1) ins aktuelle Verzeichnis.
"""

from pathlib import Path
import json, math, random
from collections import deque
from itertools import combinations
import numpy as np

# ---------- ST graph ----------
def st_graph_exact(level: int):
    V = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]
    E = set((i, j) for i, j in combinations(range(4), 2))
    for m in range(1, level + 1):
        shift = 1 << (m - 1)
        newV = []; mapped_lists = []
        for i in range(4):
            e = [0,0,0,0]; e[i] = shift
            mapped = [tuple(vk + ek for vk, ek in zip(v, e)) for v in V]
            mapped_lists.append(mapped); newV.extend(mapped)
        uniq = {}; Vuniq = []
        for v in newV:
            if v not in uniq:
                uniq[v] = len(Vuniq); Vuniq.append(v)
        newE = set()
        for i in range(4):
            l2g = {old_idx: uniq[mapped_lists[i][old_idx]] for old_idx in range(len(V))}
            for (a, b) in E:
                na, nb = l2g[a], l2g[b]
                if na != nb: newE.add(tuple(sorted((na, nb))))
        V, E = Vuniq, newE
    return V, sorted(E)

def build_adj(n: int, edges):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    return adj

def boundary_vertices(level, V):
    N = 1 << level; idx = {tuple(v): i for i, v in enumerate(V)}
    b = [(N,0,0,0), (0,N,0,0), (0,0,N,0), (0,0,0,N)]
    return [idx[p] for p in b]

def distances_from_sources(adj, sources):
    n = len(adj); dist = [-1] * n; q = deque()
    for s in sources: dist[s] = 0; q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1; q.append(v)
    return np.array(dist)

def choose_central_ball(adj, level, radius):
    V, _ = st_graph_exact(level)
    bnd = boundary_vertices(level, V)
    dist_to_bnd = distances_from_sources(adj, bnd)
    center = int(np.argmax(dist_to_bnd))
    n = len(adj); dist = [-1] * n; q = deque([center]); dist[center] = 0
    nodes = [center]
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1 and dist[u] + 1 <= radius:
                dist[v] = dist[u] + 1; q.append(v); nodes.append(v)
    return center, sorted(nodes)

def induced_H1(adj, nodes):
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    H = np.zeros((n, n), dtype=float)
    for u in nodes:
        for v in adj[u]:
            if v in idx:
                i, j = idx[u], idx[v]
                H[i, j] = H[j, i] = -1.0
    return H, idx

# ---------- dynamics / fitting ----------
def eigh_H(H):
    return np.linalg.eigh(H)

def psi_from_eigs(evals, evecs, x0, tgrid, J=1.0):
    phi0 = evecs.T[:, x0]
    phases = np.exp(-1j * (J * evals[:, None]) * tgrid[None, :])
    return evecs @ (phases * phi0[:, None])

def bfs_dists(adj, src):
    n = len(adj); d = [-1]*n; q = deque([src]); d[src] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] == -1:
                d[v] = d[u] + 1; q.append(v)
    return np.array(d)

def first_crossing_times(arr_t, targets, tgrid, eps):
    out = {}
    for j in targets:
        arr = arr_t[j, :]
        idxs = np.where(arr >= eps)[0]
        out[j] = float(tgrid[idxs[0]]) if len(idxs) > 0 else math.nan
    return out

def linear_fit(d, t):
    x = np.array(d, float); y = np.array(t, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2: return math.nan, math.nan, math.nan
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, a = beta
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(a), float(b), float(r2)

def bootstrap_v(d, t, B=300):
    pairs = [(d[i], t[i]) for i in range(len(d)) if np.isfinite(d[i]) and np.isfinite(t[i])]
    if len(pairs) < 3: return math.nan, (math.nan, math.nan)
    random.seed(2025)
    vs = []
    for _ in range(B):
        samp = random.choices(pairs, k=len(pairs))
        dx = [p[0] for p in samp]; tx = [p[1] for p in samp]
        a, _, _ = linear_fit(dx, tx)
        if np.isfinite(a) and a > 0:
            vs.append(1.0 / a)
    if not vs: return math.nan, (math.nan, math.nan)
    vs = sorted(vs)
    med = float(np.median(vs))
    lo = float(vs[int(0.025 * len(vs))]); hi = float(vs[int(0.975 * len(vs)) - 1])
    return med, (lo, hi)

def main():
    out_dir = Path.cwd()
    level = 5
    V, E = st_graph_exact(level)
    adj = build_adj(len(V), E)
    center, nodes = choose_central_ball(adj, level, radius=7)
    H1, idx = induced_H1(adj, nodes)
    x0 = idx[center]

    # Adj im Ball + Distanzen
    adj_ball = [[] for _ in range(len(nodes))]
    for u_glob in nodes:
        i = idx[u_glob]
        for v_glob in adj[u_glob]:
            if v_glob in idx:
                j = idx[v_glob]; adj_ball[i].append(j)
    d_ball = bfs_dists(adj_ball, x0)
    shells = sorted(set(int(d) for d in d_ball if d > 0))

    # Isotropie-Sampling
    K = 5
    targets, dists = [], []
    for d in shells[:14]:
        js = np.where(d_ball == d)[0]
        pick = js[:K] if len(js) >= K else js
        for j in pick:
            targets.append(int(j)); dists.append(int(d))

    evals, evecs = eigh_H(H1)
    epsilons = (5e-3, 1e-2, 2e-2, 5e-2)
    results = []

    for J in (0.5, 1.0, 2.0):
        tmax = 40.0 / J; T = 1201
        tgrid = np.linspace(0.0, tmax, T)
        psi = psi_from_eigs(evals, evecs, x0, tgrid, J=J)
        amp = np.abs(psi)
        fits, v_list = [], []
        for eps in epsilons:
            t_first = first_crossing_times(amp, targets, tgrid, eps)
            times = [t_first[j] for j in targets]
            a, b, r2 = linear_fit(dists, times)
            if np.isfinite(a) and a > 0:
                v = 1.0 / a
                v_list.append(v)
                fits.append({"epsilon": float(eps), "slope_a": a, "intercept_b": b, "R2": r2, "v_hat": v})
        v_mean = float(np.mean(v_list)) if v_list else math.nan
        v_sd = float(np.std(v_list, ddof=1)) if len(v_list) > 1 else math.nan

        # Bootstrap bei eps=0.02
        eps0 = 2e-2
        t_first0 = first_crossing_times(amp, targets, tgrid, eps0)
        times0 = [t_first0[j] for j in targets]
        v_med, (vlo, vhi) = bootstrap_v(np.array(dists, float), np.array(times0, float), B=300)

        results.append({
            "J": J,
            "n_ball": len(nodes),
            "deg_max": max(len(nbrs) for nbrs in adj_ball),
            "epsilons": epsilons,
            "v_mean": v_mean, "v_sd": v_sd,
            "v_boot_med": v_med, "v_boot_ci": [vlo, vhi],
            "fits": fits,
            "tgrid": [float(tgrid[0]), float(tgrid[-1]), int(T)]
        })

        # CSV für J=1, eps=0.02 (Targets)
        if abs(J - 1.0) < 1e-12:
            lines = ["distance,t_first"]
            for d, j in zip(dists, targets):
                lines.append(f"{d},{t_first0[j]}")
            (out_dir / "E2_STL5_eps0p02_alltargets.csv").write_text("\n".join(lines))

    # Save JSON
    (out_dir / "E2_STL5_hardcheck.json").write_text(json.dumps({"ST_L5_ball": results}, indent=2))
    print("Saved:", out_dir / "E2_STL5_hardcheck.json")

if __name__ == "__main__":
    main()
