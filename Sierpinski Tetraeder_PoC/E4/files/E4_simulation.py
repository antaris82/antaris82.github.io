
import numpy as np
import pandas as pd
from collections import deque
from itertools import combinations
import math, time

# ---------- Sierpinski-Tetraeder (ST) Graph ----------
def st_graph_exact(level:int):
    """
    Build ST graph (level) as vertices in 4-simplex integer lattice and undirected edges.
    Fixed coordinate builder (no variable shadowing).
    """
    V=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    E=set((i,j) for i,j in combinations(range(4),2))
    for m in range(1,level+1):
        shift=1<<(m-1)
        newV=[]; mapped=[]
        for i in range(4):
            e=[0,0,0,0]; e[i]=shift
            mlist=[tuple(vk+ek for vk,ek in zip(v,e)) for v in V]
            mapped.append(mlist); newV.extend(mlist)
        uniq={}; Vuniq=[]
        for vtx in newV:
            if vtx not in uniq: uniq[vtx]=len(Vuniq); Vuniq.append(vtx)
        newE=set()
        for i in range(4):
            l2g={old:uniq[mapped[i][old]] for old in range(len(V))}
            for (a,b) in E:
                na,nb=l2g[a],l2g[b]
                if na!=nb: newE.add(tuple(sorted((na,nb))))
        V,E=Vuniq,newE
    return V,sorted(E)

def build_adj(n, edges):
    adj=[[] for _ in range(n)]
    for i,j in edges:
        adj[i].append(j); adj[j].append(i)
    return adj

def boundary_vertices(level, V):
    N=1<<level; idx={tuple(v):i for i,v in enumerate(V)}
    b=[(N,0,0,0),(0,N,0,0),(0,0,N,0),(0,0,0,N)]
    return [idx[p] for p in b]

def distances_from_sources(adj, sources):
    n=len(adj); dist=[-1]*n; q=deque()
    for s in sources: dist[s]=0; q.append(s)
    while q:
        u=q.popleft()
        for v in adj[u]:
            if dist[v]==-1:
                dist[v]=dist[u]+1; q.append(v)
    return np.array(dist)

def choose_center(adj, level, V):
    b=boundary_vertices(level,V)
    d=distances_from_sources(adj,b)
    return int(np.argmax(d))

def st_ball(adj_full, center, radius):
    n=len(adj_full); dist=[-1]*n; q=deque([center]); dist[center]=0; nodes=[center]
    while q:
        u=q.popleft()
        for v in adj_full[u]:
            if dist[v]==-1 and dist[u]+1<=radius:
                dist[v]=dist[u]+1; q.append(v); nodes.append(v)
    return sorted(nodes)

def bfs_spanning_tree(adj, root):
    n=len(adj)
    parent=[-1]*n; q=deque([root]); parent[root]=-2; edges=set()
    while q:
        u=q.popleft()
        for v in adj[u]:
            if parent[v]==-1:
                parent[v]=u; edges.add(tuple(sorted((u,v)))); q.append(v)
    return sorted(edges)

# ---------- Weighted adjacency / Laplacian ----------
def build_weighted_matrices(n, edges_st, edges_tree, theta):
    A = np.zeros((n,n), dtype=float)
    for (i,j) in edges_st:
        A[i,j] += (1.0-theta); A[j,i] += (1.0-theta)
    for (i,j) in edges_tree:
        A[i,j] += theta; A[j,i] += theta
    # Normalize by max row-sum to compare time scales across theta
    scale = np.max(np.sum(np.abs(A), axis=1))
    if scale > 0:
        A /= scale
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return A, L

# ---------- E1: Heat kernel (spectral-dimension proxy) ----------
def estimate_ds_from_heatkernel(L, tmin=0.6, tmax=10.0, m=22):
    evals, evecs = np.linalg.eigh(L)
    tvals = np.exp(np.linspace(np.log(tmin), np.log(tmax), m))
    phi2 = evecs**2
    pdiag = np.array([float(np.mean(phi2 @ np.exp(-t*evals))) for t in tvals])
    lo = int(0.2*m); hi = int(0.8*m)
    x = np.log(tvals[lo:hi]); y = np.log(pdiag[lo:hi])
    Afit = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(Afit, y, rcond=None)[0]
    ds_est = -2*slope
    # R^2 (goodness of power-law fit)
    yhat = slope*x + intercept
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0
    return {"ds_est": float(ds_est), "r2_powerlaw": float(r2)}

# ---------- E2: CTQW front-speed v_* ----------
def estimate_front_speed_ctqw(A, source, adj, tmax=140.0, T=2800, eps=4e-3):
    # Hamiltonian H = -A
    H = -A
    evals, evecs = np.linalg.eigh(H)
    tgrid = np.linspace(0.0, tmax, int(T))
    # Initial |source>
    phi0 = evecs.T[:, source]
    phases = np.exp(-1j * (evals[:,None]) * tgrid[None,:])
    psi_t = evecs @ (phases * phi0[:,None])
    amp = np.abs(psi_t)

    # BFS distances in the (unweighted) subgraph
    n = A.shape[0]
    dist = [-1]*n; q=deque([source]); dist[source]=0
    while q:
        u=q.popleft()
        for v in adj[u]:
            if dist[v]==-1:
                dist[v]=dist[u]+1; q.append(v)
    dist = np.array(dist)
    Rmax = int(np.percentile(dist[dist>=0], 90))

    # First arrival times per distance shell
    arrivals = []
    for r in range(1, Rmax+1):
        idx = np.where(dist==r)[0]
        if len(idx)==0: continue
        shell_amp = np.max(amp[idx, :], axis=0)
        hits = np.where(shell_amp >= eps)[0]
        arrivals.append(float(tgrid[hits[0]])) if len(hits)>0 else arrivals.append(np.nan)

    rs = np.arange(1, Rmax+1, dtype=float)
    mask = np.isfinite(arrivals)
    rs_fit = rs[mask]; ts_fit = np.array(arrivals)[mask]
    if len(ts_fit) >= 5:
        Afit = np.vstack([ts_fit, np.ones_like(ts_fit)]).T
        slope, intercept = np.linalg.lstsq(Afit, rs_fit, rcond=None)[0]  # r ≈ v* t + b
        vstar = slope
        yhat = slope*ts_fit + intercept
        ss_res = np.sum((rs_fit - yhat)**2); ss_tot = np.sum((rs_fit - np.mean(rs_fit))**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0
    else:
        vstar = np.nan; r2 = 0.0
    return {"vstar_est": float(vstar), "r2_linear": float(r2)}

# ---------- Main experiment ----------
def run_E4(level=6, radius=7, thetas=(0.0,0.25,0.5,0.75,1.0),
           heat_t=(0.6, 10.0, 22), ctqw_tmax=140.0, ctqw_T=2800, eps=4e-3):
    V, E = st_graph_exact(level)
    adj_full = build_adj(len(V), E)
    center = choose_center(adj_full, level, V)
    nodes = st_ball(adj_full, center, radius)
    sub_idx = {v:i for i,v in enumerate(nodes)}
    nsub = len(nodes)

    # Build subgraph adjacency list and edge list
    edges_sub = []
    adj_sub = [[] for _ in range(nsub)]
    for u0 in nodes:
        i=sub_idx[u0]
        for v0 in adj_full[u0]:
            if v0 in sub_idx:
                j=sub_idx[v0]
                if i<j: edges_sub.append((i,j))
                if j not in adj_sub[i]: adj_sub[i].append(j)
                if i not in adj_sub[j]: adj_sub[j].append(i)

    # BFS spanning tree edges
    tree_edges = bfs_spanning_tree(adj_sub, sub_idx[center])

    results = []
    for theta in thetas:
        Aθ, Lθ = build_weighted_matrices(nsub, edges_sub, tree_edges, theta)
        hk = estimate_ds_from_heatkernel(Lθ, tmin=heat_t[0], tmax=heat_t[1], m=heat_t[2])
        ct = estimate_front_speed_ctqw(Aθ, sub_idx[center], adj_sub, tmax=ctqw_tmax, T=ctqw_T, eps=eps)
        results.append({
            "theta": theta,
            "n_sub": nsub,
            "ds_est": hk["ds_est"],
            "ds_r2": hk["r2_powerlaw"],
            "vstar_est": ct["vstar_est"],
            "vstar_r2": ct["r2_linear"]
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    t0=time.time()
    df = run_E4(level=6, radius=7, thetas=(0.0,0.25,0.5,0.75,1.0),
                heat_t=(0.6, 10.0, 22), ctqw_tmax=140.0, ctqw_T=2800, eps=4e-3)
    elapsed=time.time()-t0
    out_csv = "E4_homotopy_summary.csv"
    df.to_csv(out_csv, index=False)
    print("Elapsed:", round(elapsed,2), "s")
    print("Wrote", out_csv)
