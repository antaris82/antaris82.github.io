# stlib/linops.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import deque

__all__ = ["laplacian_from_edges","normalized_laplacian","bfs_distances"]

def laplacian_from_edges(nodes, edges, weights=None):
    """Erzeuge kombinatorischen Laplace-Operator L = D - A (CSR).
       nodes: list hashables, edges: list of (u,v) with u!=v.
       weights: dict[(u,v)] -> w (symmetrisch), default 1.0
    """
    n = len(nodes); idx = {u:i for i,u in enumerate(nodes)}
    data=[]; rows=[]; cols=[]
    deg = np.zeros(n, dtype=float)
    if weights is None: weights = {}
    for (u,v) in edges:
        i,j = idx[u], idx[v]
        w = weights.get((u,v), weights.get((v,u), 1.0))
        # off-diagonals
        rows.extend([i,j]); cols.extend([j,i]); data.extend([-w,-w])
        deg[i]+=w; deg[j]+=w
    # Diagonal
    rows.extend(range(n)); cols.extend(range(n)); data.extend(deg.tolist())
    L = sp.csr_matrix((data,(rows,cols)), shape=(n,n))
    return L, idx

def normalized_laplacian(L:sp.csr_matrix):
    """Symmetrisch normalisierte Variante: L_norm = I - D^{-1/2} A D^{-1/2}."""
    # L = D - A  => A = D - L
    D = sp.diags(L.diagonal())
    A = D - L
    with np.errstate(divide='ignore'):
        d = np.array(D.diagonal())
        d_inv_sqrt = np.where(d>0, 1.0/np.sqrt(d), 0.0)
    D_is = sp.diags(d_inv_sqrt)
    Lnorm = sp.eye(L.shape[0]) - (D_is @ A @ D_is)
    return Lnorm.tocsr()

def bfs_distances(nodes, edges, source_idx:int):
    """Ungewichtete BFS-Distanzen vom Knoten index source_idx (0..n-1)."""
    n = len(nodes)
    # Build adjacency index list
    adj = [[] for _ in range(n)]
    idx = {u:i for i,u in enumerate(nodes)}
    for (u,v) in edges:
        i,j = idx[u], idx[v]
        adj[i].append(j); adj[j].append(i)
    dist = np.full(n, -1, dtype=int)
    q = deque([source_idx]); dist[source_idx]=0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v]<0:
                dist[v]=dist[u]+1
                q.append(v)
    return dist
