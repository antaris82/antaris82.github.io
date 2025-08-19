# stlib/graphs_regular.py
# 2D/3D-Gitter (mit/ohne Periodizität) als ungewichteter Graph.
from __future__ import annotations
from collections import defaultdict

__all__ = ["grid_graph_2d", "grid_graph_3d"]

def grid_graph_2d(nx:int, ny:int, periodic:bool=False):
    adj = {}
    for i in range(nx):
        for j in range(ny):
            u = (i,j)
            if u not in adj: adj[u]=set()
            # Nachbarn
            for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ii, jj = i+di, j+dj
                if periodic:
                    ii %= nx; jj %= ny
                    v=(ii,jj)
                    adj[u].add(v)
                    adj.setdefault(v,set()).add(u)
                else:
                    if 0<=ii<nx and 0<=jj<ny:
                        v=(ii,jj)
                        adj[u].add(v)
                        adj.setdefault(v,set()).add(u)
    nodes = list(adj.keys())
    edges = set()
    for u,nbrs in adj.items():
        for v in nbrs:
            if u<v: edges.add((u,v))
    return nodes, sorted(list(edges))

def grid_graph_3d(nx:int, ny:int, nz:int, periodic:bool=False):
    adj = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                u=(i,j,k); adj.setdefault(u,set())
                for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ii,jj,kk = i+di, j+dj, k+dk
                    if periodic:
                        ii%=nx; jj%=ny; kk%=nz
                        v=(ii,jj,kk)
                        adj[u].add(v); adj.setdefault(v,set()).add(u)
                    else:
                        if 0<=ii<nx and 0<=jj<ny and 0<=kk<nz:
                            v=(ii,jj,kk)
                            adj[u].add(v); adj.setdefault(v,set()).add(u)
    nodes = list(adj.keys())
    edges = set()
    for u,nbrs in adj.items():
        for v in nbrs:
            if u<v: edges.add((u,v))
    return nodes, sorted(list(edges))
