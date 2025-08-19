# stlib/graph_st.py
# Konstruktion der Graph-Approximanten des Sierpiński-Tetraeders (pcf, 3D-Gasket).
# Reine Graph-Sicht: Start K4, rekursiv 4 Kopien + Eck-Identifikationen (i<->j).
# Koordinaten: IFS F_i(x)=(x+v_i)/2 mit Tetraeder-Ecken v_i.
from __future__ import annotations
import math
from collections import defaultdict, deque

import numpy as np

__all__ = ["build_st_graph", "build_st_level", "outer_corners", "coords_from_addresses"]

def _k4():
    nodes = ['0','1','2','3']
    adj = {u:set() for u in nodes}
    for i in range(4):
        for j in range(i+1,4):
            a,b = str(i),str(j)
            adj[a].add(b); adj[b].add(a)
    corners = {0:'0',1:'1',2:'2',3:'3'}
    return adj, corners

def _prefix_copy(adj, prefix):
    new = {}
    for u, nbrs in adj.items():
        new[prefix+u] = set(prefix+v for v in nbrs)
    return new

def _merge_nodes(adj, keep, remove):
    if keep == remove: return
    # Verbinde Nachbarn von remove mit keep
    for v in list(adj[remove]):
        adj[v].discard(remove)
        if v != keep:
            adj[v].add(keep)
            adj[keep].add(v)
    # Entferne Knoten
    del adj[remove]

def build_st_level(m:int):
    """Erzeuge Adjazenz (als dict[str,set[str]]) und Corner-Labels für Level m (m>=0)."""
    if m<0: raise ValueError("m>=0")
    adj, corners = _k4()  # Level 0
    for level in range(1, m+1):
        # 4 Kopien mit Präfix 0..3
        copies = [ _prefix_copy(adj, str(p)) for p in range(4) ]
        # Vereinigung
        new_adj = {}
        for cp in copies:
            for u, nbrs in cp.items():
                if u not in new_adj: new_adj[u] = set()
                new_adj[u].update(nbrs)
        # Ecken der alten Stufe in jeder Kopie
        old_corners = corners
        new_corners = {}
        # Verklebung: für i<j: identifiziere (i + old_corners[j]) ~ (j + old_corners[i])
        for i in range(4):
            for j in range(i+1,4):
                a = f"{i}{old_corners[j]}"
                b = f"{j}{old_corners[i]}"
                # Merge b into a
                if b not in new_adj or a not in new_adj:
                    # sollte extrem selten sein; Robustheit
                    continue
                _merge_nodes(new_adj, a, b)
        # Neue äußere Ecken: (c + old_corners[c])
        for c in range(4):
            new_corners[c] = f"{c}{old_corners[c]}"
        adj, corners = new_adj, new_corners
    return adj, corners

def outer_corners(corners:dict[int,str]):
    return [corners[i] for i in range(4)]

def coords_from_addresses(nodes):
    """3D-Koordinaten für Visualisierung via IFS.
       v0..v3 sind die Ecken eines regulären Tetraeders.
    """
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, math.sqrt(3)/2.0, 0.0])
    v3 = np.array([0.5, math.sqrt(3)/6.0, math.sqrt(2.0/3.0)])
    V = [v0,v1,v2,v3]
    coords = {}
    for name in nodes:
        x = np.zeros(3)
        for k,ch in enumerate(name):
            x = 0.5*(x + V[int(ch)])
        coords[name] = x
    return coords

def build_st_graph(m:int):
    """High-level: liefert (nodes_list, edges_list, corners_list, coords_dict)."""
    adj, corners = build_st_level(m)
    nodes = list(adj.keys())
    edges = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u<v: edges.add((u,v))
    coords = coords_from_addresses(nodes)
    return nodes, sorted(list(edges)), outer_corners(corners), coords
