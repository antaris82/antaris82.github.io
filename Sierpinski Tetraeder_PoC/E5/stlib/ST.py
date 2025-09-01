# ST_min.py — Minimal Sierpinski‑Tetrahedron (ST) graph generator
# Only dependency: networkx (and numpy for coordinates)
#
# Public API:
#   build_graph(level: int) -> networkx.Graph
#
# Notes:
#   • L=0 returns K4 (regular tetrahedron skeleton).
#   • For L>=1, graph is generated via IFS replication with vertex merging.
#   • Each node has attribute "pos" = np.ndarray shape (3,) for plotting.

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import networkx as nx

# Regular tetrahedron corners (edge length = 1)
V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2, 0.0])
V3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])
V  = [V0, V1, V2, V3]

def _key_of(pos: np.ndarray, scale: int) -> Tuple[int,int,int]:
    """Snap a floating position to an integer grid at resolution `scale` (2^k).
    Ensures exact merging of coincident vertices across IFS copies."""
    arr = np.rint(pos * scale).astype(int)
    return int(arr[0]), int(arr[1]), int(arr[2])

def build_graph(level: int) -> nx.Graph:
    """Return Sierpinski‑Tetrahedron graph at refinement `level` (>=0).
    
    Construction:
      Start with G0 = K4 on vertices V0..V3. For each step, apply the four
      contractions f_i(x)=(x+V_i)/2 to all vertices/edges, merge coincident
      vertices (using integer grid snapping with denominator 2^k), and take
      the union of edges.
    """
    L = int(level)
    if L < 0:
        raise ValueError("level must be >= 0")

    # Base graph G0 = K4 with positions
    G = nx.Graph()
    for i, P in enumerate(V):
        G.add_node(i, pos=P.copy())
    G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])

    scale = 1  # denominator = 2^0

    for _ in range(L):
        scale *= 2  # after one contraction step
        new_G = nx.Graph()
        key_to_id: Dict[Tuple[int,int,int], int] = {}
        id_to_pos: Dict[int, np.ndarray] = {}

        # Build four copies, merge vertices by snapped key
        for Vi in V:
            map_old_to_new: Dict[int, int] = {}
            for u, data in G.nodes(data=True):
                pos_u = data["pos"]
                new_pos = (pos_u + Vi) * 0.5
                key = _key_of(new_pos, scale)
                if key not in key_to_id:
                    nid = len(key_to_id)
                    key_to_id[key] = nid
                    id_to_pos[nid] = new_pos
                else:
                    nid = key_to_id[key]
                map_old_to_new[u] = nid

            for a, b in G.edges():
                na, nb = map_old_to_new[a], map_old_to_new[b]
                if na != nb:
                    new_G.add_edge(na, nb)

        # Finalize positions
        for nid, p in id_to_pos.items():
            new_G.nodes[nid]["pos"] = p
        G = new_G

    return G

if __name__ == "__main__":
    for L in range(0, 5):
        G = build_graph(L)
        print(f"L={L}: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
