
from __future__ import annotations
from typing import Dict, Tuple, List, Set, Optional
import numpy as np
import networkx as nx
import importlib.util
import os

# Regular tetrahedron corners (edge length = 1)
V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2, 0.0])
V3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])
V  = [V0, V1, V2, V3]

def _key_of(pos: np.ndarray, scale: int) -> Tuple[int,int,int]:
    arr = np.rint(pos * scale).astype(int)
    return int(arr[0]), int(arr[1]), int(arr[2])

def _internal_build_graph(level: int) -> nx.Graph:
    """Minimal ST generator; keeps only 'pos' attribute."""
    L = int(level)
    if L < 0:
        raise ValueError("level must be >= 0")
    G = nx.Graph()
    for i, P in enumerate(V):
        G.add_node(i, pos=P.copy())
    G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    scale = 1
    for _ in range(L):
        scale *= 2
        new_G = nx.Graph()
        key_to_id: Dict[Tuple[int,int,int], int] = {}
        id_to_pos: Dict[int, np.ndarray] = {}
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
        for nid, p in id_to_pos.items():
            new_G.nodes[nid]["pos"] = p
        G = new_G
    return G

def build_graph(level: int, prefer_external: bool = True,
                external_path: Optional[str] = "/mnt/data/ST.py") -> nx.Graph:
    """Build ST graph. If an external ST.py exposing build_graph(level) exists,
    use it; otherwise fall back to the internal implementation."""
    if prefer_external and external_path and os.path.exists(external_path):
        try:
            spec = importlib.util.spec_from_file_location("ST_external", external_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "build_graph"):
                return mod.build_graph(level)  # type: ignore[attr-defined]
        except Exception as e:
            # Fall through to internal if external import fails
            pass
    return _internal_build_graph(level)

def build_graph_with_addresses(level: int) -> nx.Graph:
    """Build ST graph and attach address sets for each node.
    A node may have multiple addresses due to vertex merging (shared boundaries).
    Node attributes:
      - pos: np.ndarray(3,)
      - addresses: set[str] of length L words over alphabet {'0','1','2','3'}
                   representing IFS maps applied in order.
    """
    L = int(level)
    if L < 0: raise ValueError("level must be >= 0")
    # Start level 0 with four vertices labeled by single‑letter addresses
    G = nx.Graph()
    for i, P in enumerate(V):
        G.add_node(i, pos=P.copy(), addresses={str(i)})
    G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    scale = 1
    for step in range(L):
        scale *= 2
        new_G = nx.Graph()
        key_to_id: Dict[Tuple[int,int,int], int] = {}
        id_to_pos: Dict[int, np.ndarray] = {}
        id_to_addr: Dict[int, Set[str]] = {}
        for Vi_idx, Vi in enumerate(V):
            for u, data in G.nodes(data=True):
                pos_u = data["pos"]
                new_pos = (pos_u + Vi) * 0.5
                key = _key_of(new_pos, scale)
                if key not in key_to_id:
                    nid = len(key_to_id)
                    key_to_id[key] = nid
                    id_to_pos[nid] = new_pos
                    id_to_addr[nid] = set()
                nid = key_to_id[key]
                # extend every existing address by current copy index
                for a in data.get("addresses", {""}):
                    id_to_addr[nid].add(a + str(Vi_idx))
        # edges
        for Vi in V:
            # build a temporary mapping for this Vi to add edges
            map_old_to_new: Dict[int,int] = {}
            for u, data in G.nodes(data=True):
                new_pos = (data["pos"] + Vi) * 0.5
                key = _key_of(new_pos, scale)
                map_old_to_new[u] = key_to_id[key]
            for a, b in G.edges():
                na, nb = map_old_to_new[a], map_old_to_new[b]
                if na != nb:
                    new_G.add_edge(na, nb)
        # finalize
        for nid, p in id_to_pos.items():
            new_G.add_node(nid, pos=p, addresses=id_to_addr[nid])
        G = new_G
    return G
