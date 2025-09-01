
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import networkx as nx

def build_address_tree(G: nx.Graph) -> Dict[str, List[int]]:
    """Return mapping address -> list of node ids carrying this address."""
    addr_map: Dict[str, List[int]] = {}
    for u, data in G.nodes(data=True):
        addrs = data.get("addresses", None)
        if not addrs:
            continue
        for a in addrs:
            addr_map.setdefault(a, []).append(u)
    return addr_map

def node_primary_address(G: nx.Graph, node: int) -> str:
    """Pick a stable representative address (lexicographically minimal)."""
    addrs = G.nodes[node].get("addresses", None)
    if not addrs:
        raise ValueError("Graph has no 'addresses' on nodes; use build_graph_with_addresses.")
    return sorted(addrs)[0]

def blocks_from_prefix(G: nx.Graph, prefix_len: int) -> Dict[int, str]:
    """Assign each node to a block by truncating its primary address to prefix_len.
    Returns mapping node -> prefix string."""
    blocks: Dict[int, str] = {}
    for u in G.nodes():
        a = node_primary_address(G, u)
        blocks[u] = a[:prefix_len] if prefix_len <= len(a) else a
    return blocks
