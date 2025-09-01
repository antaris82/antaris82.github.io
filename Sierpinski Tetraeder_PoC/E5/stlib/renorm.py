
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import networkx as nx
from .address import blocks_from_prefix

def coarse_grain_graph(G: nx.Graph, prefix_len: int) -> nx.Graph:
    """Block‑coarse‑grain by address prefix.
    Each block becomes a coarse node; edges exist if any fine edge crosses blocks.
    Node attribute 'members' lists fine nodes of the block.
    """
    assign = blocks_from_prefix(G, prefix_len)
    # map prefix -> new node id
    prefixes: List[str] = sorted(set(assign.values()))
    block_id: Dict[str, int] = {p:i for i,p in enumerate(prefixes)}

    H = nx.Graph()
    for p,i in block_id.items():
        members = [u for u,pp in assign.items() if pp==p]
        H.add_node(i, prefix=p, members=members)
    # edges
    for u,v in G.edges():
        pu, pv = assign[u], assign[v]
        if pu != pv:
            H.add_edge(block_id[pu], block_id[pv])
    return H

def coarse_grain_state_average(rho: "np.ndarray", nodes: List[int], blocks: Dict[int,str],
                               dims_per_node: int = 2) -> "np.ndarray":
    """Coarse‑graining channel for states: for each block, average the 1‑site
    reduced density matrices of its member nodes to a single qubit.
    Assembles the coarse state's tensor product in block order.
    This is a simple CPTP map (convex mixture of partial traces).

    Parameters
    ----------
    rho : (d^N,d^N) array, full fine‑level state of listed `nodes`.
    nodes : list of node ids (order defines tensor site order).
    blocks : mapping node->block label (strings). Only nodes in `nodes` are used.
    dims_per_node : local Hilbert dim (default qubit=2).

    Returns
    -------
    rho_coarse : (d^M, d^M) array with M=#blocks_on_nodes (ordered by sorted block labels).
    """
    import numpy as np
    from .trace import reduce_to_nodes

    # Determine block order restricted to provided nodes
    block_labels = sorted({blocks[u] for u in nodes})
    # For each block, collect members in the given `nodes` order
    block_members = [[u for u in nodes if blocks[u]==bl] for bl in block_labels]

    # Build 1-site reduced DM for each member and average per block
    def one_site_rdm(idx: int) -> np.ndarray:
        # idx refers to position in `nodes`
        return reduce_to_nodes(rho, [idx], dims_per_node*np.ones(len(nodes), dtype=int)).copy()

    rdms = []
    for members in block_members:
        member_positions = [nodes.index(u) for u in members]
        if len(member_positions)==0:
            # empty block on subset -> use maximally mixed
            r = np.eye(dims_per_node)/dims_per_node
        else:
            mats = [one_site_rdm(i) for i in member_positions]
            r = sum(mats)/len(mats)
        rdms.append(r)

    # Tensor product of coarse single‑site states
    out = rdms[0]
    for r in rdms[1:]:
        out = np.kron(out, r)
    return out
