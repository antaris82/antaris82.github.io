
from __future__ import annotations
import numpy as np
import networkx as nx

def save_state_npz(path: str, rho: np.ndarray, meta: dict = None):
    np.savez_compressed(path, rho=rho, meta=np.array([meta], dtype=object))

def load_state_npz(path: str):
    data = np.load(path, allow_pickle=True)
    rho = data["rho"]
    meta = data["meta"][0].item() if "meta" in data else {}
    return rho, meta

def save_graph_graphml(path: str, G: nx.Graph):
    nx.write_graphml(G, path)

def load_graph_graphml(path: str) -> nx.Graph:
    return nx.read_graphml(path)
