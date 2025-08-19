# stlib/dtn.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["dirichlet_to_neumann","perturb_edges_in_cell"]

def dirichlet_to_neumann(L, boundary_idx):
    """Berechne DtN-Map Λ: u_b -> f_b (Randfluss) für kombinatorischen L.
       Partitioniere in b (Rand) und i (Innen): L = [[L_bb, L_bi],[L_ib, L_ii]].
       Für Basenvektoren e_k auf b lösen wir u_i und berechnen f_b.
    """
    n = L.shape[0]
    boundary_idx = np.array(sorted(boundary_idx), dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[boundary_idx]=False
    interior_idx = np.where(mask)[0]
    # Blöcke
    Lbb = L[boundary_idx[:,None], boundary_idx]
    Lbi = L[boundary_idx[:,None], interior_idx]
    Lib = L[interior_idx[:,None], boundary_idx]
    Lii = L[interior_idx[:,None], interior_idx]
    solver = spla.factorized(Lii.tocsc())
    nb = len(boundary_idx)
    Lambda = np.zeros((nb,nb), dtype=float)
    for k in range(nb):
        ub = np.zeros(nb, dtype=float); ub[k]=1.0
        # Löse Innen: Lii u_i = - L_ib u_b
        rhs = - (Lib @ ub)
        ui = solver(rhs)
        fb = (Lbb @ ub) + (Lbi @ ui)
        Lambda[:,k] = np.array(fb).ravel()
    return Lambda

def perturb_edges_in_cell(nodes, edges, cell_prefix:str, factor:float):
    """Skaliere Gewichte aller Kanten, deren BEIDE Endpunkte im Adress‑Prefix liegen."""
    weights = {}
    S = set(u for u in nodes if str(u).startswith(cell_prefix))
    for (u,v) in edges:
        if (u in S) and (v in S):
            weights[(u,v)] = factor
    return weights
