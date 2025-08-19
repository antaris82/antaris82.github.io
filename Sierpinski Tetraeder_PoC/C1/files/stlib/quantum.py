# stlib/quantum.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["unitary_packet","lr_front_radius"]

def unitary_packet(H, psi0_idx:int, times, normalize=True):
    """Zeitentwicklung |psi(t)>=e^{-i t H}|psi0>. H: CSR (hermitesch, reell genügt).
       Rückgabe: array shape (len(times), N) der Wahrscheinlichkeiten.
    """
    n = H.shape[0]
    psi0 = np.zeros(n, dtype=complex); psi0[psi0_idx]=1.0
    out = np.empty((len(times), n), dtype=float)
    for a, t in enumerate(times):
        y = spla.expm_multiply((-1j*t)*H, psi0)
        if normalize:
            nrm = np.linalg.norm(y); 
            if nrm>0: y = y / nrm
        out[a,:] = np.real(y*np.conj(y))
    return out

def lr_front_radius(prob_t, distances, quantile=0.9):
    """Quantils‑Radius der Verteilung vs. Distanz (als „Front“)."""
    radii = []
    for p in prob_t:
        # gruppiere nach dist
        dmax = distances.max()
        mass = np.zeros(dmax+1, dtype=float)
        for i,di in enumerate(distances):
            if di>=0: mass[di]+=p[i]
        c = np.cumsum(mass)
        r = np.searchsorted(c, quantile)
        radii.append(r)
    return np.array(radii, dtype=float)
