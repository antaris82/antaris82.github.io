# stlib/gkls.py
# GKLS-Dephasierung für Einteilchen-Hubble auf Graphen
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

__all__ = ["gkls_dephasing_stepper"]

def gkls_dephasing_stepper(H, gamma:float, dt:float):
    """Erzeuge Zeitschritt-Funktion für d rho/dt = -i[H,rho] - gamma * offdiag(rho).
       Diskretisierung: explizit (Heun/2. Ordnung) auf Dichtematrix.
       Achtung: O(N^2) Speicher/Zeit. Für N>~300 langsam.
    """
    n = H.shape[0]
    H = H.tocsr()
    def offdiag_damp(R):
        R2 = R.copy()
        np.fill_diagonal(R2, 0.0)  # nur Offdiagonale betroffen
        return R2
    def commutator(H, R):
        # -i [H,R] = -i(HR - RH). H spärlich, R dicht
        HR = (H @ R)
        RH = (R @ H.toarray())
        return -1j*(HR - RH)
    def step(R):
        # Heun: R1 = R + dt * F(R); R2 = R + dt * F(R1); R <- 0.5*(R1+R2)
        F0 = commutator(H,R) - gamma*offdiag_damp(R)
        R1 = R + dt*F0
        F1 = commutator(H,R1) - gamma*offdiag_damp(R1)
        Rn = R + 0.5*dt*(F0+F1)
        # Leichte Numerik-Stabilisierung: Hermitisieren + Spur 1
        Rn = 0.5*(Rn + Rn.conj().T)
        tr = np.real(np.trace(Rn))
        if tr!=0: Rn = Rn / tr
        return Rn
    return step
