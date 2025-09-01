
from __future__ import annotations
from typing import Callable, List, Optional, Sequence
import numpy as np

def _comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def lvne_rhs_t(rho: np.ndarray,
               H_of_t: Callable[[float], np.ndarray],
               t: float,
               L_of_t: Optional[Callable[[float], Sequence[np.ndarray]]] = None) -> np.ndarray:
    """RHS at time t: dρ/dt = -i[H(t),ρ] + Σ_k (L_k ρ L_k^† - ½{L_k^†L_k,ρ})."""
    Ht = H_of_t(t)
    rhs = -1j * _comm(Ht, rho)
    if L_of_t is not None:
        Ls = L_of_t(t)
        for L in Ls:
            rhs += L @ rho @ L.conj().T - 0.5*(L.conj().T@L@rho + rho@L.conj().T@L)
    return rhs

def lvne_rk4_t(rho0: np.ndarray,
               tlist: np.ndarray,
               H_of_t: Callable[[float], np.ndarray],
               L_of_t: Optional[Callable[[float], Sequence[np.ndarray]]] = None) -> list:
    """RK4 for time-dependent H(t), L_k(t). Returns list of ρ(t)."""
    rho = rho0.copy()
    out = [rho.copy()]
    for t0, t1 in zip(tlist[:-1], tlist[1:]):
        dt = t1 - t0
        k1 = lvne_rhs_t(rho, H_of_t, t0, L_of_t)
        k2 = lvne_rhs_t(rho + 0.5*dt*k1, H_of_t, t0 + 0.5*dt, L_of_t)
        k3 = lvne_rhs_t(rho + 0.5*dt*k2, H_of_t, t0 + 0.5*dt, L_of_t)
        k4 = lvne_rhs_t(rho + dt*k3, H_of_t, t1, L_of_t)
        rho = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        rho = 0.5*(rho + rho.conj().T)  # enforce Hermitian
        tr = np.trace(rho)
        if abs(tr-1)>1e-12:
            rho /= tr
        out.append(rho.copy())
    return out

def lindblad_from_rates_z(gamma_z: float, N: int, sites: List[int]) -> list:
    """Return dephasing Lindblad ops sqrt(gamma_z)*Z_site for given sites."""
    from .hilbert import op_on_site, pauli
    Z = pauli("Z")
    ops = []
    for s in sites:
        ops.append(np.sqrt(gamma_z) * op_on_site(Z, s, N))
    return ops
