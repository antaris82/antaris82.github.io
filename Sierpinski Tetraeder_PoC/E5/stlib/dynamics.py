
from __future__ import annotations
import numpy as np

def _comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def lvne_rhs(rho: np.ndarray, H: np.ndarray, lindblad_ops=None) -> np.ndarray:
    """Right‑hand side of dρ/dt = -i[H,ρ] + sum_k (L_k ρ L_k^† - 1/2 {L_k^† L_k, ρ})."""
    rhs = -1j * _comm(H, rho)
    if lindblad_ops:
        for L in lindblad_ops:
            rhs += L @ rho @ L.conj().T - 0.5*(L.conj().T@L@rho + rho@L.conj().T@L)
    return rhs

def lvne_rk4(rho0: np.ndarray, H: np.ndarray, tlist, lindblad_ops=None) -> list:
    """Integrate Liouville–von Neumann equation via explicit RK4.
    Returns list of ρ(t) (including initial at tlist[0])."""
    rho = rho0.copy()
    out = [rho.copy()]
    for t0, t1 in zip(tlist[:-1], tlist[1:]):
        dt = t1 - t0
        k1 = lvne_rhs(rho, H, lindblad_ops)
        k2 = lvne_rhs(rho + 0.5*dt*k1, H, lindblad_ops)
        k3 = lvne_rhs(rho + 0.5*dt*k2, H, lindblad_ops)
        k4 = lvne_rhs(rho + dt*k3, H, lindblad_ops)
        rho = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # enforce Hermiticity + positivity (soft projection)
        rho = 0.5*(rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr-1)>1e-12: rho /= tr
        out.append(rho.copy())
    return out

def time_evolve_unitary(rho0: np.ndarray, H: np.ndarray, tlist) -> list:
    """Unitary evolution ρ(t) = U ρ U^† with U = exp(-i H Δt) stepwise via eigendecomp."""
    rho = rho0.copy()
    out = [rho.copy()]
    w, V = np.linalg.eigh(H)  # Hermitian assumed
    for t0, t1 in zip(tlist[:-1], tlist[1:]):
        dt = t1 - t0
        phases = np.exp(-1j * w * dt)
        U = (V * phases) @ V.conj().T
        rho = U @ rho @ U.conj().T
        out.append(rho.copy())
    return out

def add_dephasing_lindblad(gamma_z: float, N: int, site: int) -> np.ndarray:
    """Return L = sqrt(gamma_z) Z_site Lindblad operator for dephasing on one site."""
    from .hilbert import op_on_site, pauli
    Z = pauli("Z")
    return np.sqrt(gamma_z) * op_on_site(Z, site, N)
