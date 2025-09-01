
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

def qubit_layout(nodes: List[int]) -> Dict[int,int]:
    """Return mapping node_id -> tensor index (0..N-1) in given order."""
    return {u:i for i,u in enumerate(nodes)}

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# Pauli and identity
def pauli(name: str) -> np.ndarray:
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    if name.upper()=="I": return I
    if name.upper()=="X": return X
    if name.upper()=="Y": return Y
    if name.upper()=="Z": return Z
    raise ValueError("pauli expects one of I,X,Y,Z")

def op_on_site(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """Lift single‑site operator to N‑site tensor by Kron with identities."""
    mats = [pauli("I") for _ in range(N)]
    mats[site] = op
    return kron_all(mats)

def basis_state(bits: List[int]) -> np.ndarray:
    """Return |b_0 ... b_{N-1}> as a column vector (2^N,1)."""
    N = len(bits)
    vec = np.array([1.0+0j])
    for b in bits:
        e0 = np.array([1,0], dtype=complex)
        e1 = np.array([0,1], dtype=complex)
        vec = np.kron(vec, e0 if b==0 else e1)
    return vec.reshape((-1,1))

def random_density_matrix(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(dim,dim)) + 1j*rng.normal(size=(dim,dim))
    rho = X @ X.conj().T
    rho = rho / np.trace(rho)
    return rho

def build_ising_hamiltonian(nodes: List[int], edges: List[Tuple[int,int]],
                            Jz: float = 1.0, hx: float = 0.0) -> np.ndarray:
    """H = sum_{(i,j)} Jz Z_i Z_j + sum_i hx X_i"""
    N = len(nodes); idx = {u:i for i,u in enumerate(nodes)}
    H = np.zeros((2**N, 2**N), dtype=complex)
    Z = pauli("Z"); X = pauli("X")
    for (u,v) in edges:
        i,j = idx[u], idx[v]
        H += Jz * (op_on_site(Z,i,N) @ op_on_site(Z,j,N))
    if abs(hx)>0:
        for u in nodes:
            i = idx[u]
            H += hx * op_on_site(X,i,N)
    return H

def build_xx_hamiltonian(nodes: List[int], edges: List[Tuple[int,int]], J: float = 1.0) -> np.ndarray:
    """H = J * sum_{(i,j)} (X_i X_j + Y_i Y_j)"""
    N = len(nodes); idx = {u:i for i,u in enumerate(nodes)}
    H = np.zeros((2**N, 2**N), dtype=complex)
    X = pauli("X"); Y = pauli("Y")
    for (u,v) in edges:
        i,j = idx[u], idx[v]
        H += J * (op_on_site(X,i,N) @ op_on_site(X,j,N) + op_on_site(Y,i,N) @ op_on_site(Y,j,N))
    return H
