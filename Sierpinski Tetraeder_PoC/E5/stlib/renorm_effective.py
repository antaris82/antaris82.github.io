
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
import numpy as np

from .hilbert import pauli, op_on_site
from .trace import partial_trace

# ---------- Utilities for tensor permutations ----------

def _permute_operator_qubits(H: np.ndarray, perm: List[int]) -> np.ndarray:
    """Permute qubit ordering of operator H (2^N x 2^N) by 'perm' on both bra/ket.
    perm maps NEW index -> OLD index (i_new = i_old[perm[...]])? We choose:
    After permutation, the *new* position k holds what was previously at position perm[k].
    """
    N = int(np.log2(H.shape[0]))
    assert H.shape == (2**N, 2**N)
    # reshape into 2N axes and transpose
    X = H.reshape(*([2]*N + [2]*N))
    # Apply same permutation to bra and ket axes
    axes = list(perm) + [N + p for p in perm]
    Xp = np.transpose(X, axes=axes)
    return Xp.reshape(2**N, 2**N)

def _kron_list(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# ---------- Block isometries W: (2^m -> 2) ----------

def isometry_majority(m: int) -> np.ndarray:
    """Map 2^m -> 2 by encoding majority of Z-basis bits.
    |0_L> ~ sum_{|x|<m/2} |x>, |1_L> ~ sum_{|x|>m/2} |x>, with Gram-Schmidt.
    Ties (|x|=m/2) split evenly. Returns W with shape (2**m, 2) and W^†W = I_2.
    """
    # Build vectors in computational basis
    vec0 = np.zeros((2**m,1), dtype=complex)
    vec1 = np.zeros((2**m,1), dtype=complex)
    for idx in range(2**m):
        pop = bin(idx).count("1")
        if pop*2 < m:  # more zeros
            vec0[idx,0] = 1.0
        elif pop*2 > m:  # more ones
            vec1[idx,0] = 1.0
        else:
            vec0[idx,0] += 0.5**0.5
            vec1[idx,0] += 0.5**0.5
    # Orthonormalize
    def norm(v): 
        n = np.linalg.norm(v); 
        return v/n if n>0 else v
    v0 = norm(vec0)
    # Gram-Schmidt
    v1 = vec1 - (v0.conj().T @ vec1)*v0
    v1 = norm(v1)
    W = np.hstack([v0, v1])  # (2^m, 2)
    return W

def isometry_parity(m: int) -> np.ndarray:
    """Map by even/odd parity in Z-basis. Orthonormal by construction."""
    psi_even = np.zeros((2**m,1), dtype=complex)
    psi_odd  = np.zeros((2**m,1), dtype=complex)
    for idx in range(2**m):
        if bin(idx).count("1") % 2 == 0:
            psi_even[idx,0] = 1.0
        else:
            psi_odd[idx,0] = 1.0
    psi_even /= np.linalg.norm(psi_even)
    psi_odd  /= np.linalg.norm(psi_odd)
    return np.hstack([psi_even, psi_odd])

def isometry_magnetization_extrema(m: int) -> np.ndarray:
    """Map to |0...0> and |1...1> (extremal magnetizations)."""
    e0 = np.zeros((2**m,1), dtype=complex); e0[0,0] = 1.0
    e1 = np.zeros((2**m,1), dtype=complex); e1[-1,0] = 1.0
    return np.hstack([e0, e1])

def build_block_isometry(m: int, method: str = "parity") -> np.ndarray:
    method = method.lower()
    if method == "parity": return isometry_parity(m)
    if method == "majority": return isometry_majority(m)
    if method == "magnetization": return isometry_magnetization_extrema(m)
    raise ValueError("Unknown isometry method. Choose: parity, majority, magnetization.")

# ---------- Effective Hamiltonian via block isometries ----------

def renormalize_hamiltonian_by_blocks(
    H: np.ndarray,
    nodes: List[int],
    blocks: Dict[int, str],
    method: str = "parity",
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """Compute H_eff = W^† H' W where:
       • nodes: tensor ordering of qubits in H (length N)
       • blocks: node->block label (strings); block order = sorted unique labels
       • method: isometry choice for each block (2^m -> 2)

    Returns:
       H_eff (2^M x 2^M), block_labels (sorted), block_members mapping.

    Note: This function *reorders* H so that qubits are grouped by block internally.
    """
    N = len(nodes)
    # 1) Determine block order and members (in given nodes order)
    block_labels = sorted({blocks[u] for u in nodes})
    block_members = {bl: [u for u in nodes if blocks[u]==bl] for bl in block_labels}
    # 2) Build permutation that groups nodes by block
    nodes_grouped = [u for bl in block_labels for u in block_members[bl]]
    # perm maps new index -> old position; compute indices
    pos = {u:i for i,u in enumerate(nodes)}
    perm = [pos[u] for u in nodes_grouped]
    # 3) Permute H accordingly
    Hp = _permute_operator_qubits(H, perm)
    # 4) Build block isometries and their Kron product
    W_blocks = []
    block_sizes = []
    for bl in block_labels:
        m = len(block_members[bl])
        block_sizes.append(m)
        W_blocks.append(build_block_isometry(m, method=method))  # (2^m -> 2)
    W = _kron_list(W_blocks)  # shape (2^N -> 2^M)
    # 5) Project: H_eff = W^† H' W
    Heff = W.conj().T @ Hp @ W
    return Heff, block_labels, block_members

# ---------- Map fine operators to effective block operators ----------

def renormalize_operator_on_block(op_fine: np.ndarray, m: int, method: str = "parity") -> np.ndarray:
    """Given an operator on m-qubit block (2^m x 2^m), return effective 2x2 operator W^† O W."""
    W = build_block_isometry(m, method=method)
    return W.conj().T @ op_fine @ W
