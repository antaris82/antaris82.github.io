
from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, Dict, Any
import numpy as np
from datetime import datetime

def _ensure_psd(A: np.ndarray, tol: float = 1e-10) -> bool:
    w = np.linalg.eigvalsh(0.5*(A + A.conj().T))
    return np.all(w >= -tol)

def _sqrt_psd(E: np.ndarray) -> np.ndarray:
    # spectral decomposition
    w, V = np.linalg.eigh(0.5*(E + E.conj().T))
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.conj().T

# ---------- POVM by effects or Kraus ----------

def povm_measure_effects(rho: np.ndarray, effects: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Perform a POVM specified by positive 'effects' E_k, with sum_k E_k = I.
    Returns (probs, post_states) with Lüders update ρ_k = sqrt(E_k) ρ sqrt(E_k) / p_k.
    """
    D = rho.shape[0]
    I = np.eye(D, dtype=complex)
    S = sum(effects)
    # Normalize slight drift if needed
    if np.linalg.norm(S - I) > 1e-8:
        # project onto identity direction (simple renorm)
        # This is a light guard for numeric issues; not for ill-defined POVMs.
        raise ValueError("Effects do not sum to identity within tolerance.")
    probs = []
    posts = []
    for E in effects:
        if not _ensure_psd(E):
            raise ValueError("Effect not PSD.")
        M = _sqrt_psd(E)
        num = M @ rho @ M.conj().T
        p = np.real_if_close(np.trace(num))
        p = float(np.real(p))
        probs.append(p)
        if p > 0:
            posts.append(num / p)
        else:
            posts.append(num)  # zero-prob branch
    return np.array(probs), posts

def povm_measure_kraus(rho: np.ndarray, kraus_sets: Sequence[Sequence[np.ndarray]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """POVM specified by sets of Kraus operators {M_{k,α}} for outcomes k.
    Effects are E_k = Σ_α M^† M. Update: ρ_k = (Σ_α M ρ M^†)/p_k.
    """
    probs = []
    posts = []
    for Ms in kraus_sets:
        num = sum(M @ rho @ M.conj().T for M in Ms)
        p = float(np.real_if_close(np.trace(num)))
        probs.append(p)
        posts.append(num/p if p>0 else num)
    return np.array(probs), posts

# ---------- Projective measurement helpers ----------

def projective_measure_z_on_sites(rho: np.ndarray, sites: Sequence[int], N: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Projective measurement of σ_z on a set of sites; returns probs and post-states for outcomes b∈{0,1}^{|sites|}."""
    # Build projectors by tensoring |0><0| or |1><1| on selected sites, I elsewhere.
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)
    # Prepare 2^{|sites|} outcomes
    outcomes = []
    for m in range(2**len(sites)):
        bits = [(m>>k)&1 for k in range(len(sites))]
        outcomes.append(bits)
    # Build Kraus for each outcome
    from .hilbert import op_on_site, pauli
    I = pauli("I")
    kraus_ops = []
    for bits in outcomes:
        mats = [I for _ in range(N)]
        for s, b in zip(sites, bits):
            mats[s] = P0 if b==0 else P1
        # single Kraus per outcome (projector)
        K = mats[0]
        for M in mats[1:]:
            K = np.kron(K, M)
        kraus_ops.append([K])
    return povm_measure_kraus(rho, kraus_ops)

# ---------- Simple JSONL logger ----------

class EventLogger:
    def __init__(self, path: str):
        self.path = path
        # touch file
        with open(self.path, "a", encoding="utf-8") as f:
            pass

    def log(self, event_type: str, payload: Dict[str, Any]):
        rec = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "type": event_type,
            "payload": payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json_dumps_safe(rec) + "\n")

def json_dumps_safe(obj) -> str:
    import json, numpy as np
    def default(o):
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "shape": o.shape, "dtype": str(o.dtype), "data": o.tolist()}
        return str(o)
    return json.dumps(obj, default=default)
