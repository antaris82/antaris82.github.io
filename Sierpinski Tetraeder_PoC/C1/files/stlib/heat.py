# stlib/heat.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["hutch_trace_expm","spectral_dimension"]

def hutch_trace_expm(L, t, n_probes=16, rng=None):
    """Schätze tr(exp(-t L)) mittels Hutchinson; Rückgabe: trace_est, stderr."""
    if rng is None: rng = np.random.default_rng()
    n = L.shape[0]
    acc = []
    for _ in range(n_probes):
        z = rng.choice([-1.0,1.0], size=n).astype(float)
        y = spla.expm_multiply((-t)*L, z)  # Al‑Mohy–Higham implementiert in SciPy
        acc.append(np.dot(z,y))
    arr = np.array(acc, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)/np.sqrt(len(arr)))

def spectral_dimension(ts, pbar, eps=1e-9):
    """ds(t) = -2 d/d log t log pbar(t), zentrale Differenzen auf log‑Skala."""
    logt = np.log(ts); logp = np.log(np.maximum(pbar, eps))
    ds = np.zeros_like(pbar)
    for k in range(1,len(ts)-1):
        dt = logt[k+1]-logt[k-1]
        dp = logp[k+1]-logp[k-1]
        ds[k] = -2.0 * (dp/dt)
    ds[0]=ds[1]; ds[-1]=ds[-2]
    return ds
