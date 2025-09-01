
from __future__ import annotations
from typing import List, Iterable, Sequence
import numpy as np

def _dims_to_shape(dims: Sequence[int]) -> int:
    n = 1
    for d in dims: n *= int(d)
    return n

def partial_trace(rho: np.ndarray, dims: Sequence[int], keep: Iterable[int]) -> np.ndarray:
    """General partial trace.
    Parameters
    ----------
    rho : ndarray shape (D,D) with D=prod(dims)
    dims : list/tuple of local dims [d0,d1,...,d_{N-1}]
    keep : indices of subsystems to keep (sorted or unsorted)
    Returns
    -------
    rho_reduced : ndarray with shape (prod(d_keep), prod(d_keep))
    """
    import numpy as np
    dims = [int(d) for d in dims]
    N = len(dims)
    keep = sorted(list(set(int(i) for i in keep)))
    trace = [i for i in range(N) if i not in keep]
    # reshape to (d0,...,dN, d0,...,dN)
    rho_reshaped = rho.reshape(*(dims + dims))
    # trace out unwanted systems by summing over matching axes
    for t in reversed(trace):
        rho_reshaped = rho_reshaped.trace(axis1=t, axis2=t+N)
    d_keep = 1
    for i in keep: d_keep *= dims[i]
    return rho_reshaped.reshape((d_keep, d_keep))

def reduce_to_nodes(rho: np.ndarray, positions: List[int], dims: Sequence[int]) -> np.ndarray:
    """Convenience: partial trace keeping a *subset of positions* (tensor sites)."""
    return partial_trace(rho, dims, keep=positions)

def trace_out_nodes(rho: np.ndarray, positions_to_trace: List[int], dims: Sequence[int]) -> np.ndarray:
    """Partial trace tracing out given positions, i.e. keep the complement."""
    N = len(dims)
    keep = [i for i in range(N) if i not in positions_to_trace]
    return partial_trace(rho, dims, keep=keep)
