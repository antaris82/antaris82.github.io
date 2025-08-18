#!/usr/bin/env python3
"""
E2 — Chain benchmark (tight-binding, CTQW).
Misst die ballistische Front via first-crossing-Zeiten t_eps(d) und schätzt v_hat.
Speichert CSV/JSON lokal (aktuelles Verzeichnis).
"""

from pathlib import Path
import json, math
import numpy as np

def chain_hamiltonian(L: int, J: float = 1.0) -> np.ndarray:
    H = np.zeros((L, L), dtype=float)
    for i in range(L - 1):
        H[i, i + 1] = H[i + 1, i] = -J
    return H

def eigh_propagator_times(H: np.ndarray, x0: int, tgrid: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    phi0 = evecs.T @ np.eye(H.shape[0])[:, x0]
    psi_t = np.empty((H.shape[0], len(tgrid)), dtype=np.complex128)
    for k, t in enumerate(tgrid):
        psi_t[:, k] = evecs @ (np.exp(-1j * evals * t) * phi0)
    return psi_t

def first_crossing_times(amplitudes: np.ndarray, targets: list[int], tgrid: np.ndarray, eps: float) -> dict[int, float]:
    t_first = {}
    for j in targets:
        arr = amplitudes[j, :]
        idxs = np.where(arr >= eps)[0]
        t_first[j] = float(tgrid[idxs[0]]) if len(idxs) > 0 else math.nan
    return t_first

def linear_fit_t_vs_d(dists: list[int], times: list[float]) -> tuple[float, float, float]:
    x = np.array(dists, dtype=float)
    y = np.array(times, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return math.nan, math.nan, math.nan
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, a = beta
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(a), float(b), float(r2)

def main():
    out_dir = Path.cwd()

    # Params (konservativ, robust)
    L, J = 121, 1.0
    x0 = L // 2
    tgrid = np.linspace(0.0, 60.0, 1601)
    dists = list(range(5, 41, 3))                 # rechte Seite ab center
    targets = [x0 + d for d in dists if x0 + d < L]
    epsilons = [5e-3, 1e-2, 2e-2, 5e-2]

    # Propagation
    H = chain_hamiltonian(L, J)
    psi = eigh_propagator_times(H, x0, tgrid)
    amp = np.abs(psi)

    v_est, fits = [], []
    for eps in epsilons:
        t_first = first_crossing_times(amp, targets, tgrid, eps)
        times = [t_first[j] for j in targets]
        a, b, r2 = linear_fit_t_vs_d(dists[:len(times)], times)
        if np.isfinite(a) and a > 0:
            v_est.append(1.0 / a)
            fits.append({"epsilon": eps, "slope_a": a, "intercept_b": b, "R2": r2, "v_hat": 1.0 / a})

    res = {
        "model": "chain",
        "L": L, "J": J,
        "v_hat_mean": float(np.mean(v_est)) if v_est else math.nan,
        "v_hat_sd": float(np.std(v_est, ddof=1)) if len(v_est) > 1 else math.nan,
        "fits": fits
    }

    # Save
    (out_dir / "E2_chain_results.json").write_text(json.dumps(res, indent=2))
    # CSV für ein repr. epsilon (0.02)
    eps0 = 2e-2
    t_first0 = first_crossing_times(amp, targets, tgrid, eps0)
    lines = ["distance,t_first"]
    for d, j in zip(dists[:len(targets)], targets):
        lines.append(f"{d},{t_first0[j]}")
    (out_dir / "E2_chain_firstcross.csv").write_text("\n".join(lines))

    print("Saved:", out_dir / "E2_chain_results.json", "and", out_dir / "E2_chain_firstcross.csv")

if __name__ == "__main__":
    main()
