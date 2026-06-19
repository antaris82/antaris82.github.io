#!/usr/bin/env python3
"""
Phase-density extrapolation test.

Input:
  CSV outputs from test_phase_density_scaling.py

Main outputs:
  1. theta_infty estimates for local phase density
  2. decay fits for phase variance/std
  3. decay fits for centered residual curvature
  4. kernel regime classification

No new growth simulation is run here by default. This script fits the existing
level-by-level data so the extrapolation logic is separated from the dynamics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


METHODS = ["polar", "eigen", "G_weighted"]
KEY_KERNELS = ["shell_norm_inverse_square", "exp_0p25", "critical_exp_1over3"]


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def linear_fit_feature(L: np.ndarray, y: np.ndarray, feature: np.ndarray) -> dict:
    X = np.vstack([np.ones_like(feature), feature]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return {
        "theta_inf": float(beta[0]),
        "amplitude": float(beta[1]),
        "rmse": rmse(y, yhat),
        "yhat_last": float(yhat[-1]),
    }


def fit_theta_models(L: np.ndarray, y: np.ndarray) -> List[dict]:
    rows = []

    # Exponential approach theta_inf + A r^L; grid over r.
    best = None
    for r in np.linspace(0.05, 0.995, 190):
        f = r ** L
        fit = linear_fit_feature(L, y, f)
        fit.update({"model": "theta_inf_plus_A_r_pow_L", "r": float(r), "feature": "r^L"})
        if best is None or fit["rmse"] < best["rmse"]:
            best = fit
    rows.append(best)

    features = [
        ("theta_inf_plus_A_over_L", 1.0 / L),
        ("theta_inf_plus_A_over_L2", 1.0 / (L * L)),
        ("theta_inf_plus_A_times_3pow_minus_L", 3.0 ** (-L)),
        ("theta_inf_plus_A_over_sqrt_nodes", 1.0 / np.sqrt((3.0 ** (L + 1) - 1.0) / 2.0)),
    ]
    for name, f in features:
        fit = linear_fit_feature(L, y, f)
        fit.update({"model": name, "r": float("nan"), "feature": name})
        rows.append(fit)

    return sorted(rows, key=lambda d: d["rmse"])


def fit_positive_decay(L: np.ndarray, y: np.ndarray) -> List[dict]:
    mask = np.isfinite(y) & (y > 0) & np.isfinite(L)
    L = L[mask]
    y = y[mask]
    rows = []
    if len(y) < 3:
        return rows

    logy = np.log(y)

    # y = C r^L
    X = np.vstack([np.ones_like(L), L]).T
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    logC, log_r = beta
    yhat = np.exp(X @ beta)
    rows.append({
        "model": "C_times_r_pow_L",
        "C": float(math.exp(logC)),
        "r": float(math.exp(log_r)),
        "exponent": float("nan"),
        "rmse": rmse(y, yhat),
        "last_pred": float(yhat[-1]),
    })

    # y = C L^-p
    logL = np.log(L)
    X = np.vstack([np.ones_like(logL), logL]).T
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    logC, slope = beta
    p = -slope
    yhat = np.exp(X @ beta)
    rows.append({
        "model": "C_times_L_minus_p",
        "C": float(math.exp(logC)),
        "r": float("nan"),
        "exponent": float(p),
        "rmse": rmse(y, yhat),
        "last_pred": float(yhat[-1]),
    })

    # y = C N^-alpha, with N=(3^(L+1)-1)/2
    N = (3.0 ** (L + 1) - 1.0) / 2.0
    logN = np.log(N)
    X = np.vstack([np.ones_like(logN), logN]).T
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    logC, slope = beta
    alpha = -slope
    yhat = np.exp(X @ beta)
    rows.append({
        "model": "C_times_N_minus_alpha",
        "C": float(math.exp(logC)),
        "r": float("nan"),
        "exponent": float(alpha),
        "rmse": rmse(y, yhat),
        "last_pred": float(yhat[-1]),
    })

    return sorted(rows, key=lambda d: d["rmse"])


def kernel_regime(kernel: str) -> str:
    mapping = {
        "inverse_square": "locally decaying but globally accumulating/supercritical",
        "exp_0p40": "supercritical exponential remote kernel (3ρ > 1)",
        "critical_exp_1over3": "critical remote kernel (3ρ ≈ 1)",
        "exp_0p25": "subcritical exponential remote kernel (3ρ < 1)",
        "shell_norm_inverse_square": "shell-normalized local/subcritical candidate",
    }
    return mapping.get(kernel, "unknown")


def load_global_density(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "phase_density_by_parent_level.csv")
    return df[df["parent_level"] == -1].copy()


def load_global_residuals(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "phase_loop_residuals_by_method_level.csv")
    # This file is already summarized over all loop modes by kernel/method/global_level.
    # If it contains method+global level only, loop_mode absent; keep generic.
    return df.copy()


def fit_phase_density(global_density: pd.DataFrame, tail_start: int) -> List[dict]:
    rows = []
    for kernel in sorted(global_density["kernel"].unique()):
        kdf_all = global_density[global_density["kernel"] == kernel].sort_values("global_level")
        kdf = kdf_all[kdf_all["global_level"] >= tail_start].copy()
        if len(kdf) < 4:
            kdf = kdf_all.copy()
        L = kdf["global_level"].to_numpy(dtype=float)
        for method in METHODS:
            mean_col = f"{method}_mean_linear_deg"
            std_col = f"{method}_std_linear_deg"
            if mean_col not in kdf or std_col not in kdf:
                continue
            y = kdf[mean_col].to_numpy(dtype=float)
            theta_models = fit_theta_models(L, y)
            best_theta = theta_models[0]
            for rank, fit in enumerate(theta_models[:3], start=1):
                rows.append({
                    "kernel": kernel,
                    "method": method,
                    "fit_kind": "theta_mean",
                    "rank": rank,
                    "model": fit["model"],
                    "theta_inf": fit["theta_inf"],
                    "amplitude": fit["amplitude"],
                    "r": fit.get("r", float("nan")),
                    "decay_exponent": float("nan"),
                    "rmse": fit["rmse"],
                    "last_observed": float(y[-1]),
                    "last_pred": fit["yhat_last"],
                    "fit_tail_start": int(L[0]),
                    "regime": kernel_regime(kernel),
                })
            std_vals = kdf[std_col].to_numpy(dtype=float)
            std_models = fit_positive_decay(L, std_vals)
            for rank, fit in enumerate(std_models[:3], start=1):
                rows.append({
                    "kernel": kernel,
                    "method": method,
                    "fit_kind": "phase_std_decay",
                    "rank": rank,
                    "model": fit["model"],
                    "theta_inf": float("nan"),
                    "amplitude": fit["C"],
                    "r": fit.get("r", float("nan")),
                    "decay_exponent": fit.get("exponent", float("nan")),
                    "rmse": fit["rmse"],
                    "last_observed": float(std_vals[-1]),
                    "last_pred": fit["last_pred"],
                    "regime": kernel_regime(kernel),
                })
    return rows


def fit_residuals(global_resid: pd.DataFrame, tail_start: int) -> List[dict]:
    rows = []
    # Expected columns from phase_loop_residuals_by_method_level.csv:
    # kernel, method, global_level, mean_abs_centered_level_deg, ...
    for kernel in sorted(global_resid["kernel"].unique()):
        for method in METHODS:
            rdf_all = global_resid[(global_resid["kernel"] == kernel) & (global_resid["method"] == method)].sort_values("global_level")
            rdf = rdf_all[rdf_all["global_level"] >= tail_start].copy()
            if len(rdf) < 3:
                rdf = rdf_all.copy()
            if len(rdf) < 3:
                continue
            L = rdf["global_level"].to_numpy(dtype=float)
            for col, kind in [
                ("mean_abs_centered_level_deg", "centered_residual_level_decay"),
                ("mean_abs_centered_global_deg", "centered_residual_global_decay"),
            ]:
                if col not in rdf:
                    continue
                y = rdf[col].to_numpy(dtype=float)
                models = fit_positive_decay(L, y)
                for rank, fit in enumerate(models[:3], start=1):
                    rows.append({
                        "kernel": kernel,
                        "method": method,
                        "fit_kind": kind,
                        "rank": rank,
                        "model": fit["model"],
                        "theta_inf": float("nan"),
                        "amplitude": fit["C"],
                        "r": fit.get("r", float("nan")),
                        "decay_exponent": fit.get("exponent", float("nan")),
                        "rmse": fit["rmse"],
                        "last_observed": float(y[-1]),
                        "last_pred": fit["last_pred"],
                        "regime": kernel_regime(kernel),
                    })
    return rows


def make_summary(rows: List[dict]) -> str:
    df = pd.DataFrame(rows)
    lines = []
    lines.append("PHASE DENSITY EXTRAPOLATION TEST")
    lines.append("")
    lines.append("KEY KERNELS — best polar theta fits")
    for kernel in KEY_KERNELS:
        sub = df[(df.kernel == kernel) & (df.method == "polar") & (df.fit_kind == "theta_mean") & (df["rank"] == 1)]
        if len(sub):
            r = sub.iloc[0]
            lines.append(
                f"  {kernel}: theta_inf={r.theta_inf:.9f} deg, "
                f"last={r.last_observed:.9f}, model={r.model}, rmse={r.rmse:.6e}, regime={r.regime}"
            )
    lines.append("")
    lines.append("KEY KERNELS — best polar phase-std decay")
    for kernel in KEY_KERNELS:
        sub = df[(df.kernel == kernel) & (df.method == "polar") & (df.fit_kind == "phase_std_decay") & (df["rank"] == 1)]
        if len(sub):
            r = sub.iloc[0]
            if np.isfinite(r.r):
                decay = f"r={r.r:.6f}"
            else:
                decay = f"exponent={r.decay_exponent:.6f}"
            lines.append(
                f"  {kernel}: last_std={r.last_observed:.9f}, model={r.model}, {decay}, rmse={r.rmse:.6e}"
            )
    lines.append("")
    lines.append("KEY KERNELS — best polar centered residual decay")
    for kernel in KEY_KERNELS:
        sub = df[(df.kernel == kernel) & (df.method == "polar") & (df.fit_kind == "centered_residual_level_decay") & (df["rank"] == 1)]
        if len(sub):
            r = sub.iloc[0]
            if np.isfinite(r.r):
                decay = f"r={r.r:.6f}"
            else:
                decay = f"exponent={r.decay_exponent:.6f}"
            lines.append(
                f"  {kernel}: last_residual={r.last_observed:.9f}, model={r.model}, {decay}, rmse={r.rmse:.6e}"
            )

    lines.append("")
    lines.append("ALL KERNELS — best polar theta_inf")
    for kernel in sorted(df.kernel.unique()):
        sub = df[(df.kernel == kernel) & (df.method == "polar") & (df.fit_kind == "theta_mean") & (df["rank"] == 1)]
        if len(sub):
            r = sub.iloc[0]
            lines.append(f"  {kernel}: theta_inf={r.theta_inf:.9f} deg, last={r.last_observed:.9f}, model={r.model}, rmse={r.rmse:.3e}")

    return "\n".join(lines)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(input_dir: Path, outdir: Path, tail_start: int) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    global_density = load_global_density(input_dir)
    global_resid = load_global_residuals(input_dir)

    rows = []
    rows.extend(fit_phase_density(global_density, tail_start))
    rows.extend(fit_residuals(global_resid, tail_start))
    for r in rows:
        r.setdefault("fit_tail_start", tail_start)
    write_csv(outdir / "phase_density_extrapolation_fits.csv", rows)

    summary = make_summary(rows)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("/mnt/data/phase_density_scaling_out_L8"))
    ap.add_argument("--outdir", type=Path, default=Path("phase_density_extrapolation_out"))
    ap.add_argument("--tail-start", type=int, default=4)
    args = ap.parse_args()
    print(run(args.input_dir, args.outdir, args.tail_start))


if __name__ == "__main__":
    main()
