#!/usr/bin/env python3
"""
Effective operator sector-closure test using the shell-normalized kernel.

This script imports the growth model from test_conductance_scaling_generalization.py
and measures whether the local skew response operator has a stable derived
invariant 2D plane.

The key distinction:
- Any nonzero 3x3 skew matrix has an exact axis-orthogonal J-plane.
- The nontrivial question is whether that axis is stable/canonical enough,
  whether it aligns with the standard sibling sum-zero sector, and whether
  the symmetric part is usable as a metric candidate.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import List, Dict

import numpy as np

EPS = 1e-12


def load_scaling_model():
    path = Path("/mnt/data/test_conductance_scaling_generalization.py")
    spec = importlib.util.spec_from_file_location("scaling", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ScalingModel


def column_stochastic(M: np.ndarray) -> np.ndarray:
    P = M.copy().astype(float)
    for j in range(3):
        s = P[:, j].sum()
        if s > EPS:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    return P


def standard_basis() -> np.ndarray:
    u1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    u2 = np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)
    return np.vstack([u1, u2]).T


def skew_axis(A: np.ndarray) -> np.ndarray:
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def plane_basis_from_axis(axis: np.ndarray) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + EPS)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, a) * a
    u = u / (np.linalg.norm(u) + EPS)
    v = np.cross(a, u)
    v = v / (np.linalg.norm(v) + EPS)
    return np.vstack([u, v]).T


def local_diag(M: np.ndarray) -> dict:
    P = column_stochastic(M)
    S = 0.5 * (P + P.T)
    A = 0.5 * (P - P.T)
    ev = np.linalg.eigvals(P)
    complex_pair = float(np.max(np.abs(np.imag(ev))) > 1e-9)

    c = np.ones(3) / math.sqrt(3.0)
    Bstd = standard_basis()
    Astd = Bstd.T @ A @ Bstd
    alpha_std = math.sqrt(max(0.0, float(-np.trace(Astd @ Astd) / 2.0)))
    if alpha_std > EPS:
        Jstd = Astd / alpha_std
        std_J2_resid = float(np.linalg.norm(Jstd @ Jstd + np.eye(2)))
    else:
        std_J2_resid = float("inf")
    std_leakage = float(np.linalg.norm(c.T @ A @ Bstd))

    axis = skew_axis(A)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= EPS:
        return {
            "complex_pair": complex_pair,
            "std_leakage": std_leakage,
            "std_J2_resid": std_J2_resid,
            "axis_norm": axis_norm,
            "axis_x": float("nan"),
            "axis_y": float("nan"),
            "axis_z": float("nan"),
            "axis_align_const": float("nan"),
            "derived_J2_resid": float("inf"),
            "derived_plane_leakage": float("inf"),
            "Sder_sign_definite": 0.0,
            "Sder_anisotropy": float("nan"),
        }

    if np.dot(axis, c) < 0:
        axis = -axis
    axis_unit = axis / (np.linalg.norm(axis) + EPS)
    axis_align_const = float(np.dot(axis_unit, c))

    Bder = plane_basis_from_axis(axis_unit)
    Ader = Bder.T @ A @ Bder
    alpha_der = math.sqrt(max(0.0, float(-np.trace(Ader @ Ader) / 2.0)))
    if alpha_der > EPS:
        Jder = Ader / alpha_der
        derived_J2_resid = float(np.linalg.norm(Jder @ Jder + np.eye(2)))
    else:
        derived_J2_resid = float("inf")
    derived_plane_leakage = float(np.linalg.norm(axis_unit.T @ A @ Bder))

    Sder = Bder.T @ S @ Bder
    s_eigs = np.linalg.eigvalsh(Sder)
    sign_def = float(np.all(s_eigs > 1e-10) or np.all(s_eigs < -1e-10))
    mean_abs = float(np.mean(np.abs(s_eigs)))
    anisotropy = float((np.max(s_eigs) - np.min(s_eigs)) / (mean_abs + EPS))

    return {
        "complex_pair": complex_pair,
        "std_leakage": std_leakage,
        "std_J2_resid": std_J2_resid,
        "axis_norm": axis_norm,
        "axis_x": float(axis_unit[0]),
        "axis_y": float(axis_unit[1]),
        "axis_z": float(axis_unit[2]),
        "axis_align_const": axis_align_const,
        "derived_J2_resid": derived_J2_resid,
        "derived_plane_leakage": derived_plane_leakage,
        "Sder_sign_definite": sign_def,
        "Sder_anisotropy": anisotropy,
    }


def mean(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(model, global_level: int) -> tuple[dict, List[dict]]:
    rows = []
    by_parent_level: Dict[int, List[dict]] = {}
    for n in model.nodes.values():
        if len(n.children) != 3:
            continue
        M = model.local_matrix_for_parent(n.id)
        if M is None:
            continue
        d = local_diag(M)
        d["parent_level"] = n.level
        rows.append(d)
        by_parent_level.setdefault(n.level, []).append(d)

    axes = np.array([[r["axis_x"], r["axis_y"], r["axis_z"]] for r in rows if np.isfinite(r["axis_x"])])
    if len(axes):
        m = axes.mean(axis=0)
        mn = float(np.linalg.norm(m))
        if mn > EPS:
            mu = m / mn
            coh = float(np.mean(axes @ mu))
        else:
            coh = 0.0
    else:
        mn = float("nan")
        coh = float("nan")

    level_row = {
        "global_level": global_level,
        "nodes": len(model.nodes),
        "completed_triples": len(rows),
        "frac_complex_pair": mean(rows, "complex_pair"),
        "mean_std_leakage": mean(rows, "std_leakage"),
        "mean_std_J2_resid": mean(rows, "std_J2_resid"),
        "mean_axis_align_const": mean(rows, "axis_align_const"),
        "mean_axis_norm": mean(rows, "axis_norm"),
        "axis_mean_norm": mn,
        "axis_coherence_to_mean": coh,
        "mean_derived_J2_resid": mean(rows, "derived_J2_resid"),
        "mean_derived_plane_leakage": mean(rows, "derived_plane_leakage"),
        "frac_Sder_sign_definite": mean(rows, "Sder_sign_definite"),
        "mean_Sder_anisotropy": mean(rows, "Sder_anisotropy"),
    }

    parent_rows = []
    for pl, ds in sorted(by_parent_level.items()):
        parent_rows.append({
            "global_level": global_level,
            "parent_level": pl,
            "count": len(ds),
            "frac_complex_pair": mean(ds, "complex_pair"),
            "mean_std_leakage": mean(ds, "std_leakage"),
            "mean_axis_align_const": mean(ds, "axis_align_const"),
            "mean_derived_J2_resid": mean(ds, "derived_J2_resid"),
            "frac_Sder_sign_definite": mean(ds, "Sder_sign_definite"),
            "mean_Sder_anisotropy": mean(ds, "Sder_anisotropy"),
        })
    return level_row, parent_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    ScalingModel = load_scaling_model()
    model = ScalingModel(kernel="shell_norm_inverse_square", max_level=max_level, mode="log")

    frontier = [model.root]
    level_rows = []
    parent_rows = []
    for level in range(1, max_level + 1):
        frontier = model.grow_one_level(frontier, level)
        lr, pr = summarize(model, level)
        level_rows.append(lr)
        parent_rows.extend(pr)

    write_csv(outdir / "sector_closure_level_summaries.csv", level_rows)
    write_csv(outdir / "sector_closure_parent_level_summaries.csv", parent_rows)

    final = level_rows[-1]
    lines = [
        "SHELL-NORMALIZED EFFECTIVE OPERATOR SECTOR CLOSURE",
        f"  final level={max_level}, nodes={final['nodes']}, completed triples={final['completed_triples']}",
        f"  frac complex pair={final['frac_complex_pair']:.3f}",
        f"  mean standard-sector leakage={final['mean_std_leakage']:.6e}",
        f"  mean standard-sector J2 residual={final['mean_std_J2_resid']:.6e}",
        f"  mean axis alignment with constant={final['mean_axis_align_const']:.6f}",
        f"  axis mean norm={final['axis_mean_norm']:.6f}",
        f"  axis coherence to mean={final['axis_coherence_to_mean']:.6f}",
        f"  mean derived-plane J2 residual={final['mean_derived_J2_resid']:.6e}",
        f"  mean derived-plane leakage={final['mean_derived_plane_leakage']:.6e}",
        f"  frac S-derived-plane sign-definite={final['frac_Sder_sign_definite']:.3f}",
        f"  mean S-derived-plane anisotropy={final['mean_Sder_anisotropy']:.6f}",
        "",
        "LEVEL TREND",
    ]
    for r in level_rows:
        lines.append(
            f"  L={r['global_level']}: nodes={r['nodes']}, triples={r['completed_triples']}, "
            f"std_leak={r['mean_std_leakage']:.6e}, axis_align={r['mean_axis_align_const']:.6f}, "
            f"axis_coh={r['axis_coherence_to_mean']:.6f}, S_aniso={r['mean_Sder_anisotropy']:.6f}"
        )
    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=9)
    ap.add_argument("--outdir", type=Path, default=Path("effective_operator_sector_closure_out"))
    args = ap.parse_args()
    print(run(args.max_level, args.outdir))


if __name__ == "__main__":
    main()
