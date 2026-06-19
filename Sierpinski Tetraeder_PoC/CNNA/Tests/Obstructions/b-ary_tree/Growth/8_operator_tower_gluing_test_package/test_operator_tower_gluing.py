#!/usr/bin/env python3
"""
Operator tower gluing test for shell-controlled CNNA/NGF growth.

Goal
----
We already found stable local J-like planes on completed sibling triples.
This test checks whether those local axes/planes glue coherently along the
Parent -> Child tower.

Important limitation
--------------------
The provenance tree has no closed loops by itself. Therefore this test checks
vertical/tower compatibility and local sibling-subtower coherence, not genuine
global holonomy. True holonomy/frustration requires additional closure/gluing
edges in the effective geometry.

Default kernel
--------------
K(d) = 1 / (3^(d-1) d^2)
via test_conductance_scaling_generalization.ScalingModel.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12


def load_scaling_model():
    path = Path("/mnt/data/test_conductance_scaling_generalization.py")
    spec = importlib.util.spec_from_file_location("scaling", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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


def skew_axis(A: np.ndarray) -> np.ndarray:
    # A = cross(axis, .)
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def cross_matrix(a: np.ndarray) -> np.ndarray:
    x, y, z = a
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def local_axis_and_J(model, parent: int) -> Optional[dict]:
    M = model.local_matrix_for_parent(parent)
    if M is None:
        return None
    P = column_stochastic(M)
    S = 0.5 * (P + P.T)
    A = 0.5 * (P - P.T)
    ev = np.linalg.eigvals(P)
    c = np.ones(3) / math.sqrt(3.0)
    axis = skew_axis(A)
    norm = float(np.linalg.norm(axis))
    if norm <= EPS:
        return None
    # Orient axes consistently by positive dot with the constant/birth-order normal.
    if np.dot(axis, c) < 0:
        axis = -axis
    a = axis / (np.linalg.norm(axis) + EPS)
    J3 = cross_matrix(a)  # J3^2 = a a^T - I, so J^2=-I on a^⊥.
    S_eigs = np.linalg.eigvalsh(S)
    complex_pair = float(np.max(np.abs(np.imag(ev))) > 1e-9)
    return {
        "axis": a,
        "J3": J3,
        "complex_pair": complex_pair,
        "axis_align_const": float(np.dot(a, c)),
        "S_min": float(np.min(S_eigs)),
        "S_max": float(np.max(S_eigs)),
    }


def angle_deg_from_dot(dot: float) -> float:
    d = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(d))


def pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    dot = float(np.dot(a, b))
    dot_abs = abs(dot)
    angle = angle_deg_from_dot(dot_abs)
    # Plane distance between orthogonal planes is sin(angle between normals).
    plane_dist = math.sqrt(max(0.0, 1.0 - dot_abs * dot_abs))
    J_mis = float(np.linalg.norm(cross_matrix(a) - cross_matrix(b), ord="fro"))
    return {
        "axis_dot": dot,
        "axis_abs_dot": dot_abs,
        "axis_angle_deg": angle,
        "plane_distance": plane_dist,
        "J_frobenius_mismatch": J_mis,
    }


def mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def maxv(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.max(vals)) if vals else float("nan")


def summarize_pairs(rows: List[dict], group_key: str) -> List[dict]:
    groups: Dict[object, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        out.append({
            group_key: k,
            "count": len(rs),
            "mean_axis_abs_dot": mean([r["axis_abs_dot"] for r in rs]),
            "min_axis_abs_dot": float(np.min([r["axis_abs_dot"] for r in rs])) if rs else float("nan"),
            "mean_angle_deg": mean([r["axis_angle_deg"] for r in rs]),
            "max_angle_deg": maxv([r["axis_angle_deg"] for r in rs]),
            "mean_plane_distance": mean([r["plane_distance"] for r in rs]),
            "mean_J_mismatch": mean([r["J_frobenius_mismatch"] for r in rs]),
        })
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(max_level: int, outdir: Path) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    ScalingModel = load_scaling_model()
    model = ScalingModel(kernel="shell_norm_inverse_square", max_level=max_level, mode="log")
    frontier = [model.root]
    for level in range(1, max_level + 1):
        frontier = model.grow_one_level(frontier, level)

    # Local axis/J data for all completed parents.
    local: Dict[int, dict] = {}
    for n in model.nodes.values():
        if len(n.children) == 3:
            d = local_axis_and_J(model, n.id)
            if d is not None:
                local[n.id] = d

    # Vertical parent -> child gluing pairs.
    vertical_rows = []
    for p, dp in local.items():
        for c in model.nodes[p].children:
            if c in local:
                m = pair_metrics(dp["axis"], local[c]["axis"])
                vertical_rows.append({
                    "parent": p,
                    "child": c,
                    "parent_level": model.nodes[p].level,
                    "child_level": model.nodes[c].level,
                    **m,
                })

    # Sibling-subtower coherence: compare axes of completed child-subtriples inside one parent.
    sibling_rows = []
    for p, dp in local.items():
        child_completed = [c for c in model.nodes[p].children if c in local]
        if len(child_completed) == 3:
            axes = [local[c]["axis"] for c in child_completed]
            pairdots = []
            angles = []
            for i in range(3):
                for j in range(i + 1, 3):
                    m = pair_metrics(axes[i], axes[j])
                    pairdots.append(m["axis_abs_dot"])
                    angles.append(m["axis_angle_deg"])
            mean_axis = np.mean(np.array(axes), axis=0)
            mean_axis_norm = float(np.linalg.norm(mean_axis))
            mean_axis_unit = mean_axis / (mean_axis_norm + EPS)
            parent_child_mean = pair_metrics(dp["axis"], mean_axis_unit)
            sibling_rows.append({
                "parent": p,
                "parent_level": model.nodes[p].level,
                "mean_child_axis_abs_dot": mean(pairdots),
                "min_child_axis_abs_dot": float(np.min(pairdots)),
                "mean_child_angle_deg": mean(angles),
                "max_child_angle_deg": maxv(angles),
                "child_axis_mean_norm": mean_axis_norm,
                "parent_to_child_mean_angle_deg": parent_child_mean["axis_angle_deg"],
                "parent_to_child_mean_J_mismatch": parent_child_mean["J_frobenius_mismatch"],
            })

    vertical_by_parent_level = summarize_pairs(vertical_rows, "parent_level")
    vertical_by_child_level = summarize_pairs(vertical_rows, "child_level")

    local_rows = []
    for p, d in local.items():
        a = d["axis"]
        local_rows.append({
            "parent": p,
            "parent_level": model.nodes[p].level,
            "axis_x": float(a[0]),
            "axis_y": float(a[1]),
            "axis_z": float(a[2]),
            "axis_align_const": d["axis_align_const"],
            "complex_pair": d["complex_pair"],
            "S_min": d["S_min"],
            "S_max": d["S_max"],
        })

    write_csv(outdir / "local_axes.csv", local_rows)
    write_csv(outdir / "vertical_gluing_pairs.csv", vertical_rows)
    write_csv(outdir / "vertical_by_parent_level.csv", vertical_by_parent_level)
    write_csv(outdir / "vertical_by_child_level.csv", vertical_by_child_level)
    write_csv(outdir / "sibling_subtower_gluing.csv", sibling_rows)

    all_axes = np.array([[r["axis_x"], r["axis_y"], r["axis_z"]] for r in local_rows])
    mean_axis = np.mean(all_axes, axis=0)
    mean_axis_norm = float(np.linalg.norm(mean_axis))
    mean_axis_unit = mean_axis / (mean_axis_norm + EPS)
    axis_coherence = float(np.mean(all_axes @ mean_axis_unit))

    lines = []
    lines.append("OPERATOR TOWER GLUING TEST")
    lines.append(f"  final level={max_level}, nodes={len(model.nodes)}, completed local axes={len(local_rows)}")
    lines.append(f"  vertical parent-child gluing pairs={len(vertical_rows)}")
    lines.append(f"  sibling-subtower gluing triples={len(sibling_rows)}")
    lines.append(f"  global mean axis=({mean_axis_unit[0]:.9f}, {mean_axis_unit[1]:.9f}, {mean_axis_unit[2]:.9f})")
    lines.append(f"  global axis mean norm={mean_axis_norm:.9f}")
    lines.append(f"  global axis coherence={axis_coherence:.9f}")
    if vertical_rows:
        lines.append("")
        lines.append("VERTICAL GLUING ALL")
        lines.append(f"  mean |dot|={mean([r['axis_abs_dot'] for r in vertical_rows]):.9f}")
        lines.append(f"  min |dot|={float(np.min([r['axis_abs_dot'] for r in vertical_rows])):.9f}")
        lines.append(f"  mean angle deg={mean([r['axis_angle_deg'] for r in vertical_rows]):.9f}")
        lines.append(f"  max angle deg={maxv([r['axis_angle_deg'] for r in vertical_rows]):.9f}")
        lines.append(f"  mean plane distance={mean([r['plane_distance'] for r in vertical_rows]):.9e}")
        lines.append(f"  mean J mismatch={mean([r['J_frobenius_mismatch'] for r in vertical_rows]):.9e}")
    if sibling_rows:
        lines.append("")
        lines.append("SIBLING-SUBTOWER GLUING")
        lines.append(f"  mean child-axis |dot|={mean([r['mean_child_axis_abs_dot'] for r in sibling_rows]):.9f}")
        lines.append(f"  min child-axis |dot|={float(np.min([r['min_child_axis_abs_dot'] for r in sibling_rows])):.9f}")
        lines.append(f"  mean child angle deg={mean([r['mean_child_angle_deg'] for r in sibling_rows]):.9f}")
        lines.append(f"  max child angle deg={maxv([r['max_child_angle_deg'] for r in sibling_rows]):.9f}")
        lines.append(f"  mean parent-to-child-mean angle deg={mean([r['parent_to_child_mean_angle_deg'] for r in sibling_rows]):.9f}")

    lines.append("")
    lines.append("VERTICAL BY PARENT LEVEL")
    for r in vertical_by_parent_level:
        lines.append(
            f"  parent_level={r['parent_level']}: count={r['count']}, "
            f"mean |dot|={r['mean_axis_abs_dot']:.9f}, max angle={r['max_angle_deg']:.9f}, "
            f"mean J mismatch={r['mean_J_mismatch']:.9e}"
        )

    summary = "\n".join(lines)
    (outdir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=9)
    ap.add_argument("--outdir", type=Path, default=Path("operator_tower_gluing_out"))
    args = ap.parse_args()
    print(run(args.max_level, args.outdir))


if __name__ == "__main__":
    main()
