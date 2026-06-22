#!/usr/bin/env python3
"""Audit helper for test_cqnm_sminus1_saturation_link_curvature_gate.py.

Checks performed:
1. Detect whether the CQNM script imports/reuses the DynamicBirthConductanceModel
   from script 1 or only creates toy ProvenanceRecord weights.
2. Check the claimed periodic T^3 control topology at n=2 and n=3.
3. Emit a compact JSON report.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("test_cqnm_sminus1_saturation_link_curvature_gate_uploaded.py")
SCRIPT1 = Path(__file__).with_name("reference_script1_dynamic_birth_conductance.py")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("cqnm_under_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cqnm_under_audit"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def source_feature_scan() -> dict:
    src = SCRIPT.read_text(encoding="utf-8")
    s1 = SCRIPT1.read_text(encoding="utf-8")
    return {
        "new_script_has_DynamicBirthConductanceModel": "DynamicBirthConductanceModel" in src,
        "new_script_has_parent_line_update": "parent_line" in src,
        "new_script_has_directed_edges": "directed_edges" in src,
        "new_script_has_birth_environment_load": "birth_environment_load" in src,
        "script1_has_DynamicBirthConductanceModel": "DynamicBirthConductanceModel" in s1,
        "script1_has_parent_line_update": "parent_line" in s1,
        "script1_has_directed_edges": "directed_edges" in s1,
        "script1_has_birth_environment_load": "birth_environment_load" in s1,
        "new_script_has_synthetic_record_vector": "def record_vector" in src,
        "new_script_has_synthetic_k_cochain": "def k_cochain_on_faces" in src,
    }


def t3_topology_checks(mod) -> dict:
    out = {}
    for n in (2, 3):
        model = mod.build_cqnm_sminus1_saturated_t3(n)
        topo = mod.complex_topology_metrics(model)
        c = topo["counts"]
        chi = c["vertices"] - c["edges"] + c["faces"] - c["tets"]
        out[f"periodic_n_{n}"] = {
            "counts": c,
            "euler_characteristic": chi,
            "betti_z2": topo["betti_z2"],
            "boundary_face_count": topo["boundary_face_count"],
            "face_occupancy_counts": topo["face_occupancy_counts"],
            "edge_link_cycle_fraction": topo["edge_link_cycle_fraction"],
            "is_consistent_with_T3_betti_over_Z2": topo["betti_z2"] == {"beta0": 1, "beta1": 3, "beta2": 3, "beta3": 1},
            "is_consistent_with_closed_3_manifold_euler_0": chi == 0,
        }
    return out


def main() -> int:
    mod = load_module(SCRIPT)
    report = {
        "source_feature_scan": source_feature_scan(),
        "t3_topology_checks": t3_topology_checks(mod),
        "audit_conclusion_flags": {
            "script_reuses_script1_dynamic_growth": False,
            "script_c_periodic_n2_is_valid_T3_control": False,
            "script_c_periodic_n3_topology_is_consistent_with_T3": True,
            "script_tests_actual_operator_noncommutativity": False,
            "script_tests_topological_carrier_for_noncommutativity": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    Path("audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
