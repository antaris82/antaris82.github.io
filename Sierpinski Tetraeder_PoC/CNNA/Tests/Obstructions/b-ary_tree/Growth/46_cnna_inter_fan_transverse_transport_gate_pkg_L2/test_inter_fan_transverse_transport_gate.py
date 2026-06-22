#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import cnna_non_shelling_core as core

EPS = 1e-12


class InterFanTransportGrowth(core.DynamicProvenanceGrowth):
    """Dynamic provenance growth with an explicit inter-parent-fan transverse transport gate.

    This is a diagnostic hypothesis, not a derived CNNA law.  The base Script-1/2
    engine already creates sequential sibling asymmetry inside a parent fan.  This
    subclass tests the suspected missing layer: transport of that transverse
    orientation through the common parent/grandparent context into neighbouring
    parent fans.
    """

    def __init__(
        self,
        *args,
        interfan_strength: float = 0.0,
        interfan_decay: float = 0.55,
        interfan_update_g: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.interfan_strength = float(interfan_strength)
        self.interfan_decay = float(interfan_decay)
        self.interfan_update_g = bool(interfan_update_g)
        self.interfan_edges: Dict[Tuple[int, int], float] = defaultdict(float)
        self.interfan_records: List[dict] = []

    def _fan_sibling_parents(self, parent: int) -> List[int]:
        gp = self.nodes[parent].parent
        if gp is None:
            return []
        return [q for q in self.nodes[gp].children if q != parent]

    def _transverse_z3_pairing(self, child: int, other: int) -> Tuple[float, float, float]:
        c = self.nodes[child]
        o = self.nodes[other]
        phase = 2.0 * math.pi * (c.birth_order - o.birth_order) / 3.0
        z3_sin = math.sin(phase)
        z3_cos = math.cos(phase)
        handed = float(np.dot(c.e1, o.e2) - np.dot(c.e2, o.e1))
        radial = max(0.0, float(np.dot(c.radial, o.radial)))
        dist = float(np.linalg.norm(c.pos - o.pos))
        return z3_sin, handed, radial / (1.0 + 0.35 * dist)

    def _apply_interfan_transport(self, parent: int, child: int) -> None:
        if self.interfan_strength <= 0.0:
            return
        neighbour_parent_fans = self._fan_sibling_parents(parent)
        if not neighbour_parent_fans:
            return
        total = 0.0
        count = 0
        for q in neighbour_parent_fans:
            # transport into already existing children of neighbouring parent fans;
            # if the neighbouring fan has not been born yet there is nothing to transmit to.
            for other in self.nodes[q].children:
                z3_sin, handed, geom = self._transverse_z3_pairing(child, other)
                orientation = abs(z3_sin) * (0.5 + 0.5 * abs(handed))
                if orientation < 1e-9:
                    continue
                level_gap = abs(self.nodes[child].level - self.nodes[other].level)
                decay = self.interfan_decay ** level_gap
                base = self.interfan_strength * decay * orientation * geom
                base *= math.sqrt(max(EPS, self.nodes[child].birth_g * self.nodes[other].birth_g))
                # asymmetric transport: the sign of the Z3 phase decides which direction is stronger.
                asym = max(-0.85, min(0.85, 0.55 * z3_sin + 0.25 * handed))
                w_forward = base * (1.0 + asym)
                w_backward = base * (1.0 - asym)
                self.directed_edges[(other, child)] += w_forward
                self.directed_edges[(child, other)] += w_backward
                self.interfan_edges[(other, child)] += w_forward
                self.interfan_edges[(child, other)] += w_backward
                if self.interfan_update_g:
                    self.nodes[child].g += 0.015 * w_forward
                    self.nodes[other].g += 0.010 * w_backward
                total += w_forward + w_backward
                count += 1
        if count:
            self.interfan_records.append({
                "t": self.t,
                "parent": parent,
                "child": child,
                "edge_pairs": count,
                "weight_total": total,
                "weight_mean": total / count,
            })

    def add_child(self, parent: int, order: int) -> int:
        child = super().add_child(parent, order)
        self._apply_interfan_transport(parent, child)
        return child


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def move_gain(row: dict) -> float:
    return (
        4.0 * max(0.0, float(row.get("delta_beta2", 0)))
        + 3.0 * max(0.0, float(row.get("delta_beta1", 0)))
        + 0.02 * max(0.0, -float(row.get("delta_boundary_faces", 0)))
        + 0.001 * float(row.get("response_score", 0.0))
    )


def select_candidate(rows: List[dict], cls: str, selector: str) -> Optional[dict]:
    sub = [r for r in rows if r.get("move_class") == cls and r.get("status") == "ok"]
    if not sub:
        return None
    if selector == "response":
        return sorted(sub, key=lambda r: int(r.get("response_rank_legal") or 10**9))[0]
    if selector == "topology":
        return sorted(sub, key=move_gain, reverse=True)[0]
    raise ValueError(selector)


def apply_and_measure(model: InterFanTransportGrowth, K: core.SimplicialComplex, row: Optional[dict], source: str) -> dict:
    if row is None:
        return {"status": "missing_candidate"}
    L, reason, encoded = core.apply_candidate_row(K, row)
    base = core.full_metrics(model, K, source)
    if L is None:
        return {
            "status": reason,
            "candidate_id": row.get("candidate_id"),
            "move_class": row.get("move_class"),
            "selector_response_rank": row.get("response_rank_legal"),
            "encoded_move": encoded,
        }
    after = core.full_metrics(model, L, source)
    return {
        "status": reason,
        "candidate_id": row.get("candidate_id"),
        "move_class": row.get("move_class"),
        "selector_response_rank": row.get("response_rank_legal"),
        "encoded_move": encoded,
        "delta_beta1": after["beta1"] - base["beta1"],
        "delta_beta2": after["beta2"] - base["beta2"],
        "delta_beta3": after["beta3"] - base["beta3"],
        "delta_boundary_faces": after["boundary_faces"] - base["boundary_faces"],
        "after_beta0": after["beta0"],
        "after_beta1": after["beta1"],
        "after_beta2": after["beta2"],
        "after_beta3": after["beta3"],
        "after_boundary_fraction": after["boundary_fraction"],
        "after_edge_link_cycle_fraction": after["edge_link_cycle_fraction"],
        "after_K_mean": after["K_mean"],
        "after_harmonic_ratio": after["harmonic_ratio"],
        "after_exact_residual_ratio": after["exact_residual_ratio"],
    }


def compact_best(rows: List[dict], cls: str) -> dict:
    sub = [r for r in rows if r.get("move_class") == cls and r.get("status") == "ok"]
    if not sub:
        return {}
    by_response = select_candidate(rows, cls, "response")
    by_topology = select_candidate(rows, cls, "topology")
    def c(r: Optional[dict]) -> dict:
        if r is None:
            return {}
        keep = ["candidate_id", "response_score", "response_rank_legal", "delta_beta1", "delta_beta2", "delta_boundary_faces", "new_beta1", "new_beta2", "K_pair_norm", "directed_coupling", "directed_imbalance", "transverse_complementarity", "address_similarity"]
        return {k: r.get(k, "") for k in keep}
    return {"top_response": c(by_response), "top_topology": c(by_topology)}


def run_variant(args: argparse.Namespace, variant: dict, out: Path) -> dict:
    name = variant["name"]
    model = InterFanTransportGrowth(
        mode=args.mode,
        growth_rule=variant["growth_rule"],
        transverse_amp=args.transverse_amp,
        interfan_strength=variant["interfan_strength"],
        interfan_decay=args.interfan_decay,
        interfan_update_g=args.interfan_update_g,
    )
    model.grow(args.max_level)
    K = core.build_dynamic_outward_ngf_complex(model)
    rows, audit = core.enumerate_moves(
        model,
        K,
        source=args.source,
        max_boundary_faces=args.max_boundary_faces,
        max_single_vertices=args.max_single_vertices,
        max_pair_candidates=args.max_pair_candidates,
        max_rows=args.max_rows,
    )
    variant_out = out / name
    variant_out.mkdir(parents=True, exist_ok=True)
    write_csv(variant_out / "move_candidates.csv", rows)
    write_csv(variant_out / "interfan_transport_records.csv", model.interfan_records)
    base_metrics = core.full_metrics(model, K, args.source)
    applied_rows: List[dict] = []
    for cls in ["shelling_disk_move", "cap_move", "handle_candidate", "quotient_candidate"]:
        for selector in ["response", "topology"]:
            row = select_candidate(rows, cls, selector)
            meas = apply_and_measure(model, K, row, args.source)
            applied_rows.append({"variant": name, "target_class": cls, "selector": selector, **meas})
    write_csv(variant_out / "applied_move_results.csv", applied_rows)
    payload = {
        "name": name,
        "growth_rule": variant["growth_rule"],
        "interfan_strength": variant["interfan_strength"],
        "interfan_edge_count": len(model.interfan_edges),
        "interfan_weight_total": float(sum(model.interfan_edges.values())),
        "interfan_record_count": len(model.interfan_records),
        "base_metrics": base_metrics,
        "audit_summary": audit,
        "best_handle": compact_best(rows, "handle_candidate"),
        "best_quotient": compact_best(rows, "quotient_candidate"),
        "applied_moves": applied_rows,
    }
    (variant_out / "variant_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def make_markdown(summary: dict) -> Tuple[str, str, str, str]:
    variants = summary["variants"]
    lines = [
        "# SUMMARY",
        "",
        "This package tests whether the missing layer is not local sibling asymmetry, but inter-parent-fan transverse transport.",
        "",
        "The base growth remains Script-1-like: a newborn senses its parent line and older siblings, then backreacts onto ancestors and older siblings. The new diagnostic branch adds an explicit transport of the local Z3/transverse orientation into neighbouring parent fans through the common grandparent context.",
        "",
        "This is a controlled hypothesis test, not a derived CNNA law.",
        "",
        "## Main numerical comparison",
        "",
        "| variant | growth rule | interfan strength | interfan edges | beta | base harmonic | top handle rank | top quotient rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        b = v["base_metrics"]
        audit = v["audit_summary"]
        beta = f"({b['beta0']},{b['beta1']},{b['beta2']},{b['beta3']})"
        lines.append(
            f"| {v['name']} | {v['growth_rule']} | {v['interfan_strength']:.3g} | {v['interfan_edge_count']} | {beta} | {b['harmonic_ratio']:.6g} | {audit.get('top_handle_rank')} | {audit.get('top_quotient_rank')} |"
        )
    lines += [
        "",
        "## Core reading",
        "",
        "The local sequential sibling offset is not the missing ingredient; it already exists in real_growth. The missing layer is whether that local transverse orientation is transported between neighbouring parent-fans strongly enough to rank topology-effective non-shelling moves above local shelling/cap moves.",
        "",
        "If interfan transport improves the response rank of beta-positive handle/quotient moves only in real_growth, it supports the inter-fan diagnosis. If symmetrized/no-backreaction controls improve similarly, the move remains externally introduced rather than CNNA-derived.",
    ]
    results = ["# RESULTS", ""]
    for v in variants:
        b = v["base_metrics"]
        results += [
            f"## {v['name']}",
            "",
            f"- growth_rule: `{v['growth_rule']}`",
            f"- interfan_strength: `{v['interfan_strength']}`",
            f"- interfan_edge_count: `{v['interfan_edge_count']}`",
            f"- interfan_weight_total: `{v['interfan_weight_total']:.8g}`",
            f"- base beta: `({b['beta0']},{b['beta1']},{b['beta2']},{b['beta3']})`",
            f"- base boundary_fraction: `{b['boundary_fraction']:.6g}`",
            f"- base K_mean: `{b['K_mean']:.6g}`",
            f"- base harmonic_ratio: `{b['harmonic_ratio']:.6g}`",
            "",
            "Best handle candidates:",
            "",
            "```json",
            json.dumps(v["best_handle"], indent=2, sort_keys=True),
            "```",
            "",
            "Best quotient candidates:",
            "",
            "```json",
            json.dumps(v["best_quotient"], indent=2, sort_keys=True),
            "```",
            "",
            "Applied moves:",
            "",
            "```json",
            json.dumps(v["applied_moves"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    interp = [
        "# SOURCE_AUDIT_1_40",
        "",
        "Carried forward from the earlier Growth-series audit:",
        "",
        "- Script 1/2: sequential birth/backreaction is the real growth mode; symmetrized birth is a control.",
        "- Script 12: shell-normalized inverse-square is a response-kernel/locality result, not evidence that shelling topology is desirable.",
        "- Script 35: the operator sector must use K_abc=[A_ab,A_bc], not a synthetic face cochain.",
        "- Script 40: immediate local tetrahedral closure is an obstruction because it closes the parent fan too locally.",
        "",
        "This package tests the newly isolated seam: the local fan-transverse Z3/asymmetry exists, but it is not automatically transported across parent-fans. The injected inter-fan rule is therefore marked as a diagnostic hypothesis, not as a derived CNNA rule.",
    ]
    readme = [
        "# Inter-fan transverse transport gate",
        "",
        "Run:",
        "",
        "```bash",
        "python test_inter_fan_transverse_transport_gate.py --max-level 2 --make-zip",
        "```",
        "",
        "Outputs include per-variant move candidates, applied move results, JSON summaries, RESULTS.md and SUMMARY.md.",
    ]
    return "\n".join(lines), "\n".join(results), "\n".join(interp), "\n".join(readme)


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [
        {"name": "real_local_only", "growth_rule": "real_growth", "interfan_strength": 0.0},
        {"name": "real_interfan_transport", "growth_rule": "real_growth", "interfan_strength": args.interfan_strength},
        {"name": "sym_interfan_transport", "growth_rule": "symmetrized_birth", "interfan_strength": args.interfan_strength},
        {"name": "no_backreaction_interfan_transport", "growth_rule": "no_backreaction", "interfan_strength": args.interfan_strength},
    ]
    payloads = [run_variant(args, v, out) for v in variants]
    rows = []
    for p in payloads:
        b = p["base_metrics"]
        a = p["audit_summary"]
        rows.append({
            "variant": p["name"],
            "growth_rule": p["growth_rule"],
            "interfan_strength": p["interfan_strength"],
            "interfan_edge_count": p["interfan_edge_count"],
            "interfan_weight_total": p["interfan_weight_total"],
            "beta0": b["beta0"], "beta1": b["beta1"], "beta2": b["beta2"], "beta3": b["beta3"],
            "boundary_fraction": b["boundary_fraction"],
            "edge_link_cycle_fraction": b["edge_link_cycle_fraction"],
            "K_mean": b["K_mean"],
            "harmonic_ratio": b["harmonic_ratio"],
            "top_handle_rank": a.get("top_handle_rank"),
            "top_quotient_rank": a.get("top_quotient_rank"),
            "topologically_effective_non_shelling_count": a.get("topologically_effective_non_shelling_count"),
        })
    write_csv(out / "comparative_interfan_summary.csv", rows)
    summary = {
        "args": vars(args),
        "variants": payloads,
        "comparative_rows": rows,
        "interpretation_flags": {
            "base_real_still_ball_like": payloads[0]["base_metrics"]["beta1"] == 0 and payloads[0]["base_metrics"]["beta2"] == 0,
            "interfan_is_only_diagnostic_hypothesis": True,
            "check_real_vs_controls_before_claiming_derivation": True,
        },
    }
    (out / "comparative_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md, results_md, audit_md, readme = make_markdown(summary)
    (out / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    (out / "RESULTS.md").write_text(results_md, encoding="utf-8")
    (out / "SOURCE_AUDIT_1_40.md").write_text(audit_md, encoding="utf-8")
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def package(out: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__), arcname=Path(__file__).name)
        z.write(Path(__file__).with_name("cnna_non_shelling_core.py"), arcname="cnna_non_shelling_core.py")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-level", type=int, default=2)
    p.add_argument("--mode", choices=["linear", "log", "saturating"], default="linear")
    p.add_argument("--source", choices=["record", "live", "handoff", "aging"], default="live")
    p.add_argument("--transverse-amp", type=float, default=0.42)
    p.add_argument("--interfan-strength", type=float, default=0.16)
    p.add_argument("--interfan-decay", type=float, default=0.55)
    p.add_argument("--interfan-update-g", action="store_true")
    p.add_argument("--max-boundary-faces", type=int, default=90)
    p.add_argument("--max-single-vertices", type=int, default=12)
    p.add_argument("--max-pair-candidates", type=int, default=2500)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--out", type=str, default="inter_fan_transverse_transport_out_L3")
    p.add_argument("--make-zip", action="store_true")
    args = p.parse_args()
    summary = run(args)
    print(json.dumps({"out": args.out, "comparative_rows": summary["comparative_rows"]}, indent=2))
    if args.make_zip:
        package(Path(args.out), Path(f"cnna_inter_fan_transverse_transport_gate_pkg_L{args.max_level}.zip"))


if __name__ == "__main__":
    main()
