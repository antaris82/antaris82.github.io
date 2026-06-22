#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import cnna_non_shelling_core as core
import test_nonlinear_asymmetry_cascade_growth as nl
import test_harmonic_k_orientation_kappa_gate as hk
import test_pairing_transport_antisym_birth_coherence_gate as prev

EPS = 1e-12
Face = Tuple[int, int, int]
Edge = Tuple[int, int]


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rotate_about_axis(v: np.ndarray, axis: np.ndarray, step: int, phase_sign: int = 1) -> np.ndarray:
    step = int(step) % 3
    if step == 0:
        return np.array(v, dtype=float)
    a = np.array(axis, dtype=float)
    na = float(np.linalg.norm(a))
    if na < EPS:
        return np.array(v, dtype=float)
    a = a / na
    theta = float(phase_sign) * 2.0 * math.pi * float(step) / 3.0
    c = math.cos(theta)
    s = math.sin(theta)
    v = np.array(v, dtype=float)
    return c * v + s * np.cross(a, v) + (1.0 - c) * float(np.dot(a, v)) * a


def effective_birth_order(model: core.DynamicProvenanceGrowth, node: int, erase_phase_for_strict_sym: bool) -> int:
    n = model.nodes[node]
    if n.parent is None or n.birth_order == 0:
        return 0
    if erase_phase_for_strict_sym and getattr(model, 'growth_rule', '') == 'symmetrized_birth':
        return 2
    return int(n.birth_order)


def face_birth_label(model: core.DynamicProvenanceGrowth, f: Face, erase_phase_for_strict_sym: bool) -> int:
    s = 0
    for v in f:
        bo = effective_birth_order(model, v, erase_phase_for_strict_sym)
        if bo > 0:
            s += bo - 1
    return int(s % 3)


def best_z3_step(target: np.ndarray, source_in_target_chart: np.ndarray, axis: np.ndarray, signed: bool, phase_sign: int) -> Tuple[int, float, float]:
    nt = float(np.linalg.norm(target))
    ns = float(np.linalg.norm(source_in_target_chart))
    if nt < EPS or ns < EPS:
        return 0, 0.0, 1.0
    best_step = 0
    best_cos = -1e99
    best_key = -1e99
    for step in (0, 1, 2):
        cand = rotate_about_axis(source_in_target_chart, axis, step, phase_sign)
        nc = float(np.linalg.norm(cand))
        cos = float(np.dot(target, cand) / ((nt * nc) + EPS))
        key = cos if signed else abs(cos)
        if key > best_key + 1e-15:
            best_key = key
            best_cos = cos
            best_step = step
    residual = 1.0 - (best_cos if signed else abs(best_cos))
    return int(best_step), float(best_cos), float(residual)


def edge_map_from_faces(faces: List[Face]) -> Dict[Edge, List[int]]:
    out: Dict[Edge, List[int]] = {}
    for i, f in enumerate(faces):
        a, b, c = f
        for e in (tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))):
            out.setdefault(e, []).append(i)
    return out


def add_constraint(adj: Dict[int, List[Tuple[int, int, str, float]]], i: int, j: int, step_ij: int, kind: str, residual: float) -> None:
    # label[j] = label[i] + step_ij mod 3
    adj.setdefault(i, []).append((j, int(step_ij) % 3, kind, float(residual)))
    adj.setdefault(j, []).append((i, (-int(step_ij)) % 3, kind, float(residual)))


def local_face_vectors(model: core.DynamicProvenanceGrowth, faces: List[Face], args: argparse.Namespace) -> np.ndarray:
    if not faces:
        return np.zeros((0, 3), dtype=float)
    return np.array([
        prev.axial(prev.face_K_directed(model, f, args.source, args.phase_sign, args.antisym_eta, args.erase_phase_for_strict_sym))
        for f in faces
    ], dtype=float)


def build_constraints(
    model: core.DynamicProvenanceGrowth,
    K: core.SimplicialComplex,
    pairing_log: List[dict],
    W_raw: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> Tuple[Dict[int, List[Tuple[int, int, str, float]]], List[dict]]:
    faces = K.faces()
    idx = {tuple(f): i for i, f in enumerate(faces)}
    adj: Dict[int, List[Tuple[int, int, str, float]]] = {}
    rows: List[dict] = []
    if mode == 'none' or mode == 'birth_sum':
        return adj, rows
    signed = mode.endswith('_signed')
    use_pair = mode.startswith('pair_graph') or mode.startswith('face_graph')
    use_face = mode.startswith('face_graph')

    if use_face:
        edge_map = edge_map_from_faces(faces)
        for e, inds in sorted(edge_map.items()):
            active = sorted(set(inds))
            for a_pos in range(len(active)):
                for b_pos in range(a_pos + 1, len(active)):
                    i, j = active[a_pos], active[b_pos]
                    ni = hk.face_normal(model, faces[i], 'birth_order')
                    nj = hk.face_normal(model, faces[j], 'birth_order')
                    R = prev.rotation_from_to(nj, ni)
                    src_j_in_i = R @ W_raw[j]
                    step, cos, residual = best_z3_step(W_raw[i], src_j_in_i, ni, signed=signed, phase_sign=args.propagation_phase_sign)
                    add_constraint(adj, i, j, step, 'shared_edge_face_graph', residual)
                    rows.append({
                        'kind': 'shared_edge_face_graph',
                        'face_i': str(list(faces[i])),
                        'face_j': str(list(faces[j])),
                        'shared_edge': str(list(e)),
                        'best_step_j_minus_i': step,
                        'best_cos': cos,
                        'best_residual': residual,
                    })

    if use_pair:
        for k, log in enumerate(pairing_log):
            if not log.get('applied'):
                continue
            fa, fb = prev.parse_pair_faces(log)
            if fa is None or fb is None or fa not in idx or fb not in idx:
                continue
            ia, ib = idx[fa], idx[fb]
            na = hk.face_normal(model, fa, 'birth_order')
            nb = hk.face_normal(model, fb, 'birth_order')
            R_b_to_a = prev.rotation_from_to(nb, -na)
            kb_to_a = R_b_to_a @ W_raw[ib]
            step, cos, residual = best_z3_step(W_raw[ia], kb_to_a, -na, signed=signed, phase_sign=args.propagation_phase_sign)
            add_constraint(adj, ia, ib, step, 'actual_pair_graph', residual)
            rows.append({
                'kind': 'actual_pair_graph',
                'pair_index': k,
                'face_i': str(list(fa)),
                'face_j': str(list(fb)),
                'best_step_j_minus_i': step,
                'best_cos': cos,
                'best_residual': residual,
                'A_invariant': log.get('A_invariant', ''),
                'directed_imbalance': log.get('directed_imbalance', ''),
                'transverse_complementarity': log.get('transverse_complementarity', ''),
                'decision_used_delta_beta': log.get('decision_used_delta_beta', ''),
            })
    return adj, rows


def propagate_labels(
    model: core.DynamicProvenanceGrowth,
    K: core.SimplicialComplex,
    pairing_log: List[dict],
    W_raw: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> Tuple[Dict[int, int], dict, List[dict]]:
    faces = K.faces()
    labels: Dict[int, int] = {}
    rows: List[dict] = []
    if mode == 'none':
        return {i: 0 for i in range(len(faces))}, {
            'phase_mode': mode, 'phase_graph_edge_count': 0, 'phase_conflict_count': 0,
            'phase_conflict_fraction': 0.0, 'phase_constraint_mean_residual': 0.0,
        }, rows
    if mode == 'birth_sum':
        labels = {i: face_birth_label(model, f, args.erase_phase_for_strict_sym) for i, f in enumerate(faces)}
        return labels, {
            'phase_mode': mode, 'phase_graph_edge_count': 0, 'phase_conflict_count': 0,
            'phase_conflict_fraction': 0.0, 'phase_constraint_mean_residual': 0.0,
        }, rows

    adj, constraint_rows = build_constraints(model, K, pairing_log, W_raw, mode, args)
    for i in range(len(faces)):
        if i in labels:
            continue
        labels[i] = face_birth_label(model, faces[i], args.erase_phase_for_strict_sym)
        dq = deque([i])
        while dq:
            u = dq.popleft()
            for v, step, kind, residual in sorted(adj.get(u, []), key=lambda x: (x[0], x[1], x[2])):
                proposed = (labels[u] + step) % 3
                if v not in labels:
                    labels[v] = proposed
                    dq.append(v)
    conflicts = 0
    checked = 0
    conflict_rows: List[dict] = []
    residuals = []
    for u, edges in adj.items():
        for v, step, kind, residual in edges:
            if u > v:
                continue
            checked += 1
            residuals.append(float(residual))
            expected = (labels[u] + step) % 3
            ok = (labels.get(v, 0) % 3) == expected
            if not ok:
                conflicts += 1
                conflict_rows.append({
                    'kind': kind,
                    'face_u': str(list(faces[u])),
                    'face_v': str(list(faces[v])),
                    'label_u': labels.get(u, 0),
                    'label_v': labels.get(v, 0),
                    'expected_label_v': expected,
                    'step_v_minus_u': step,
                    'constraint_residual': residual,
                })
    rows = constraint_rows + conflict_rows
    stats = {
        'phase_mode': mode,
        'phase_graph_edge_count': checked,
        'phase_conflict_count': conflicts,
        'phase_conflict_fraction': float(conflicts / checked) if checked else 0.0,
        'phase_constraint_mean_residual': float(np.mean(residuals)) if residuals else 0.0,
        'phase_label_counts': {str(k): int(sum(1 for v in labels.values() if v == k)) for k in (0, 1, 2)},
    }
    return labels, stats, rows


def transported_pair_fields_propagated(
    model: core.DynamicProvenanceGrowth,
    K: core.SimplicialComplex,
    pairing_log: List[dict],
    labels: Dict[int, int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    faces = K.faces()
    idx = {tuple(f): i for i, f in enumerate(faces)}
    W_pair = np.zeros((len(faces), 3), dtype=float)
    scalar_pair = np.zeros(len(faces), dtype=float)
    rows: List[dict] = []
    for k, log in enumerate(pairing_log):
        if not log.get('applied'):
            continue
        fa, fb = prev.parse_pair_faces(log)
        if fa is None or fb is None or fa not in idx or fb not in idx:
            continue
        ia, ib = idx[fa], idx[fb]
        ka = prev.axial(prev.face_K_directed(model, fa, args.source, args.phase_sign, args.antisym_eta, args.erase_phase_for_strict_sym))
        kb = prev.axial(prev.face_K_directed(model, fb, args.source, args.phase_sign, args.antisym_eta, args.erase_phase_for_strict_sym))
        na = hk.face_normal(model, fa, 'birth_order')
        nb = hk.face_normal(model, fb, 'birth_order')
        R_b_to_a = prev.rotation_from_to(nb, -na)
        kb_to_a = R_b_to_a @ kb
        step = (int(labels.get(ib, 0)) - int(labels.get(ia, 0))) % 3
        kb_to_a_prop = rotate_about_axis(kb_to_a, -na, step, args.propagation_phase_sign)
        pair_vec_a = ka + kb_to_a_prop
        pair_vec_b = -(R_b_to_a.T @ rotate_about_axis(pair_vec_a, -na, (-step) % 3, args.propagation_phase_sign))
        W_pair[ia] += pair_vec_a
        W_pair[ib] += pair_vec_b
        scalar_strength = float(np.linalg.norm(pair_vec_a))
        scalar_pair[ia] += scalar_strength
        scalar_pair[ib] += scalar_strength
        pre_cos = float(np.dot(ka, kb_to_a) / ((np.linalg.norm(ka) * np.linalg.norm(kb_to_a)) + EPS))
        post_cos = float(np.dot(ka, kb_to_a_prop) / ((np.linalg.norm(ka) * np.linalg.norm(kb_to_a_prop)) + EPS))
        rows.append({
            'pair_index': k,
            'event_t': log.get('event_t', ''),
            'cascade_index': log.get('cascade_index', ''),
            'move_class': log.get('move_class', ''),
            'face_a': str(list(fa)),
            'face_b': str(list(fb)),
            'label_a': int(labels.get(ia, 0)),
            'label_b': int(labels.get(ib, 0)),
            'propagation_step_b_minus_a': step,
            'pre_transport_cosine': pre_cos,
            'post_transport_cosine': post_cos,
            'pre_abs_residual': 1.0 - abs(pre_cos),
            'post_abs_residual': 1.0 - abs(post_cos),
            'pre_signed_residual': 1.0 - pre_cos,
            'post_signed_residual': 1.0 - post_cos,
            'pair_vec_a_norm': float(np.linalg.norm(pair_vec_a)),
            'pair_vec_b_norm': float(np.linalg.norm(pair_vec_b)),
            'A_invariant': log.get('A_invariant', ''),
            'directed_imbalance': log.get('directed_imbalance', ''),
            'transverse_complementarity': log.get('transverse_complementarity', ''),
            'decision_used_delta_beta': log.get('decision_used_delta_beta', ''),
        })
    return W_pair, scalar_pair, rows


def phase_propagation_metrics(
    model: core.DynamicProvenanceGrowth,
    K: core.SimplicialComplex,
    pairing_log: List[dict],
    mode: str,
    args: argparse.Namespace,
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    faces = K.faces()
    topo = core.topology(K)
    H, eigs = hk.harmonic_basis_faces(K)
    W_raw = local_face_vectors(model, faces, args)
    labels, label_stats, constraint_rows = propagate_labels(model, K, pairing_log, W_raw, mode, args)
    W_pair, scalar_pair, pair_rows = transported_pair_fields_propagated(model, K, pairing_log, labels, args)
    W_pair_H = prev.vector_field_projection(H, W_pair)
    scalar_H = prev.scalar_projection(H, scalar_pair)
    pair_total = float(np.linalg.norm(W_pair)) + EPS
    pair_H_total = float(np.linalg.norm(W_pair_H))
    scalar_total = float(np.linalg.norm(scalar_pair)) + EPS
    scalar_H_total = float(np.linalg.norm(scalar_H))
    pair_raw_coh, pair_raw_axis_coh, pair_raw_support = prev.support_coherence(W_pair)
    pair_H_coh, pair_H_axis_coh, pair_H_support = prev.support_coherence(W_pair_H)
    pair3_coh, pair3_defect, pair3_count, pair3_rows = prev.three_face_coherence(faces, W_pair)
    pair_raw_kappa = prev.kappa_ratios(model, faces, W_pair)
    pair_H_kappa = prev.kappa_ratios(model, faces, W_pair_H)
    pre_abs = [float(r['pre_abs_residual']) for r in pair_rows]
    post_abs = [float(r['post_abs_residual']) for r in pair_rows]
    pre_signed = [float(r['pre_signed_residual']) for r in pair_rows]
    post_signed = [float(r['post_signed_residual']) for r in pair_rows]
    metrics = {
        'phase_mode': mode,
        'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'],
        'harmonic_dim_real': int(H.shape[1]) if H.ndim == 2 else 0,
        'applied_pair_count': len(pair_rows),
        'pair_transport_total_norm': pair_total - EPS,
        'pair_transport_harmonic_norm': pair_H_total,
        'pair_transport_harmonic_ratio': pair_H_total / pair_total,
        'pair_scalar_total_norm': scalar_total - EPS,
        'pair_scalar_harmonic_norm': scalar_H_total,
        'pair_scalar_harmonic_ratio': scalar_H_total / scalar_total,
        'pair_raw_orientation_coherence': pair_raw_coh,
        'pair_raw_axis_coherence': pair_raw_axis_coh,
        'pair_raw_support_count': pair_raw_support,
        'pair_orientation_coherence': pair_H_coh,
        'pair_axis_coherence': pair_H_axis_coh,
        'pair_H_support_count': pair_H_support,
        'pair_3face_coherence': pair3_coh,
        'pair_shared_edge_3face_phase_defect': pair3_defect,
        'pair_3face_count': pair3_count,
        'pre_interfan_abs_residual': float(np.mean(pre_abs)) if pre_abs else 0.0,
        'post_interfan_abs_residual': float(np.mean(post_abs)) if post_abs else 0.0,
        'interfan_abs_residual_reduction': (float(np.mean(pre_abs)) - float(np.mean(post_abs))) if pre_abs else 0.0,
        'pre_interfan_signed_residual': float(np.mean(pre_signed)) if pre_signed else 0.0,
        'post_interfan_signed_residual': float(np.mean(post_signed)) if post_signed else 0.0,
        'interfan_signed_residual_reduction': (float(np.mean(pre_signed)) - float(np.mean(post_signed))) if pre_signed else 0.0,
        'mean_post_pair_transport_cosine': float(np.mean([float(r['post_transport_cosine']) for r in pair_rows])) if pair_rows else 0.0,
        'decision_used_delta_beta_any': any(str(r.get('decision_used_delta_beta', '')).lower() == 'true' for r in pair_rows),
        'measured_delta_beta2_sum': sum(int(float(r.get('measured_delta_beta2', 0) or 0)) for r in pair_rows),
    }
    metrics.update(label_stats)
    metrics.update({f'pair_raw_{k}': v for k, v in pair_raw_kappa.items()})
    metrics.update({f'pair_{k}': v for k, v in pair_H_kappa.items()})
    face_rows = []
    H_norms = np.linalg.norm(W_pair_H, axis=1) if len(faces) else np.array([])
    P_norms = np.linalg.norm(W_pair, axis=1) if len(faces) else np.array([])
    for i, f in enumerate(faces):
        if i >= len(P_norms) or (P_norms[i] <= 0 and H_norms[i] <= 0):
            continue
        face_rows.append({
            'phase_mode': mode,
            'face': str(list(f)),
            'label': int(labels.get(i, 0)),
            'birth_sum_label': face_birth_label(model, f, args.erase_phase_for_strict_sym),
            'birth_orders': str([model.nodes[v].birth_order for v in f]),
            'pair_transport_norm': float(P_norms[i]),
            'pair_H_norm': float(H_norms[i]),
        })
    face_rows.sort(key=lambda r: (r['pair_H_norm'], r['pair_transport_norm']), reverse=True)
    return metrics, pair_rows, face_rows, constraint_rows + pair3_rows


def build_variant(variant: str, args: argparse.Namespace, out: Path):
    model = nl.build_model(variant, args)
    model.grow(args.max_level)
    baseline_K = core.build_dynamic_outward_ngf_complex(model)
    baseline_metrics = core.full_metrics(model, baseline_K, args.source)
    auto_K, birth_log, pairing_log, candidate_sample, scans = nl.build_nonlinear_auto_complex(model, args, out / variant, variant)
    auto_metrics = core.full_metrics(model, auto_K, args.source)
    return model, auto_K, baseline_metrics, auto_metrics, birth_log, pairing_log, candidate_sample, scans


def run_variant(variant: str, args: argparse.Namespace, out: Path) -> dict:
    vout = out / variant
    vout.mkdir(parents=True, exist_ok=True)
    model, K, baseline, auto, birth_log, pairing_log, candidate_sample, scans = build_variant(variant, args, out)
    mode_rows = []
    all_pair_rows: List[dict] = []
    all_face_rows: List[dict] = []
    all_constraint_rows: List[dict] = []
    for mode in args.propagation_modes:
        metrics, pair_rows, face_rows, constraint_rows = phase_propagation_metrics(model, K, pairing_log, mode, args)
        mode_rows.append(metrics)
        for r in pair_rows:
            rr = dict(r); rr['phase_mode'] = mode; all_pair_rows.append(rr)
        for r in face_rows:
            all_face_rows.append(r)
        for r in constraint_rows:
            rr = dict(r); rr['phase_mode'] = mode; all_constraint_rows.append(rr)
    write_csv(vout / 'birth_geometry_log.csv', birth_log)
    write_csv(vout / 'nonlinear_pairing_cascade_log.csv', pairing_log)
    write_csv(vout / 'phase_propagated_pair_transport_pairs.csv', all_pair_rows)
    write_csv(vout / 'phase_propagated_pair_transport_faces_top.csv', all_face_rows[:args.keep_top_faces])
    write_csv(vout / 'phase_constraints_and_3face_rows.csv', all_constraint_rows[:args.keep_top_faces * 5])
    summary = {
        'variant': variant,
        'max_level': args.max_level,
        'source': args.source,
        'antisym_eta': args.antisym_eta,
        'phase_sign': args.phase_sign,
        'propagation_phase_sign': args.propagation_phase_sign,
        'erase_phase_for_strict_sym': args.erase_phase_for_strict_sym,
        'baseline_metrics_legacy_core': baseline,
        'auto_metrics_legacy_core': auto,
        'phase_mode_rows': mode_rows,
        'automatic_pairings_applied': sum(1 for x in pairing_log if x.get('applied')),
        'automatic_pairing_attempts_logged': len(pairing_log),
        'births_with_cascade_logs': scans,
        'interpretation_flags': {
            'beta2_opened': auto['beta2'] > baseline['beta2'],
            'decision_used_delta_beta_any': any(m['decision_used_delta_beta_any'] for m in mode_rows),
            'strict_sym_pair_transport_killed': variant == 'strict_symmetrized_control' and all(m['pair_transport_total_norm'] <= args.zero_threshold for m in mode_rows),
            'any_phase_residual_reduction': any(m['interfan_abs_residual_reduction'] > args.residual_reduction_threshold for m in mode_rows),
            'any_pair_kappa_positive': any(m['pair_kappa_orientation_ratio'] > args.kappa_orientation_threshold for m in mode_rows),
            'any_pair_H_coherence_positive': any(m['pair_orientation_coherence'] > args.coherence_threshold for m in mode_rows),
        },
    }
    (vout / 'variant_interfan_phase_propagation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def write_comparative(out: Path, rows: List[dict]) -> None:
    flat = []
    for r in rows:
        a = r['auto_metrics_legacy_core']
        for m in r['phase_mode_rows']:
            flat.append({
                'variant': r['variant'],
                'phase_mode': m['phase_mode'],
                'beta0': a['beta0'], 'beta1': a['beta1'], 'beta2': a['beta2'], 'beta3': a['beta3'],
                'pairings': r['automatic_pairings_applied'],
                'pre_interfan_abs_residual': m['pre_interfan_abs_residual'],
                'post_interfan_abs_residual': m['post_interfan_abs_residual'],
                'interfan_abs_residual_reduction': m['interfan_abs_residual_reduction'],
                'phase_graph_edge_count': m['phase_graph_edge_count'],
                'phase_conflict_fraction': m['phase_conflict_fraction'],
                'pair_transport_harmonic_ratio': m['pair_transport_harmonic_ratio'],
                'pair_scalar_harmonic_ratio': m['pair_scalar_harmonic_ratio'],
                'pair_kappa_orientation_ratio': m['pair_kappa_orientation_ratio'],
                'pair_kappa_birth_orientation_ratio': m['pair_kappa_birth_orientation_ratio'],
                'pair_orientation_coherence': m['pair_orientation_coherence'],
                'pair_raw_orientation_coherence': m['pair_raw_orientation_coherence'],
                'pair_3face_coherence': m['pair_3face_coherence'],
                'mean_post_pair_transport_cosine': m['mean_post_pair_transport_cosine'],
                'decision_used_delta_beta_any': m['decision_used_delta_beta_any'],
            })
    write_csv(out / 'comparative_interfan_phase_propagation_summary.csv', flat)


def make_docs(summary: dict) -> Tuple[str, str, str, str]:
    rows = []
    for r in summary['variant_rows']:
        a = r['auto_metrics_legacy_core']
        for m in r['phase_mode_rows']:
            rows.append((r['variant'], a, m, r['automatic_pairings_applied']))
    lines = [
        '| variant | mode | beta | pairings | residual pre | residual post | reduction | graph edges | conflict frac | pair harm | pair kappa | pair H coh | used Δβ? |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant, a, m, pc in rows:
        lines.append(
            f"| {variant} | {m['phase_mode']} | ({a['beta0']},{a['beta1']},{a['beta2']},{a['beta3']}) | {pc} | "
            f"{m['pre_interfan_abs_residual']:.6g} | {m['post_interfan_abs_residual']:.6g} | {m['interfan_abs_residual_reduction']:.6g} | "
            f"{m['phase_graph_edge_count']} | {m['phase_conflict_fraction']:.6g} | {m['pair_transport_harmonic_ratio']:.6g} | "
            f"{m['pair_kappa_orientation_ratio']:.6g} | {m['pair_orientation_coherence']:.6g} | {m['decision_used_delta_beta_any']} |"
        )
    table = '\n'.join(lines)
    readme = """# Interfan phase propagation obstruction gate

Run:

```bash
python3 test_interfan_phase_propagation_obstruction_gate.py
```

This package extends the directed antisymmetric birth-transport + pairing-transport test.  It deliberately does **not** apply a final `sym(M)` in the tested vertex operator.
"""
    smd = f"""# SUMMARY — interfan phase propagation obstruction gate

Question:

```text
Can a Z3-compatible phase propagation, derived only from birth order, local response vectors, and the actual face/pair graph, reduce the interfan residual and make the pair-transported H2 sector oriented/coherent?
```

The tested local operator remains real and directed:

```text
M = metric_part + eta * strength * (q⊗h - h⊗q)
```

No final `sym(M)` is used in the tested operator.  No `i`, `J`, Hodge star, positivity, norm axiom, spin, or complex scalar is introduced.

Propagation modes:

```text
none              baseline from package 56
birth_sum         face label = sum(birth_order-1) mod 3
pair_graph_abs    Z3 labels from actual pairing edges, maximizing |local cosine|
pair_graph_signed Z3 labels from actual pairing edges, maximizing signed local cosine
face_graph_abs    shared-edge face graph + actual pair graph, maximizing |local cosine|
face_graph_signed shared-edge face graph + actual pair graph, maximizing signed local cosine
```

{table}

Interpret conservatively: pair-graph/signed modes are diagnostics for whether the obstruction is phase-propagation-compatible; they are not yet a theorem that such a propagation is forced by CNNA.
"""
    rmd = f"""# RESULTS — interfan phase propagation obstruction gate

## Comparative table

{table}

## Gate reading

- `residual pre/post` measures actual pair-edge interfan residual before/after the Z3 phase propagation.
- `graph edges` and `conflict frac` measure whether the propagated labels are compatible over the chosen graph.
- `pair harm` is H2 projection of the actual pair-transport field after propagation.
- `pair kappa` and `pair H coh` are the oriented/coherent harmonic diagnostics.
- `decision_used_delta_beta_any` must remain false.

A strong positive gate would require: residual reduction, strict-sym kill, and increased pair kappa / pair H coherence without using beta or H2 in the phase assignment.
"""
    audit = """# SOURCE AUDIT

Hard constraints:

- no final `sym(M)` in the tested directed vertex operator;
- antisymmetric term is birth-order-derived: `q⊗h - h⊗q`;
- phase propagation uses only birth labels, local real K vectors, shared-edge face graph, and actual pairing graph;
- H2/kappa metrics are evaluated after phase labels are assigned;
- beta changes are logged/diagnostic only, not used for move or phase decisions.

Limits:

- `pair_graph_*` and `face_graph_*` are diagnostic synchronization tests, not yet a derived CNNA law;
- rotations are finite real 3D coordinate transports used to compare local charts, not an imported complex scalar or J;
- NGF/CQNM remains only a comparison framework.
"""
    return smd, rmd, audit, readme


def package(out: Path, zip_path: Path) -> None:
    files = [
        Path(__file__).name,
        'test_pairing_transport_antisym_birth_coherence_gate.py',
        'test_nonlinear_asymmetry_cascade_growth.py',
        'test_harmonic_k_orientation_kappa_gate.py',
        'cnna_non_shelling_core.py',
        'test_interfan_transport_from_asymmetry_invariants.py',
        'test_growth_with_asymmetry_gated_complement_pairing.py',
    ]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if Path(f).exists():
                z.write(f, f)
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.resolve().relative_to(Path.cwd()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-level', type=int, default=2)
    ap.add_argument('--response-mode', choices=['linear', 'log', 'saturating', 'power_saturating', 'threshold_power'], default='power_saturating')
    ap.add_argument('--source', default='live', choices=['record', 'live', 'full', 'handoff', 'aging'])
    ap.add_argument('--transverse-amp', type=float, default=0.42)
    ap.add_argument('--nonlinear-gamma', type=float, default=1.65)
    ap.add_argument('--nonlinear-threshold', type=float, default=1.8)
    ap.add_argument('--cascade-A-threshold', type=float, default=0.18)
    ap.add_argument('--cascade-gamma', type=float, default=1.75)
    ap.add_argument('--cascade-transverse-gamma', type=float, default=1.25)
    ap.add_argument('--transverse-nonlinear-weight', type=float, default=1.4)
    ap.add_argument('--directed-nonlinear-weight', type=float, default=1.1)
    ap.add_argument('--cascade-fatigue', type=float, default=0.25)
    ap.add_argument('--cascade-rescan', action='store_true', default=True)
    ap.add_argument('--allow-reuse-faces', action='store_true')
    ap.add_argument('--allow-quotient', action='store_true')
    ap.add_argument('--max-boundary-faces', type=int, default=90)
    ap.add_argument('--max-single-vertices', type=int, default=12)
    ap.add_argument('--max-pair-candidates', type=int, default=2200)
    ap.add_argument('--max-rows', type=int, default=4400)
    ap.add_argument('--max-auto-pairings', type=int, default=2)
    ap.add_argument('--max-cascade-per-birth', type=int, default=2)
    ap.add_argument('--min-tets-before-pairing', type=int, default=4)
    ap.add_argument('--min-birth-time-before-pairing', type=int, default=4)
    ap.add_argument('--min-nonlinear-score', type=float, default=0.0)
    ap.add_argument('--keep-top-candidates', type=int, default=80)
    ap.add_argument('--keep-top-faces', type=int, default=160)
    ap.add_argument('--harmonic-positive-threshold', type=float, default=1e-4)
    ap.add_argument('--coherence-threshold', type=float, default=0.15)
    ap.add_argument('--kappa-orientation-threshold', type=float, default=0.15)
    ap.add_argument('--zero-threshold', type=float, default=1e-10)
    ap.add_argument('--residual-reduction-threshold', type=float, default=0.05)
    ap.add_argument('--antisym-eta', type=float, default=1.0)
    ap.add_argument('--phase-sign', type=int, default=1, choices=[-1, 1])
    ap.add_argument('--propagation-phase-sign', type=int, default=1, choices=[-1, 1])
    ap.add_argument('--erase-phase-for-strict-sym', action='store_true', default=True)
    ap.add_argument('--propagation-modes', nargs='*', default=['none', 'birth_sum', 'pair_graph_abs', 'pair_graph_signed', 'face_graph_abs', 'face_graph_signed'])
    ap.add_argument('--variants', nargs='*', default=['real_growth', 'strict_symmetrized_control', 'no_backreaction'])
    ap.add_argument('--out', default='interfan_phase_propagation_obstruction_out_L2')
    ap.add_argument('--zip', default='cnna_interfan_phase_propagation_obstruction_gate_pkg_L2.zip')
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = [run_variant(v, args, out) for v in args.variants]
    summary = {'args': vars(args), 'variant_rows': rows}
    (out / 'comparative_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_comparative(out, rows)
    smd, rmd, audit, readme = make_docs(summary)
    (out / 'SUMMARY.md').write_text(smd, encoding='utf-8')
    (out / 'RESULTS.md').write_text(rmd, encoding='utf-8')
    (out / 'SOURCE_AUDIT.md').write_text(audit, encoding='utf-8')
    (out / 'README.md').write_text(readme, encoding='utf-8')
    package(out, Path(args.zip))
    print(json.dumps({
        'zip': args.zip,
        'out': args.out,
        'summary': [
            {
                'variant': r['variant'],
                'auto_beta': [r['auto_metrics_legacy_core'][f'beta{i}'] for i in range(4)],
                'pairings': r['automatic_pairings_applied'],
                'modes': [
                    {
                        'phase_mode': m['phase_mode'],
                        'pre_residual': m['pre_interfan_abs_residual'],
                        'post_residual': m['post_interfan_abs_residual'],
                        'reduction': m['interfan_abs_residual_reduction'],
                        'pair_harmonic': m['pair_transport_harmonic_ratio'],
                        'pair_kappa': m['pair_kappa_orientation_ratio'],
                        'pair_H_coherence': m['pair_orientation_coherence'],
                        'conflict_fraction': m['phase_conflict_fraction'],
                        'used_delta_beta': m['decision_used_delta_beta_any'],
                    } for m in r['phase_mode_rows']
                ],
            } for r in rows
        ]
    }, indent=2))


if __name__ == '__main__':
    main()
