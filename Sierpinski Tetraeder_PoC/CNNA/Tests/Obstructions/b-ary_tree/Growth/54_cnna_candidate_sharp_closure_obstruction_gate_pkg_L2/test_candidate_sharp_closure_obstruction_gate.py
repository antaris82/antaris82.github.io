#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import test_2form_closure_and_3form_defect_gate as prev
import test_real_operator_composition_from_closed_2sector_gate as op

EPS = 1e-12
Face = Tuple[int, int, int]

MODEL_LABEL = (
    "CNNA deterministic growing primal simplicial complex; provenance tree as birth-history; "
    "candidate sharp-closure obstruction diagnostic on the real C2 carrier; "
    "NGF/CQNM only as comparison, not as derivation source"
)

ANTI_SMUGGLING_NOTE = (
    "No i, J, Hodge star, C*-adjoint, positivity axiom, norm axiom, spin structure, Fourier sign, "
    "branch cut, upper/lower half-plane convention, or external orientation package is used as input. "
    "The map A# = R A^T R is only a real finite-coefficient reversal diagnostic. "
    "It is not claimed to be a derived star operation."
)


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


def edge_operator(n: int, target: int, source: int, coeff: float = 1.0) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    A[target, source] = float(coeff)
    return A


def pair_transport_operators(K: prev.GrowthComplex, faces: List[Face], signed: bool) -> List[Tuple[str, np.ndarray]]:
    idx = {f: i for i, f in enumerate(faces)}
    ops: List[Tuple[str, np.ndarray]] = []
    seen = set()
    s = -1.0 if signed else 1.0
    for rec in K.cap_records:
        a = tuple(rec['base_face'])
        b = tuple(rec['partner_face'])
        if a not in idx or b not in idx:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        ia, ib = idx[a], idx[b]
        ops.append((f"P_pair_{rec['pair_index']}_base_to_partner", edge_operator(len(faces), ib, ia, s)))
        ops.append((f"P_pair_{rec['pair_index']}_partner_to_base", edge_operator(len(faces), ia, ib, s)))
    return ops


def compress_generators(U: np.ndarray, raw: List[Tuple[str, np.ndarray]], R_raw: np.ndarray, zero_threshold: float) -> Tuple[List[str], List[np.ndarray], np.ndarray, np.ndarray]:
    if U.size == 0 or U.shape[1] == 0:
        return [], [], np.zeros((0, 0)), np.zeros((0, 0))
    d = U.shape[1]
    I = np.eye(d)
    R = op.compress_operator(U, R_raw)
    out: List[Tuple[str, np.ndarray]] = []
    for name, Araw in raw:
        if name == 'I':
            A = I
        elif name == 'R':
            A = R
        else:
            A = op.compress_operator(U, Araw)
        if name == 'I' or float(np.linalg.norm(A)) > zero_threshold:
            out.append((name, A))
    return [x[0] for x in out], [x[1] for x in out], R, I


def residual_product_closure(basis: List[np.ndarray], generators: List[np.ndarray], Q: np.ndarray) -> Tuple[float, float, float]:
    abs_vals = []
    rel_vals = []
    material_rel_vals = []
    for A in basis:
        for G in generators:
            for P in (A @ G, G @ A):
                absr = op.residual_to_basis(Q, P)
                pn = float(np.linalg.norm(P))
                relr = absr / (pn + EPS)
                abs_vals.append(absr)
                rel_vals.append(relr)
                if pn > 1e-8:
                    material_rel_vals.append(relr)
    return (
        float(max(abs_vals)) if abs_vals else 0.0,
        float(max(rel_vals)) if rel_vals else 0.0,
        float(max(material_rel_vals)) if material_rel_vals else 0.0,
    )



def incremental_closure_seed(mats: List[np.ndarray], tol: float):
    basis_mats: List[np.ndarray] = []
    basis_vecs: List[np.ndarray] = []

    def append(A: np.ndarray) -> bool:
        nrm = float(np.linalg.norm(A))
        if nrm <= tol:
            return False
        v = A.reshape(-1).astype(float)
        r = v.copy()
        for q in basis_vecs:
            r -= q * float(np.dot(q, r))
        rn = float(np.linalg.norm(r))
        if rn / (nrm + EPS) > tol:
            q = r / (rn + EPS)
            basis_vecs.append(q)
            basis_mats.append(q.reshape(A.shape))
            return True
        return False

    for A in mats:
        append(A)
    return basis_mats, basis_vecs, append


def fast_product_closure(generators: List[Tuple[str, np.ndarray]], tol: float, max_iter: int, max_dim: int) -> Tuple[List[np.ndarray], List[dict], bool]:
    basis, _vecs, append = incremental_closure_seed([G for _n, G in generators], tol)
    rows: List[dict] = []
    gen_mats = [G for _n, G in generators]
    for it in range(max_iter):
        added = 0
        current = list(basis)
        for i, A in enumerate(current):
            for j, G in enumerate(gen_mats):
                for side, P in (('right', A @ G), ('left', G @ A)):
                    if append(P):
                        added += 1
                        rows.append({'iteration': it + 1, 'side': side, 'basis_index': i, 'generator_index': j, 'new_dim': len(basis)})
                        if len(basis) >= max_dim:
                            return basis, rows, True
        if added == 0:
            rows.append({'iteration': it + 1, 'side': 'none', 'basis_index': '', 'generator_index': '', 'new_dim': len(basis)})
            break
    return basis, rows, False


def fast_forced_sharp_product_closure(generators: List[Tuple[str, np.ndarray]], R: np.ndarray, tol: float, max_iter: int, max_dim: int) -> Tuple[List[np.ndarray], List[dict], bool]:
    # Forced diagnostic closure under the candidate # and left/right multiplication by the original seed generators.
    # This intentionally mirrors the product-closure rule used in the previous package and avoids silently
    # replacing the seed system by the full ambient matrix algebra.
    gen_mats = [G for _n, G in generators]
    basis, _vecs, append = incremental_closure_seed(gen_mats, tol)
    rows: List[dict] = []
    for it in range(max_iter):
        added = 0
        current = list(basis)
        for i, A in enumerate(current):
            if append(op.sharp(A, R)):
                added += 1
                rows.append({'iteration': it + 1, 'operation': 'sharp', 'left_index': i, 'right_index': '', 'new_dim': len(basis)})
                if len(basis) >= max_dim:
                    return basis, rows, True
        current = list(basis)
        for i, A in enumerate(current):
            for j, G in enumerate(gen_mats):
                for side, P in (('right_seed', A @ G), ('left_seed', G @ A)):
                    if append(P):
                        added += 1
                        rows.append({'iteration': it + 1, 'operation': side, 'left_index': i, 'right_index': j, 'new_dim': len(basis)})
                        if len(basis) >= max_dim:
                            return basis, rows, True
        if added == 0:
            rows.append({'iteration': it + 1, 'operation': 'none', 'left_index': '', 'right_index': '', 'new_dim': len(basis)})
            break
    return basis, rows, False

def forced_sharp_product_closure(generators: List[Tuple[str, np.ndarray]], R: np.ndarray, tol: float, max_iter: int, max_dim: int) -> Tuple[List[np.ndarray], List[dict], bool]:
    basis: List[np.ndarray] = []
    for _name, G in generators:
        op.independent_append(basis, G, tol)
    rows: List[dict] = []
    capped = False
    for it in range(max_iter):
        added = 0
        current = list(basis)
        for i, A in enumerate(current):
            S = op.sharp(A, R)
            if op.independent_append(basis, S, tol):
                added += 1
                rows.append({'iteration': it + 1, 'operation': 'sharp', 'left_index': i, 'right_index': '', 'new_dim': len(basis)})
                if len(basis) >= max_dim:
                    return basis, rows, True
        current = list(basis)
        for i, A in enumerate(current):
            for j, B in enumerate(current):
                P = A @ B
                if op.independent_append(basis, P, tol):
                    added += 1
                    rows.append({'iteration': it + 1, 'operation': 'product', 'left_index': i, 'right_index': j, 'new_dim': len(basis)})
                    if len(basis) >= max_dim:
                        return basis, rows, True
        if added == 0:
            rows.append({'iteration': it + 1, 'operation': 'none', 'left_index': '', 'right_index': '', 'new_dim': len(basis)})
            break
    return basis, rows, capped


def top_sharp_obstructions(prefix: str, names: List[str], mats: List[np.ndarray], Q: np.ndarray, R: np.ndarray, topn: int) -> List[dict]:
    rows = []
    for name, A in zip(names, mats):
        S = op.sharp(A, R)
        rr = op.rel_residual_to_basis(Q, S)
        rows.append({'family': prefix, 'operator': name, 'sharp_relative_residual_to_span': rr, 'operator_norm_diagnostic': float(np.linalg.norm(A))})
    rows.sort(key=lambda r: r['sharp_relative_residual_to_span'], reverse=True)
    return rows[:topn]


def analyze_family(case_name: str, family_name: str, names: List[str], mats: List[np.ndarray], R: np.ndarray, args: argparse.Namespace, vout: Path) -> dict:
    gens = list(zip(names, mats))
    if not mats:
        return {
            'case': case_name, 'family': family_name, 'generator_count': 0, 'initial_span_dim': 0,
            'initial_product_max_residual': 0.0, 'initial_sharp_max_residual': 0.0,
            'product_algebra_dim': 0, 'product_algebra_sharp_max_residual': 0.0,
            'forced_sharp_product_dim': 0, 'forced_extra_dim_over_product': 0,
            'forced_sharp_product_closed': True, 'forced_capped': False,
            'sharp_obstruction_present': False,
        }
    Q0 = op.orthonormal_basis_from_mats(mats, args.tol)
    initial_dim = int(Q0.shape[1])
    max_prod, mean_prod, prod_rows = op.product_residuals(names, mats, Q0, args.keep_top)
    max_sharp, mean_sharp, sharp_rows = op.sharp_residuals(names, mats, Q0, R, args.keep_top)

    prod_basis, prod_growth, prod_capped = fast_product_closure(gens, args.tol, args.max_product_closure_iter, args.max_algebra_dim)
    Qp = op.orthonormal_basis_from_mats(prod_basis, args.tol)
    prod_dim = int(Qp.shape[1])
    post_abs, post_rel, post_rel_material = residual_product_closure(prod_basis, mats, Qp)
    alg_names = [f'B{i}' for i in range(len(prod_basis))]
    alg_sharp_max, alg_sharp_mean, alg_sharp_rows = op.sharp_residuals(alg_names, prod_basis, Qp, R, args.keep_top)

    forced_basis, forced_growth, forced_capped = fast_forced_sharp_product_closure(gens, R, args.tol, args.max_forced_iter, args.max_algebra_dim)
    Qf = op.orthonormal_basis_from_mats(forced_basis, args.tol)
    forced_dim = int(Qf.shape[1])
    forced_post_abs, forced_post_rel, forced_post_rel_material = residual_product_closure(forced_basis, mats, Qf)
    forced_names = [f'C{i}' for i in range(len(forced_basis))]
    forced_sharp_max, forced_sharp_mean, forced_sharp_rows = op.sharp_residuals(forced_names, forced_basis, Qf, R, args.keep_top)

    safe_family = family_name.replace('/', '_')
    write_csv(vout / f'{safe_family}_initial_product_residuals_top.csv', prod_rows)
    write_csv(vout / f'{safe_family}_initial_sharp_residuals_top.csv', sharp_rows)
    write_csv(vout / f'{safe_family}_product_closure_growth_log.csv', prod_growth)
    write_csv(vout / f'{safe_family}_product_algebra_sharp_residuals_top.csv', alg_sharp_rows)
    write_csv(vout / f'{safe_family}_forced_sharp_product_growth_log.csv', forced_growth)
    write_csv(vout / f'{safe_family}_forced_sharp_product_residuals_top.csv', forced_sharp_rows)
    write_csv(vout / f'{safe_family}_generator_summary.csv', op.matrix_summary_rows(names, mats, R))

    matrix_dim = int(mats[0].shape[0]) if mats else 0
    full_matrix_dim = int(matrix_dim * matrix_dim)
    product_full_matrix_saturation = bool(full_matrix_dim > 0 and prod_dim >= full_matrix_dim)
    forced_full_matrix_saturation = bool(full_matrix_dim > 0 and forced_dim >= full_matrix_dim)
    effective_forced_capped = bool(forced_capped and not forced_full_matrix_saturation)

    return {
        'case': case_name,
        'family': family_name,
        'carrier_operator_matrix_dim': matrix_dim,
        'full_matrix_dim_on_carrier': full_matrix_dim,
        'generator_count': len(mats),
        'generator_names': names,
        'initial_span_dim': initial_dim,
        'initial_product_max_residual': max_prod,
        'initial_product_mean_residual': mean_prod,
        'initial_sharp_max_residual': max_sharp,
        'initial_sharp_mean_residual': mean_sharp,
        'initial_product_closed': max_prod <= args.closure_threshold,
        'initial_sharp_closed': max_sharp <= args.closure_threshold,
        'product_algebra_dim': prod_dim,
        'product_algebra_full_matrix_saturation': product_full_matrix_saturation,
        'product_algebra_capped': bool(prod_capped and not product_full_matrix_saturation),
        'product_algebra_post_abs_residual': post_abs,
        'product_algebra_post_rel_material_residual': post_rel_material,
        'product_algebra_closed': (post_abs <= args.closure_threshold and not prod_capped),
        'product_algebra_sharp_max_residual': alg_sharp_max,
        'product_algebra_sharp_mean_residual': alg_sharp_mean,
        'product_algebra_sharp_closed': alg_sharp_max <= args.closure_threshold,
        'forced_sharp_product_dim': forced_dim,
        'forced_extra_dim_over_product': int(max(0, forced_dim - prod_dim)),
        'forced_full_matrix_saturation': forced_full_matrix_saturation,
        'forced_capped': effective_forced_capped,
        'forced_product_post_abs_residual': forced_post_abs,
        'forced_product_post_rel_material_residual': forced_post_rel_material,
        'forced_sharp_max_residual': forced_sharp_max,
        'forced_sharp_mean_residual': forced_sharp_mean,
        'forced_sharp_product_closed': (forced_post_abs <= args.closure_threshold and forced_sharp_max <= args.closure_threshold and not effective_forced_capped),
        'sharp_obstruction_present': alg_sharp_max > args.closure_threshold,
        'sharp_obstruction_top': top_sharp_obstructions(family_name, alg_names, prod_basis, Qp, R, args.keep_top)[:10],
    }


def raw_family(name: str, I: np.ndarray, R: np.ndarray, M_closed: np.ndarray, M_harmonic: np.ndarray, cap_ops: List[Tuple[str, np.ndarray]], pair_ops: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    if name == 'diag_core':
        return [('I', I), ('R', R), ('M_closed', M_closed), ('M_harmonic', M_harmonic)]
    if name == 'diag_no_R':
        return [('I', I), ('M_closed', M_closed), ('M_harmonic', M_harmonic)]
    if name == 'R_only':
        return [('I', I), ('R', R)]
    if name == 'cap_only':
        return [('I', I), ('R', R)] + [(f'cap::{n}', A) for n, A in cap_ops]
    if name == 'pair_only':
        return [('I', I), ('R', R)] + [(f'pair::{n}', A) for n, A in pair_ops]
    if name == 'cap_plus_diag':
        return [('I', I), ('R', R), ('M_closed', M_closed), ('M_harmonic', M_harmonic)] + [(f'cap::{n}', A) for n, A in cap_ops]
    if name == 'pair_plus_diag':
        return [('I', I), ('R', R), ('M_closed', M_closed), ('M_harmonic', M_harmonic)] + [(f'pair::{n}', A) for n, A in pair_ops]
    if name == 'cap_pair_no_diag':
        return [('I', I), ('R', R)] + [(f'cap::{n}', A) for n, A in cap_ops] + [(f'pair::{n}', A) for n, A in pair_ops]
    if name == 'all_seed':
        return [('I', I), ('R', R), ('M_closed', M_closed), ('M_harmonic', M_harmonic)] + [(f'cap::{n}', A) for n, A in cap_ops] + [(f'pair::{n}', A) for n, A in pair_ops]
    raise ValueError(f'unknown family {name}')


def run_case(case: dict, signed: bool, args: argparse.Namespace, out: Path) -> dict:
    signed_label = 'signed_R' if signed else 'unsigned_R'
    case_id = f"{case['variant']}__{signed_label}"
    vout = out / case_id
    vout.mkdir(parents=True, exist_ok=True)

    K = prev.build_growth(args.max_level, case['strict_sym'], case['use_backreaction'], case['pairings'])
    topo = prev.topology(K)
    cd = prev.chain_data(K)
    faces = cd['F']
    k2, scalar, pair_rows = prev.cochain_K2(K, case['source'], case['mode'], case['strict_sym'])
    dec = prev.decompose_2cochain(K, k2)
    c = dec['closed_vector']
    h = dec['harmonic_vector']
    e = dec['exact_vector']

    n = len(faces)
    I_raw = np.eye(n)
    R_raw = op.pair_reversal_matrix(K, faces, signed=signed)
    M_closed_raw = op.diag_operator(c)
    M_harmonic_raw = op.diag_operator(h)
    cap_raw = op.cap_operators(K, faces, case['source'])
    pair_raw = pair_transport_operators(K, faces, signed=signed)

    U = op.build_carrier_basis(K, faces, case['source'], R_raw, k2, c, h, e, False, args.tol)
    carrier_dim = int(U.shape[1]) if U.ndim == 2 else 0
    R_comp = op.compress_operator(U, R_raw) if carrier_dim else np.zeros((0, 0))
    I_comp = np.eye(carrier_dim) if carrier_dim else np.zeros((0, 0))
    R2_resid = float(np.linalg.norm(R_comp @ R_comp - I_comp)) / (float(np.linalg.norm(I_comp)) + EPS) if carrier_dim else 0.0
    R_sym_resid = float(np.linalg.norm(R_comp.T - R_comp)) / (float(np.linalg.norm(R_comp)) + EPS) if carrier_dim else 0.0
    R_nonid = int(np.sum(np.abs(R_comp - I_comp) > args.zero_threshold)) if carrier_dim else 0

    family_summaries: List[dict] = []
    for family in args.families:
        raw = raw_family(family, I_raw, R_raw, M_closed_raw, M_harmonic_raw, cap_raw, pair_raw)
        names, mats, R, _Ic = compress_generators(U, raw, R_raw, args.zero_threshold)
        family_summaries.append(analyze_family(case_id, family, names, mats, R, args, vout))

    write_csv(vout / 'case_family_summary.csv', flatten_family_rows(family_summaries, topo, dec, carrier_dim, signed, case, R2_resid, R_sym_resid, R_nonid, K, pair_rows))
    write_csv(vout / 'pairing_cap_log.csv', K.pairing_log)
    write_csv(vout / 'pairing_2form_rows.csv', pair_rows)
    summary = {
        'case_id': case_id,
        'variant': case['variant'],
        'signed_pair_reversal': signed,
        'source': case['source'],
        'mode': case['mode'],
        'strict_symmetrized': case['strict_sym'],
        'use_backreaction': case['use_backreaction'],
        'pairings_requested': case['pairings'],
        'model_label': MODEL_LABEL,
        'anti_smuggling_note': ANTI_SMUGGLING_NOTE,
        'topology': topo,
        'chain_dimensions': {'C2': n, 'C3': len(cd['T']), 'carrier_dim': carrier_dim},
        'K2_sector': {
            'closed_ratio': float(dec['closed_ratio']),
            'exact_ratio': float(dec['exact_ratio']),
            'harmonic_ratio': float(dec['harmonic_ratio']),
            'defect_ratio': float(dec['defect_ratio']),
            'harmonic_dim_real': int(dec['harmonic_dim_real']),
        },
        'pair_reversal_candidate': {
            'R_squared_identity_residual': R2_resid,
            'R_symmetric_residual': R_sym_resid,
            'R_nonidentity_entry_count': R_nonid,
            'is_nontrivial': R_nonid > 0,
        },
        'applied_pair_count': sum(1 for x in K.pairing_log if x.get('applied')),
        'decision_used_delta_beta_any': any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows),
        'families': family_summaries,
    }
    (vout / 'variant_candidate_sharp_obstruction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def flatten_family_rows(families: List[dict], topo: dict, dec: dict, carrier_dim: int, signed: bool, case: dict, R2: float, Rsym: float, Rnonid: int, K: prev.GrowthComplex, pair_rows: List[dict]) -> List[dict]:
    rows = []
    used = any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows)
    for f in families:
        rows.append({
            'variant': case['variant'],
            'source': case['source'],
            'mode': case['mode'],
            'signed_pair_reversal': signed,
            'family': f['family'],
            'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'],
            'H2_dim': int(dec['harmonic_dim_real']),
            'K_harmonic_ratio': float(dec['harmonic_ratio']),
            'K_defect_ratio': float(dec['defect_ratio']),
            'carrier_dim': carrier_dim,
            'pairings': sum(1 for x in K.pairing_log if x.get('applied')),
            'R2_residual': R2,
            'R_symmetric_residual': Rsym,
            'R_nonidentity_entries': Rnonid,
            'generators': f['generator_count'],
            'initial_span_dim': f['initial_span_dim'],
            'initial_product_max_residual': f['initial_product_max_residual'],
            'initial_sharp_max_residual': f['initial_sharp_max_residual'],
            'product_algebra_dim': f['product_algebra_dim'],
            'product_algebra_full_matrix_saturation': f.get('product_algebra_full_matrix_saturation', False),
            'product_algebra_sharp_max_residual': f['product_algebra_sharp_max_residual'],
            'forced_sharp_product_dim': f['forced_sharp_product_dim'],
            'forced_extra_dim_over_product': f['forced_extra_dim_over_product'],
            'forced_full_matrix_saturation': f.get('forced_full_matrix_saturation', False),
            'forced_closed': f['forced_sharp_product_closed'],
            'forced_capped': f['forced_capped'],
            'sharp_obstruction_present': f['sharp_obstruction_present'],
            'used_delta_beta_any': used,
        })
    return rows


def flatten_all(summaries: List[dict]) -> List[dict]:
    rows = []
    for s in summaries:
        topo = s['topology']; dec = s['K2_sector']; carrier_dim = s['chain_dimensions']['carrier_dim']; pr = s['pair_reversal_candidate']
        for f in s['families']:
            rows.append({
                'case_id': s['case_id'],
                'variant': s['variant'],
                'source': s['source'],
                'mode': s['mode'],
                'signed_pair_reversal': s['signed_pair_reversal'],
                'family': f['family'],
                'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'],
                'H2_dim': dec['harmonic_dim_real'],
                'K_harmonic_ratio': dec['harmonic_ratio'],
                'K_defect_ratio': dec['defect_ratio'],
                'carrier_dim': carrier_dim,
                'pairings': s['applied_pair_count'],
                'R2_residual': pr['R_squared_identity_residual'],
                'R_nonidentity_entries': pr['R_nonidentity_entry_count'],
                'generators': f['generator_count'],
                'initial_span_dim': f['initial_span_dim'],
                'initial_product_max_residual': f['initial_product_max_residual'],
                'initial_sharp_max_residual': f['initial_sharp_max_residual'],
                'product_algebra_dim': f['product_algebra_dim'],
                'product_algebra_full_matrix_saturation': f.get('product_algebra_full_matrix_saturation', False),
                'product_algebra_sharp_max_residual': f['product_algebra_sharp_max_residual'],
                'forced_sharp_product_dim': f['forced_sharp_product_dim'],
                'forced_extra_dim_over_product': f['forced_extra_dim_over_product'],
                'forced_full_matrix_saturation': f.get('forced_full_matrix_saturation', False),
                'forced_closed': f['forced_sharp_product_closed'],
                'forced_capped': f['forced_capped'],
                'sharp_obstruction_present': f['sharp_obstruction_present'],
                'used_delta_beta_any': s['decision_used_delta_beta_any'],
            })
    return rows


def make_docs(summary: dict) -> Tuple[str, str, str, str]:
    rows = flatten_all(summary['case_rows'])
    focus = [r for r in rows if r['variant'] == 'real_growth_live_pair_plus_response' and r['signed_pair_reversal'] is True]
    lines = [
        '| family | gen | init dim | init prod resid | init # resid | prod alg dim | full M? | prod alg # resid | forced dim | extra dim | forced closed? |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in focus:
        lines.append(
            f"| {r['family']} | {r['generators']} | {r['initial_span_dim']} | {r['initial_product_max_residual']:.6g} | "
            f"{r['initial_sharp_max_residual']:.6g} | {r['product_algebra_dim']} | {r.get('product_algebra_full_matrix_saturation', False)} | {r['product_algebra_sharp_max_residual']:.6g} | "
            f"{r['forced_sharp_product_dim']} | {r['forced_extra_dim_over_product']} | {r['forced_closed']} |"
        )
    table = '\n'.join(lines)

    # compact global obstruction ranking
    ranked = sorted(rows, key=lambda r: (r['sharp_obstruction_present'], r['product_algebra_sharp_max_residual']), reverse=True)
    rank_lines = [
        '| case | R | family | prod alg dim | # residual | forced dim | extra |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for r in ranked[:18]:
        rank_lines.append(
            f"| {r['variant']}:{r['source']}:{r['mode']} | {'signed' if r['signed_pair_reversal'] else 'unsigned'} | {r['family']} | "
            f"{r['product_algebra_dim']} | {r['product_algebra_sharp_max_residual']:.6g} | {r['forced_sharp_product_dim']} | {r['forced_extra_dim_over_product']} |"
        )
    rank_table = '\n'.join(rank_lines)

    summary_md = f"""# SUMMARY — candidate sharp-closure obstruction gate

## Model label

{MODEL_LABEL}

## Anti-smuggling constraint

{ANTI_SMUGGLING_NOTE}

## Gate question

The previous package found that the generated real product algebra on the C² carrier is product-closed after adding products, but is not stable under the candidate reversal

```text
A# = R_pair A^T R_pair.
```

This package localizes the obstruction by splitting generator families:

```text
{{I,R,M_closed,M_harmonic}}
cap maps only
pair maps only
signed vs unsigned R
record/live/full
forced #-closure
```

## Focus table: real_growth_live_pair_plus_response with signed R

{table}

## Conservative reading

- `diag_core` is the smallest direct test of the closed/harmonic diagonal 2-sector plus pair-reversal.
- `cap_only` isolates cap-boundary rank-one maps.
- `pair_only` isolates pair transport/swap maps.
- `forced_sharp_product_dim` measures the additional algebraic material required if one insists on # stability.
- A small forced extra dimension is a compatibility hint; a large jump means the candidate # is not native to the seed family.
- Full-matrix saturation is not a success criterion: it means # compatibility was bought by expanding to the whole finite carrier algebra, which is too nonselective for a derived minimal structure.

No row is a `*`-derivation.  The only legitimate claim is compatibility or obstruction of this finite real reversal diagnostic.
"""
    results_md = f"""# RESULTS — candidate sharp-closure obstruction gate

## Focus table: real_growth_live_pair_plus_response with signed R

{table}

## Highest # obstruction rows

{rank_table}

## Main interpretation

The test separates the obstruction into families rather than treating the previous product algebra as one black box.

Read the result in three layers:

```text
1. initial # residual:
   whether the raw generator span is already # stable.

2. product algebra # residual:
   whether the product-closed algebra from the raw generators is # stable.

3. forced #-product dimension:
   how much extra finite real operator material is needed if # stability is imposed.
```

A derived real `*`-candidate would require product closure and # closure to appear without a large forced enlargement, without full-matrix saturation as the only repair, and without importing positivity, norm, Hodge duality, or complex scalars.

## Negative-control condition

`strict_symmetrized_control` remains trivial when β₂ and the real C² carrier vanish.  `decision_used_delta_beta_any` remains false in all generated rows.

## Next test

```text
test_native_reversal_search_gate.py
```

Goal: do not force `A# = R A^T R`.  Instead, search within the actually generated real product algebra for an internally defined involutive anti-automorphism candidate built from available real operators only: `R`, pair maps, cap maps, and closed/harmonic diagonal actions.  The test should reject candidates that require full ambient matrix algebra or an imported transpose/inner-product interpretation.
"""
    audit_md = f"""# SOURCE AUDIT — candidate sharp-closure obstruction gate

## Inherited chain

- `test_2form_closure_and_3form_defect_gate.py`: constructs the deterministic growing primal simplicial complex and the real 2-cochain/3-defect decomposition.
- `test_real_operator_composition_from_closed_2sector_gate.py`: constructs the previous real operator seed and product-closure audit.
- This package imports both and adds only a split obstruction audit over existing real data.

## Forum-derived warning preserved

A sign switch such as `i ↔ -i` is clean only when the whole attached structure package is transported coherently: orientation, upper/lower half-plane, argument branch, logarithm/square-root branches, Fourier sign, causal sign, positivity, and adjoint conventions.  This package does not introduce any such package and therefore does not pretend that a partial sign flip solves anything.

## Candidate # status

`A# = R A^T R` remains a diagnostic formula.  `A^T` is finite coefficient dualization in the selected face basis, not a derived physical adjoint.  If # stability requires a large forced closure, the correct interpretation is obstruction/localization, not success.
"""
    readme_md = """# Candidate sharp-closure obstruction gate

Run:

```bash
python3 test_candidate_sharp_closure_obstruction_gate.py
```

Main outputs:

```text
candidate_sharp_closure_obstruction_out_L2/RESULTS.md
candidate_sharp_closure_obstruction_out_L2/SUMMARY.md
candidate_sharp_closure_obstruction_out_L2/comparative_candidate_sharp_obstruction_summary.csv
candidate_sharp_closure_obstruction_out_L2/comparative_summary.json
```

This is a CNNA deterministic growing primal simplicial complex test.  NGF/CQNM remains a comparison frame only.
"""
    return summary_md, results_md, audit_md, readme_md


def package(out: Path, zip_path: Path) -> None:
    files = [
        Path(__file__).name,
        'test_2form_closure_and_3form_defect_gate.py',
        'test_real_operator_composition_from_closed_2sector_gate.py',
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
    ap.add_argument('--out', default='candidate_sharp_closure_obstruction_out_L2')
    ap.add_argument('--zip', default='cnna_candidate_sharp_closure_obstruction_gate_pkg_L2.zip')
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--closure-threshold', type=float, default=1e-8)
    ap.add_argument('--zero-threshold', type=float, default=1e-12)
    ap.add_argument('--max-product-closure-iter', type=int, default=8)
    ap.add_argument('--max-forced-iter', type=int, default=5)
    ap.add_argument('--max-algebra-dim', type=int, default=48)
    ap.add_argument('--keep-top', type=int, default=50)
    ap.add_argument('--families', nargs='*', default=['diag_core', 'cap_only', 'pair_only', 'all_seed'])
    args = ap.parse_args()

    cases = [
        {'variant': 'real_growth_live_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'live', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_record_only_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'record', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_record_plus_live_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'full', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_live_pair_only', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'live', 'mode': 'pair_only'},
        {'variant': 'strict_symmetrized_control', 'strict_sym': True, 'use_backreaction': False, 'pairings': 0, 'source': 'record', 'mode': 'pair_plus_response'},
        {'variant': 'no_backreaction_record_pair_plus_response', 'strict_sym': False, 'use_backreaction': False, 'pairings': 2, 'source': 'record', 'mode': 'pair_plus_response'},
    ]

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = []
    for case in cases:
        for signed in (True, False):
            rows.append(run_case(case, signed, args, out))
    summary = {'args': vars(args), 'case_rows': rows}
    (out / 'comparative_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(out / 'comparative_candidate_sharp_obstruction_summary.csv', flatten_all(rows))
    smd, rmd, audit, readme = make_docs(summary)
    (out / 'SUMMARY.md').write_text(smd, encoding='utf-8')
    (out / 'RESULTS.md').write_text(rmd, encoding='utf-8')
    (out / 'SOURCE_AUDIT.md').write_text(audit, encoding='utf-8')
    (out / 'README.md').write_text(readme, encoding='utf-8')
    package(out, Path(args.zip))
    focus = [r for r in flatten_all(rows) if r['variant'] == 'real_growth_live_pair_plus_response' and r['signed_pair_reversal'] is True]
    print(json.dumps({'zip': args.zip, 'out': args.out, 'focus': focus}, indent=2))


if __name__ == '__main__':
    main()
