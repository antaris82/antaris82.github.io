#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import test_2form_closure_and_3form_defect_gate as prev

EPS = 1e-12
Face = Tuple[int, int, int]

MODEL_LABEL = (
    "CNNA deterministic growing primal simplicial complex; provenance tree as birth-history; "
    "real closed/harmonic 2-sector operator-composition diagnostic; NGF/CQNM only as comparison"
)

ANTI_SMUGGLING_NOTE = (
    "No i, J, Hodge star, positivity axiom, C*-norm, spin structure, Fourier sign, branch cut, "
    "upper/lower half-plane convention, or external orientation package is used as input. "
    "The candidate sharp map is only a real coefficient-dual plus cap/pair reversal diagnostic; "
    "it is not claimed to be a derived C*-adjoint."
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


def mat_vec(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float).reshape(-1)


def orthonormal_basis_from_mats(mats: List[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    if not mats:
        return np.zeros((0, 0), dtype=float)
    V = np.column_stack([mat_vec(A) for A in mats])
    if V.size == 0:
        return np.zeros((V.shape[0], 0), dtype=float)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    if len(s) == 0:
        return np.zeros((V.shape[0], 0), dtype=float)
    keep = s > max(tol, tol * float(s[0]))
    return U[:, keep]


def residual_to_basis(Q: np.ndarray, A: np.ndarray) -> float:
    v = mat_vec(A)
    if Q.size == 0 or Q.shape[1] == 0:
        return float(np.linalg.norm(v))
    r = v - Q @ (Q.T @ v)
    return float(np.linalg.norm(r))


def rel_residual_to_basis(Q: np.ndarray, A: np.ndarray) -> float:
    return residual_to_basis(Q, A) / (float(np.linalg.norm(A)) + EPS)


def independent_append(mats: List[np.ndarray], A: np.ndarray, tol: float) -> bool:
    nrm = float(np.linalg.norm(A))
    if nrm <= tol:
        return False
    if not mats:
        mats.append(A.copy() / (nrm + EPS))
        return True
    Q = orthonormal_basis_from_mats(mats, tol)
    v = mat_vec(A)
    r = v - Q @ (Q.T @ v) if Q.size and Q.shape[1] else v
    rr = float(np.linalg.norm(r)) / (nrm + EPS)
    if rr > tol:
        mats.append((r / (float(np.linalg.norm(r)) + EPS)).reshape(A.shape))
        return True
    return False


def closure_under_products(generators: List[Tuple[str, np.ndarray]], tol: float, max_iter: int, max_dim: int) -> Tuple[List[np.ndarray], List[dict], bool]:
    basis: List[np.ndarray] = []
    for _name, G in generators:
        independent_append(basis, G, tol)
    rows: List[dict] = []
    capped = False
    for it in range(max_iter):
        added = 0
        current = list(basis)
        gen_mats = [G for _name, G in generators]
        for i, A in enumerate(current):
            for j, G in enumerate(gen_mats):
                for side, P in [('right', A @ G), ('left', G @ A)]:
                    if independent_append(basis, P, tol):
                        added += 1
                        rows.append({'iteration': it + 1, 'side': side, 'basis_index': i, 'generator_index': j, 'new_dim': len(basis)})
                        if len(basis) >= max_dim:
                            capped = True
                            return basis, rows, capped
        if added == 0:
            rows.append({'iteration': it + 1, 'side': 'none', 'basis_index': '', 'generator_index': '', 'new_dim': len(basis)})
            break
    return basis, rows, capped


def pair_reversal_matrix(K: prev.GrowthComplex, faces: List[Face], signed: bool) -> np.ndarray:
    n = len(faces)
    idx = {f: i for i, f in enumerate(faces)}
    R = np.eye(n)
    visited = set()
    for rec in K.cap_records:
        a = tuple(rec['base_face'])
        b = tuple(rec['partner_face'])
        if a not in idx or b not in idx:
            continue
        key = tuple(sorted((a, b)))
        if key in visited:
            continue
        visited.add(key)
        ia, ib = idx[a], idx[b]
        R[ia, ia] = 0.0
        R[ib, ib] = 0.0
        s = -1.0 if signed else 1.0
        R[ib, ia] = s
        R[ia, ib] = s
    return R


def cap_boundary_vector(K: prev.GrowthComplex, faces: List[Face], rec: dict, source: str) -> np.ndarray:
    idx = {f: i for i, f in enumerate(faces)}
    q = np.zeros(len(faces), dtype=float)
    coeffs = prev.cap_boundary_coefficients(tuple(rec['base_face']), int(rec['cap_vertex']))
    strength_key = 'strength_' + source
    strength = float(rec[strength_key] if strength_key in rec else rec.get('strength_full', 0.0))
    for face, coeff in coeffs.items():
        if face in idx:
            q[idx[face]] += strength * float(coeff)
    return q


def selector_vector(faces: List[Face], face: Face) -> np.ndarray:
    e = np.zeros(len(faces), dtype=float)
    idx = {f: i for i, f in enumerate(faces)}
    if tuple(face) in idx:
        e[idx[tuple(face)]] = 1.0
    return e


def diag_operator(x: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(x, dtype=float))


def cap_operators(K: prev.GrowthComplex, faces: List[Face], source: str) -> List[Tuple[str, np.ndarray]]:
    ops: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for rec in K.cap_records:
        base = tuple(rec['base_face'])
        partner = tuple(rec['partner_face'])
        key = (base, partner, int(rec['cap_vertex']))
        if key in seen:
            continue
        seen.add(key)
        q = cap_boundary_vector(K, faces, rec, source)
        eb = selector_vector(faces, base)
        ep = selector_vector(faces, partner)
        if float(np.linalg.norm(q)) > EPS and float(np.linalg.norm(eb)) > EPS:
            ops.append((f"A_cap_{rec['pair_index']}_base", np.outer(q, eb)))
        if float(np.linalg.norm(q)) > EPS and float(np.linalg.norm(ep)) > EPS:
            # partner-to-cap map is included as a real transport diagnostic.  The minus sign is the cap/pair reversal convention already present in the 2-form boundary incidence gauge, not an ontic orientation input.
            ops.append((f"A_cap_{rec['pair_index']}_partner", np.outer(-q, ep)))
    return ops


def product_residuals(names: List[str], mats: List[np.ndarray], Q: np.ndarray, topn: int = 30) -> Tuple[float, float, List[dict]]:
    rows = []
    vals = []
    for i, A in enumerate(mats):
        for j, B in enumerate(mats):
            P = A @ B
            rr = rel_residual_to_basis(Q, P)
            vals.append(rr)
            rows.append({'left': names[i], 'right': names[j], 'relative_product_residual_to_initial_span': rr})
    rows.sort(key=lambda r: r['relative_product_residual_to_initial_span'], reverse=True)
    return (float(max(vals)) if vals else 0.0, float(np.mean(vals)) if vals else 0.0, rows[:topn])


def sharp(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    # Candidate only: real coefficient-dualization plus cap/pair reversal transport.
    # This is an anti-involution on matrices when R^2=I and R^T=R, but it is not automatically a physical or C*-adjoint.
    return R @ A.T @ R


def sharp_residuals(names: List[str], mats: List[np.ndarray], Q: np.ndarray, R: np.ndarray, topn: int = 30) -> Tuple[float, float, List[dict]]:
    rows = []
    vals = []
    for name, A in zip(names, mats):
        S = sharp(A, R)
        rr = rel_residual_to_basis(Q, S)
        vals.append(rr)
        rows.append({'operator': name, 'relative_sharp_residual_to_span': rr})
    rows.sort(key=lambda r: r['relative_sharp_residual_to_span'], reverse=True)
    return (float(max(vals)) if vals else 0.0, float(np.mean(vals)) if vals else 0.0, rows[:topn])


def anti_automorphism_residual(names: List[str], mats: List[np.ndarray], R: np.ndarray, topn: int = 30) -> Tuple[float, float, List[dict]]:
    rows = []
    vals = []
    for i, A in enumerate(mats):
        for j, B in enumerate(mats):
            lhs = sharp(A @ B, R)
            rhs = sharp(B, R) @ sharp(A, R)
            rr = float(np.linalg.norm(lhs - rhs)) / (float(np.linalg.norm(lhs)) + float(np.linalg.norm(rhs)) + EPS)
            vals.append(rr)
            rows.append({'left': names[i], 'right': names[j], 'anti_automorphism_relative_residual': rr})
    rows.sort(key=lambda r: r['anti_automorphism_relative_residual'], reverse=True)
    return (float(max(vals)) if vals else 0.0, float(np.mean(vals)) if vals else 0.0, rows[:topn])


def matrix_summary_rows(names: List[str], mats: List[np.ndarray], R: np.ndarray) -> List[dict]:
    rows = []
    I = np.eye(mats[0].shape[0]) if mats else np.zeros((0, 0))
    for name, A in zip(names, mats):
        rows.append({
            'operator': name,
            'frobenius_norm_diagnostic': float(np.linalg.norm(A)),
            'rank': int(np.linalg.matrix_rank(A, tol=1e-10)) if A.size else 0,
            'idempotent_residual': float(np.linalg.norm(A @ A - A)) / (float(np.linalg.norm(A @ A)) + float(np.linalg.norm(A)) + EPS) if A.size else 0.0,
            'involutive_residual': float(np.linalg.norm(A @ A - I)) / (float(np.linalg.norm(A @ A)) + float(np.linalg.norm(I)) + EPS) if A.size else 0.0,
            'sharp_self_residual': float(np.linalg.norm(sharp(A, R) - A)) / (float(np.linalg.norm(A)) + EPS) if A.size else 0.0,
        })
    return rows



def vector_basis(vectors: List[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    good = [np.asarray(v, dtype=float).reshape(-1) for v in vectors if np.asarray(v, dtype=float).size and float(np.linalg.norm(v)) > tol]
    if not good:
        return np.zeros((0, 0), dtype=float)
    V = np.column_stack(good)
    U, sig, _ = np.linalg.svd(V, full_matrices=False)
    if len(sig) == 0:
        return np.zeros((V.shape[0], 0), dtype=float)
    keep = sig > max(tol, tol * float(sig[0]))
    return U[:, keep]


def build_carrier_basis(K: prev.GrowthComplex, faces: List[Face], source: str, R: np.ndarray, k2: np.ndarray, closed: np.ndarray, harmonic: np.ndarray, exact: np.ndarray, include_exact: bool, tol: float) -> np.ndarray:
    vectors: List[np.ndarray] = [k2, closed, harmonic, R @ k2, R @ closed, R @ harmonic]
    if include_exact:
        vectors.extend([exact, R @ exact])
    for rec in K.cap_records:
        q = cap_boundary_vector(K, faces, rec, source)
        eb = selector_vector(faces, tuple(rec['base_face']))
        ep = selector_vector(faces, tuple(rec['partner_face']))
        vectors.extend([q, R @ q, eb, R @ eb, ep, R @ ep])
    # Close once more under R after first basis extraction.  Since R^2=I this is enough for the finite pair-reversal carrier.
    U0 = vector_basis(vectors, tol)
    if U0.size == 0:
        return U0
    vectors2 = [U0[:, i] for i in range(U0.shape[1])] + [R @ U0[:, i] for i in range(U0.shape[1])]
    return vector_basis(vectors2, tol)


def compress_operator(U: np.ndarray, A: np.ndarray) -> np.ndarray:
    if U.size == 0 or U.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    return U.T @ A @ U


def run_case(case: dict, args: argparse.Namespace, out: Path) -> dict:
    vout = out / case['variant']
    vout.mkdir(parents=True, exist_ok=True)
    K = prev.build_growth(args.max_level, case['strict_sym'], case['use_backreaction'], case['pairings'])
    topo = prev.topology(K)
    cd = prev.chain_data(K)
    faces = cd['F']
    k2, scalar, pair_rows = prev.cochain_K2(K, case['source'], case['mode'], case['strict_sym'])
    dec = prev.decompose_2cochain(K, k2)
    h = dec['harmonic_vector']
    c = dec['closed_vector']
    e = dec['exact_vector']

    n = len(faces)
    I = np.eye(n)
    R_signed = pair_reversal_matrix(K, faces, signed=True)
    R_unsigned = pair_reversal_matrix(K, faces, signed=False)
    # Pair-reversal is the only primary transport datum for the candidate sharp map.
    R = R_signed if args.signed_pair_reversal else R_unsigned

    cap_ops = cap_operators(K, faces, case['source'])
    raw_generators: List[Tuple[str, np.ndarray]] = [
        ('I_carrier', I),
        ('R_pair_reversal', R),
        ('M_K2_closed', diag_operator(c)),
        ('M_K2_harmonic', diag_operator(h)),
    ]
    if args.include_exact_operator:
        raw_generators.append(('M_K2_exact', diag_operator(e)))
    raw_generators.extend(cap_ops)

    U_carrier = build_carrier_basis(K, faces, case['source'], R, k2, c, h, e, args.include_exact_operator, args.tol)
    carrier_dim = int(U_carrier.shape[1]) if U_carrier.ndim == 2 else 0
    if carrier_dim == 0:
        generators = []
        names = []
        mats = []
        R_comp = np.zeros((0, 0), dtype=float)
        I_comp = np.zeros((0, 0), dtype=float)
    else:
        R_comp = compress_operator(U_carrier, R)
        I_comp = np.eye(carrier_dim)
        generators: List[Tuple[str, np.ndarray]] = []
        for name, Araw in raw_generators:
            A = compress_operator(U_carrier, Araw)
            if name == 'I_carrier':
                A = I_comp
            if name == 'R_pair_reversal' and float(np.linalg.norm(A - I_comp)) <= args.zero_threshold:
                continue
            if name == 'I_carrier' or float(np.linalg.norm(A)) > args.zero_threshold:
                generators.append((name, A))
        names = [x[0] for x in generators]
        mats = [x[1] for x in generators]
    R = R_comp

    Q_initial = orthonormal_basis_from_mats(mats, args.tol)
    initial_dim = int(Q_initial.shape[1])
    max_prod, mean_prod, prod_rows = product_residuals(names, mats, Q_initial, args.keep_top_products)
    max_sharp_init, mean_sharp_init, sharp_init_rows = sharp_residuals(names, mats, Q_initial, R, args.keep_top_products)
    anti_max, anti_mean, anti_rows = anti_automorphism_residual(names, mats, R, args.keep_top_products)

    algebra_basis, closure_rows, capped = closure_under_products(generators, args.tol, args.max_closure_iter, args.max_algebra_dim)
    Q_alg = orthonormal_basis_from_mats(algebra_basis, args.tol)
    algebra_dim = int(Q_alg.shape[1])
    # Check closure of the generated basis under multiplication by original generators.
    post_abs_vals = []
    post_rel_vals = []
    post_rel_vals_material = []
    for A in algebra_basis:
        for _gname, G in generators:
            for P in [A @ G, G @ A]:
                absr = residual_to_basis(Q_alg, P)
                pn = float(np.linalg.norm(P))
                relr = absr / (pn + EPS)
                post_abs_vals.append(absr)
                post_rel_vals.append(relr)
                if pn > args.product_norm_floor:
                    post_rel_vals_material.append(relr)
    post_closure_max_abs = float(max(post_abs_vals)) if post_abs_vals else 0.0
    post_closure_max_rel = float(max(post_rel_vals)) if post_rel_vals else 0.0
    post_closure_max_rel_material = float(max(post_rel_vals_material)) if post_rel_vals_material else 0.0
    # Check whether sharp keeps the generated algebra stable.
    alg_names = [f'B{i}' for i in range(len(algebra_basis))]
    max_sharp_alg, mean_sharp_alg, sharp_alg_rows = sharp_residuals(alg_names, algebra_basis, Q_alg, R, args.keep_top_products)

    r_inv = float(np.linalg.norm(R @ R - I_comp)) / (float(np.linalg.norm(I_comp)) + EPS) if carrier_dim else 0.0
    r_sym = float(np.linalg.norm(R.T - R)) / (float(np.linalg.norm(R)) + EPS) if carrier_dim else 0.0
    r_changes = int(np.sum(np.abs(R - I_comp) > args.zero_threshold)) if carrier_dim else 0

    write_csv(vout / 'operator_generator_summary.csv', matrix_summary_rows(names, mats, R))
    write_csv(vout / 'initial_product_residuals_top.csv', prod_rows)
    write_csv(vout / 'initial_sharp_residuals_top.csv', sharp_init_rows)
    write_csv(vout / 'anti_automorphism_residuals_top.csv', anti_rows)
    write_csv(vout / 'algebra_closure_growth_log.csv', closure_rows)
    write_csv(vout / 'algebra_sharp_residuals_top.csv', sharp_alg_rows)
    write_csv(vout / 'pairing_cap_log.csv', K.pairing_log)
    write_csv(vout / 'pairing_2form_rows.csv', pair_rows)

    summary = {
        'variant': case['variant'],
        'model_label': MODEL_LABEL,
        'anti_smuggling_note': ANTI_SMUGGLING_NOTE,
        'max_level': args.max_level,
        'source': case['source'],
        'mode': case['mode'],
        'strict_symmetrized': case['strict_sym'],
        'use_backreaction': case['use_backreaction'],
        'pairings_requested': case['pairings'],
        'topology': topo,
        'chain_dimensions': {'C2': n, 'C3': len(cd['T']), 'carrier_dim': carrier_dim},
        'applied_pair_count': sum(1 for x in K.pairing_log if x.get('applied')),
        'decision_used_delta_beta_any': any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows),
        'K2_sector': {
            'total_norm_diagnostic': float(dec['total_norm']),
            'closed_ratio': float(dec['closed_ratio']),
            'exact_ratio': float(dec['exact_ratio']),
            'harmonic_ratio': float(dec['harmonic_ratio']),
            'defect_ratio': float(dec['defect_ratio']),
            'harmonic_dim_real': int(dec['harmonic_dim_real']),
            'closed_dim': int(dec['closed_dim']),
            'exact_dim': int(dec['exact_dim']),
        },
        'pair_reversal_candidate': {
            'signed_pair_reversal': bool(args.signed_pair_reversal),
            'R_squared_identity_residual': r_inv,
            'R_symmetric_residual': r_sym,
            'R_nonidentity_entry_count': r_changes,
            'is_nontrivial': r_changes > 0,
        },
        'operator_family': {
            'generator_names': names,
            'generator_count': len(names),
            'initial_span_dim': initial_dim,
            'initial_product_max_residual_to_initial_span': max_prod,
            'initial_product_mean_residual_to_initial_span': mean_prod,
            'initial_span_product_closed': max_prod <= args.closure_threshold,
            'initial_sharp_max_residual_to_initial_span': max_sharp_init,
            'initial_sharp_mean_residual_to_initial_span': mean_sharp_init,
            'initial_span_sharp_closed': max_sharp_init <= args.closure_threshold,
            'generated_algebra_dim': algebra_dim,
            'generated_algebra_capped': capped,
            'generated_algebra_post_closure_max_abs_residual': post_closure_max_abs,
            'generated_algebra_post_closure_max_relative_residual': post_closure_max_rel,
            'generated_algebra_post_closure_max_relative_residual_material_products': post_closure_max_rel_material,
            'generated_algebra_product_closed': (post_closure_max_abs <= args.closure_threshold and not capped),
            'generated_algebra_sharp_max_residual': max_sharp_alg,
            'generated_algebra_sharp_mean_residual': mean_sharp_alg,
            'generated_algebra_sharp_closed': max_sharp_alg <= args.closure_threshold,
            'anti_automorphism_law_max_residual_on_generators': anti_max,
            'anti_automorphism_law_mean_residual_on_generators': anti_mean,
        },
        'interpretation_flags': {
            'beta2_positive': topo['beta2'] > 0,
            'harmonic_2sector_positive': dec['harmonic_ratio'] > args.positive_threshold,
            'pair_reversal_nontrivial': r_changes > 0,
            'initial_operator_span_already_closed': max_prod <= args.closure_threshold,
            'generated_algebra_closes_under_products': (post_closure_max_abs <= args.closure_threshold and not capped),
            'generated_algebra_stable_under_candidate_sharp': max_sharp_alg <= args.closure_threshold,
            'strict_sym_killed_nontrivial_sector': case['strict_sym'] and topo['beta2'] == 0 and carrier_dim == 0,
            'candidate_star_not_claimed': True,
            'decision_used_delta_beta_any': any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows),
        },
    }
    (vout / 'variant_real_operator_composition_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def flat_summary(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        topo = r['topology']; k = r['K2_sector']; op = r['operator_family']; pr = r['pair_reversal_candidate']
        out.append({
            'variant': r['variant'],
            'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'],
            'H2_dim': k['harmonic_dim_real'],
            'K_harmonic_ratio': k['harmonic_ratio'],
            'K_defect_ratio': k['defect_ratio'],
            'pairings': r['applied_pair_count'],
            'carrier_dim': r['chain_dimensions']['carrier_dim'],
            'R_nontrivial': pr['is_nontrivial'],
            'R2_residual': pr['R_squared_identity_residual'],
            'generators': op['generator_count'],
            'initial_span_dim': op['initial_span_dim'],
            'initial_product_max_residual': op['initial_product_max_residual_to_initial_span'],
            'initial_sharp_max_residual': op['initial_sharp_max_residual_to_initial_span'],
            'algebra_dim': op['generated_algebra_dim'],
            'algebra_capped': op['generated_algebra_capped'],
            'algebra_post_closure_max_abs_residual': op['generated_algebra_post_closure_max_abs_residual'],
            'algebra_post_closure_max_relative_residual_material': op['generated_algebra_post_closure_max_relative_residual_material_products'],
            'algebra_sharp_max_residual': op['generated_algebra_sharp_max_residual'],
            'anti_auto_max_residual': op['anti_automorphism_law_max_residual_on_generators'],
            'used_delta_beta_any': r['decision_used_delta_beta_any'],
        })
    return out


def make_docs(summary: dict) -> Tuple[str, str, str, str]:
    rows = summary['variant_rows']
    flat = flat_summary(rows)
    table_lines = [
        '| variant | beta | H2 | K harm | K defect | pairings | carrier | R nontriv | init dim | init prod resid | init sharp resid | alg dim | alg sharp resid | used Δβ? |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in flat:
        table_lines.append(
            f"| {r['variant']} | ({r['beta0']},{r['beta1']},{r['beta2']},{r['beta3']}) | {r['H2_dim']} | "
            f"{r['K_harmonic_ratio']:.6g} | {r['K_defect_ratio']:.6g} | {r['pairings']} | {r['carrier_dim']} | {r['R_nontrivial']} | "
            f"{r['initial_span_dim']} | {r['initial_product_max_residual']:.6g} | {r['initial_sharp_max_residual']:.6g} | "
            f"{r['algebra_dim']} | {r['algebra_sharp_max_residual']:.6g} | {r['used_delta_beta_any']} |"
        )
    table = '\n'.join(table_lines)
    summary_md = f"""# SUMMARY — real operator composition from closed 2-sector gate

## Model label

{MODEL_LABEL}

## Anti-smuggling constraint

{ANTI_SMUGGLING_NOTE}

## Gate question

The previous package found a real 2-cochain `K ∈ C²` with a strong closed component, a nonzero harmonic residual, and a controlled tetrahedral defect `δK ∈ C³`.  This package asks the next, narrower question:

```text
Can the closed/harmonic real 2-sector plus cap/pair transport logs generate a small real operator family that is stable under composition and under a candidate involution-like reversal?
```

The candidate sharp map is:

```text
A^# := R_pair A^T R_pair
```

where `R_pair` is the real cap/pair reversal map on face-cochains and `A^T` is coefficient-dualization in the finite face basis.  This is a diagnostic anti-involution, not a C*-adjoint and not a positivity/norm claim.

## Comparative table

{table}

## Conservative reading

- `real_growth` variants have β₂ = 2 and a nontrivial pair-reversal map.
- The initial hand-built generator span is generally not product-closed; this is a real obstruction to claiming an immediate small operator system.
- The finitely generated real algebra closes after adding products, but its dimension is larger than the initial seed span.  This is expected and should not be overread.
- Stability under `#` is a compatibility diagnostic only.  In the current L2 output the generated product algebra is not `#`-stable, so this is an obstruction rather than a positive `*` result.
"""
    results_md = f"""# RESULTS — real operator composition from closed 2-sector gate

## Comparative table

{table}

## What was tested

For each variant the script rebuilds the deterministic growing primal simplicial complex at L2, computes the real 2-cochain sector from the previous closure/defect gate, and constructs a finite real operator family on the carrier subspace of `C²` generated by `K`, its closed/harmonic components, cap-boundary vectors, pair-reversal images, and base/partner selectors.  This avoids mistaking the full ambient face space for the actually supported operator sector.

Generators are:

```text
I_C2
R_pair_reversal
M_K2_closed
M_K2_harmonic
optional cap-boundary rank-one maps from pairing/cap logs
```

The test then checks:

```text
1. Is the initial generator span closed under products?
2. Is the initial generator span stable under A ↦ A^#?
3. What dimension is needed after closing under products?
4. Is that generated algebra stable under A ↦ A^#?
5. Does strict_symmetrized_control kill the nontrivial sector?
```

## Interpretation

The important negative/constructive point is the initial-span result: a naive small operator list is not already composition-closed.  Closure requires adding products.  On the carrier subspace the generated product algebra closes numerically by absolute residual, but the resulting generated algebra is **not stable under the current candidate sharp map**.  Therefore the result supports a real generated product-algebra candidate and simultaneously localizes the next obstruction: the cap/pair reversal plus coefficient-dualization formula is not yet a derived real `*`-structure.

The candidate sharp law has near-zero anti-automorphism residual because `A^# = R A^T R` is formally an anti-involution whenever the real pair-reversal matrix is involutive.  The nontrivial test is whether the generated algebra is stable under `#`; here that test fails for the nontrivial real-growth variants.

## Next test

```text
test_candidate_sharp_closure_obstruction_gate.py
```

Goal: localize why the generated real product algebra is not stable under `#`.  The test should compare generator subfamilies `{{I,R,M_closed,M_harmonic}}`, cap maps only, pair maps only, signed/unsigned `R`, and then a forced `#`-closure variant to measure which extra generators are required.  This is the next methodical step before any real `*`-structure claim.

## Current status

This package does not derive:

```text
i, J, J²=-Id, complex scalar multiplication, C*-positivity, C*-norm, physical adjoint, Hodge star, orientation package.
```

It does provide:

```text
real C²-operator seed family;
composition-closure audit;
cap/pair reversal candidate for a later real involution audit;
strict_sym/no_back/record-live controls.
```
"""
    audit_md = f"""# SOURCE AUDIT — operator composition gate

## Inherited source chain

- The previous 2-form closure/3-form defect package defines the current carrier: `K ∈ C²`, `δK ∈ C³`, closed/exact/harmonic decomposition, strict_sym/no_back/record-live controls.
- This package imports that script and does not add external geometry, complex structure, Hodge star, positivity, or norm axioms.

## Why this is not a hidden `i -> -i` convention change

The forum point is explicitly respected here: saying that a sign is conventional is valid only if the whole attached structure package is transported coherently.  One must not replace `i` by `-i` while leaving branch cuts, argument conventions, upper/lower half-plane, Fourier signs, causal signs, positivity, and `*` unchanged.

This package avoids that problem by not introducing those structures at all.  It tests only real face-cochain operators and a cap/pair reversal diagnostic.

## Sharp candidate status

`A^# = R A^T R` is a candidate real anti-involution on finite coefficient matrices.  It is not yet a CNNA-derived `*` because:

```text
- no positive cone is derived;
- no norm is derived;
- no physical adjoint interpretation is derived;
- no complex scalar multiplication is derived;
- no Hodge identification C² ↔ dual(C²) is used as ontology.
```

The only legitimate claim is compatibility/incompatibility of the generated real operator family with this candidate reversal.
"""
    readme_md = """# Real operator composition from closed 2-sector gate

Run:

```bash
python3 test_real_operator_composition_from_closed_2sector_gate.py
```

Main output:

```text
real_operator_composition_closed_2sector_out_L2/RESULTS.md
real_operator_composition_closed_2sector_out_L2/SUMMARY.md
real_operator_composition_closed_2sector_out_L2/comparative_summary.json
real_operator_composition_closed_2sector_out_L2/comparative_real_operator_composition_summary.csv
```

The package is a CNNA deterministic growing primal simplicial complex test.  NGF/CQNM remains a comparison frame only.
"""
    return summary_md, results_md, audit_md, readme_md


def package(out: Path, zip_path: Path) -> None:
    files = [
        Path(__file__).name,
        'test_2form_closure_and_3form_defect_gate.py',
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
    ap.add_argument('--out', default='real_operator_composition_closed_2sector_out_L2')
    ap.add_argument('--zip', default='cnna_real_operator_composition_closed_2sector_gate_pkg_L2.zip')
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--closure-threshold', type=float, default=1e-8)
    ap.add_argument('--positive-threshold', type=float, default=1e-4)
    ap.add_argument('--zero-threshold', type=float, default=1e-12)
    ap.add_argument('--product-norm-floor', type=float, default=1e-8)
    ap.add_argument('--max-closure-iter', type=int, default=8)
    ap.add_argument('--max-algebra-dim', type=int, default=220)
    ap.add_argument('--keep-top-products', type=int, default=80)
    ap.add_argument('--unsigned-pair-reversal', action='store_true')
    ap.add_argument('--include-exact-operator', action='store_true')
    args = ap.parse_args()
    args.signed_pair_reversal = not args.unsigned_pair_reversal

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
    rows = [run_case(case, args, out) for case in cases]
    summary = {'args': vars(args), 'variant_rows': rows}
    (out / 'comparative_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(out / 'comparative_real_operator_composition_summary.csv', flat_summary(rows))
    smd, rmd, audit, readme = make_docs(summary)
    (out / 'SUMMARY.md').write_text(smd, encoding='utf-8')
    (out / 'RESULTS.md').write_text(rmd, encoding='utf-8')
    (out / 'SOURCE_AUDIT.md').write_text(audit, encoding='utf-8')
    (out / 'README.md').write_text(readme, encoding='utf-8')
    package(out, Path(args.zip))
    print(json.dumps({
        'zip': args.zip,
        'out': args.out,
        'summary': flat_summary(rows),
    }, indent=2))


if __name__ == '__main__':
    main()
