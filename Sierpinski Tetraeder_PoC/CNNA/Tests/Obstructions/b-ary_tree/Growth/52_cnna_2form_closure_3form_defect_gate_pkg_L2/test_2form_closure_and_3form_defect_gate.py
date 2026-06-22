#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import itertools
import json
import math
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
Vertex = int
Edge = Tuple[int, int]
Face = Tuple[int, int, int]
Tet = Tuple[int, int, int, int]


def sorted_tuple(xs: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in xs))


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


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0:
        return 0
    return int(np.sum(s > max(tol, tol * float(s[0]))))


def orthonormal_colspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if A.size == 0 or A.shape[0] == 0 or A.shape[1] == 0:
        return np.zeros((A.shape[0] if A.ndim == 2 else 0, 0), dtype=float)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0:
        return np.zeros((A.shape[0], 0), dtype=float)
    keep = s > max(tol, tol * float(s[0]))
    return U[:, keep]


def nullspace(A: np.ndarray, ncols: Optional[int] = None, tol: float = 1e-10) -> np.ndarray:
    if ncols is None:
        ncols = A.shape[1] if A.ndim == 2 else 0
    if ncols == 0:
        return np.zeros((0, 0), dtype=float)
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(ncols)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    if len(s) == 0:
        rank = 0
    else:
        rank = int(np.sum(s > max(tol, tol * float(s[0]))))
    return Vt[rank:].T.copy()


def project(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    if Q.size == 0 or Q.shape[1] == 0:
        return np.zeros_like(x)
    return Q @ (Q.T @ x)


@dataclass
class VertexData:
    birth_time: int
    birth_order: int
    level: int
    parent_face: Optional[Face]
    sibling_index: int
    ancestor_vertices: Tuple[int, ...]
    record: np.ndarray
    live: np.ndarray


@dataclass
class GrowthComplex:
    tetrahedra: set[Tet] = field(default_factory=set)
    extra_faces: set[Face] = field(default_factory=set)
    vertices: Dict[int, VertexData] = field(default_factory=dict)
    birth_log: List[dict] = field(default_factory=list)
    pairing_log: List[dict] = field(default_factory=list)
    cap_records: List[dict] = field(default_factory=list)
    next_vertex: int = 0

    def add_vertex(self, level: int, parent_face: Optional[Face], sibling_index: int, ancestor_vertices: Sequence[int], strict_sym: bool) -> int:
        v = self.next_vertex
        self.next_vertex += 1
        t = v
        if strict_sym:
            record = np.array([float(level), 1.0], dtype=float)
        else:
            parent_sum = float(sum(parent_face) if parent_face is not None else 0)
            # Two purely real birth-history channels.  They are not metric coordinates;
            # they encode deterministic sequential provenance asymmetry.
            record = np.array([
                1.0 + 0.73 * level + 0.113 * sibling_index + 0.017 * parent_sum,
                1.0 + 0.41 * level * level + 0.071 * (t + 1) + 0.029 * len(set(ancestor_vertices)),
            ], dtype=float)
        self.vertices[v] = VertexData(
            birth_time=t,
            birth_order=t,
            level=level,
            parent_face=parent_face,
            sibling_index=sibling_index,
            ancestor_vertices=tuple(int(x) for x in ancestor_vertices),
            record=record.copy(),
            live=record.copy(),
        )
        return v

    def add_tetrahedron(self, tet: Iterable[int]) -> None:
        self.tetrahedra.add(tuple(sorted_tuple(tet)))

    def add_face(self, face: Iterable[int]) -> None:
        self.extra_faces.add(tuple(sorted_tuple(face)))

    def faces(self) -> List[Face]:
        fs: set[Face] = set(self.extra_faces)
        for t in self.tetrahedra:
            for f in itertools.combinations(t, 3):
                fs.add(tuple(f))
        return sorted(fs)

    def edges(self) -> List[Edge]:
        es: set[Edge] = set()
        for f in self.faces():
            for e in itertools.combinations(f, 2):
                es.add(tuple(e))
        for t in self.tetrahedra:
            for e in itertools.combinations(t, 2):
                es.add(tuple(e))
        return sorted(es)

    def vertex_list(self) -> List[int]:
        vs = set(self.vertices.keys())
        for f in self.faces():
            vs.update(f)
        for t in self.tetrahedra:
            vs.update(t)
        return sorted(vs)

    def boundary_faces(self) -> List[Face]:
        count: Dict[Face, int] = {}
        for t in self.tetrahedra:
            for f in itertools.combinations(t, 3):
                count[tuple(f)] = count.get(tuple(f), 0) + 1
        return sorted([f for f, c in count.items() if c == 1])

    def apply_backreaction(self, newborn: int, parent_face: Face, strength: float) -> None:
        # Deterministic real live-layer update.  It changes ancestors/parent line only;
        # it is not a time law, Hilbert norm, positivity, or complex phase.
        ndata = self.vertices[newborn]
        affected = list(parent_face)
        for v in affected:
            vd = self.vertices[v]
            age_gap = max(1, ndata.birth_time - vd.birth_time)
            kernel = strength / float(age_gap + 1)
            direction = np.array([
                0.19 + 0.011 * ndata.sibling_index,
                0.13 + 0.007 * (vd.level + 1),
            ], dtype=float)
            vd.live = vd.live + kernel * direction


def oriented_boundary_faces(simplex: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
    s = tuple(simplex)
    out = []
    for i in range(len(s)):
        face = s[:i] + s[i+1:]
        # canonical simplex order is sorted; (-1)^i gives the usual incidence sign.
        out.append((tuple(face), -1 if (i % 2) else 1))
    return out


def boundary_matrix(lower: List[Tuple[int, ...]], upper: List[Tuple[int, ...]]) -> np.ndarray:
    row = {s: i for i, s in enumerate(lower)}
    B = np.zeros((len(lower), len(upper)), dtype=float)
    for j, simplex in enumerate(upper):
        for face, sign in oriented_boundary_faces(simplex):
            if face in row:
                B[row[face], j] += float(sign)
    return B


def chain_data(K: GrowthComplex) -> dict:
    V = [(v,) for v in K.vertex_list()]
    E = [tuple(e) for e in K.edges()]
    F = [tuple(f) for f in K.faces()]
    T = [tuple(t) for t in sorted(K.tetrahedra)]
    B1 = boundary_matrix(V, E)
    B2 = boundary_matrix(E, F)
    B3 = boundary_matrix(F, T)
    return {'V': V, 'E': E, 'F': F, 'T': T, 'B1': B1, 'B2': B2, 'B3': B3}


def topology(K: GrowthComplex) -> dict:
    cd = chain_data(K)
    n0, n1, n2, n3 = len(cd['V']), len(cd['E']), len(cd['F']), len(cd['T'])
    r1 = matrix_rank(cd['B1'])
    r2 = matrix_rank(cd['B2'])
    r3 = matrix_rank(cd['B3'])
    return {
        'n0': n0, 'n1': n1, 'n2': n2, 'n3': n3,
        'rank_boundary_1': r1,
        'rank_boundary_2': r2,
        'rank_boundary_3': r3,
        'beta0': n0 - r1,
        'beta1': n1 - r1 - r2,
        'beta2': n2 - r2 - r3,
        'beta3': n3 - r3,
    }


def response_channels(K: GrowthComplex, v: int, source: str) -> np.ndarray:
    vd = K.vertices[v]
    if source == 'record':
        return vd.record.copy()
    if source == 'live':
        return vd.live.copy()
    if source == 'full':
        return 0.5 * (vd.record + vd.live)
    raise ValueError(source)


def face_response_k(K: GrowthComplex, face: Face, source: str, strict_sym: bool) -> float:
    if strict_sym:
        return 0.0
    a, b, c = face
    ra = response_channels(K, a, source)
    rb = response_channels(K, b, source)
    rc = response_channels(K, c, source)
    # Real alternating 2-channel response determinant on the face.
    return float((rb[0] - ra[0]) * (rc[1] - ra[1]) - (rc[0] - ra[0]) * (rb[1] - ra[1]))


def cap_boundary_coefficients(base_face: Face, cap_vertex: int) -> Dict[Face, int]:
    ghost = tuple(sorted_tuple((*base_face, cap_vertex)))
    return {tuple(face): int(sign) for face, sign in oriented_boundary_faces(ghost)}


def cochain_K2(K: GrowthComplex, source: str, mode: str, strict_sym: bool) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    faces = K.faces()
    idx = {f: i for i, f in enumerate(faces)}
    k_response = np.zeros(len(faces), dtype=float)
    for i, f in enumerate(faces):
        k_response[i] = face_response_k(K, f, source, strict_sym)
    k_pair = np.zeros(len(faces), dtype=float)
    pair_rows: List[dict] = []
    for rec in K.cap_records:
        strength = float(rec['strength_' + source] if ('strength_' + source) in rec else rec['strength_full'])
        if mode == 'response_only':
            strength = 0.0
        coeffs = cap_boundary_coefficients(tuple(rec['base_face']), int(rec['cap_vertex']))
        local_norm2 = 0.0
        for face, coeff in coeffs.items():
            if face in idx:
                k_pair[idx[face]] += strength * float(coeff)
                local_norm2 += float(strength * coeff) ** 2
        pair_rows.append({
            'pair_index': rec['pair_index'],
            'event_t': rec['event_t'],
            'base_face': str(list(rec['base_face'])),
            'partner_face': str(list(rec['partner_face'])),
            'cap_vertex': rec['cap_vertex'],
            'source': source,
            'mode': mode,
            'strength': strength,
            'local_pair_k_norm': math.sqrt(local_norm2),
            'decision_used_delta_beta': False,
            'measured_delta_beta2': rec.get('measured_delta_beta2', ''),
        })
    if mode == 'pair_only':
        return k_pair, np.abs(k_pair), pair_rows
    if mode == 'response_only':
        return k_response, np.abs(k_response), pair_rows
    return k_response + k_pair, np.abs(k_response + k_pair), pair_rows


def decompose_2cochain(K: GrowthComplex, vec: np.ndarray) -> dict:
    cd = chain_data(K)
    B2 = cd['B2']
    B3 = cd['B3']
    n2 = len(cd['F'])
    d1 = B2.T                  # C1 -> C2
    d2 = B3.T                  # C2 -> C3
    Q_exact = orthonormal_colspace(d1)
    Q_closed = nullspace(d2, ncols=n2)
    exact = project(Q_exact, vec)
    closed = project(Q_closed, vec)
    stacked = np.vstack([d2, Q_exact.T]) if Q_exact.shape[1] else d2.copy()
    Q_harm = nullspace(stacked, ncols=n2)
    harmonic = project(Q_harm, vec)
    defect = d2 @ vec if d2.size else np.zeros(0, dtype=float)
    total_norm = float(np.linalg.norm(vec))
    closed_norm = float(np.linalg.norm(closed))
    exact_norm = float(np.linalg.norm(exact))
    harmonic_norm = float(np.linalg.norm(harmonic))
    defect_norm = float(np.linalg.norm(defect))
    exact_in_closed = project(Q_exact, closed)
    harmonic_in_closed = project(Q_harm, closed)
    closed_residual = closed - exact_in_closed - harmonic_in_closed
    return {
        'total_norm': total_norm,
        'closed_norm': closed_norm,
        'exact_norm': exact_norm,
        'harmonic_norm': harmonic_norm,
        'defect_norm': defect_norm,
        'closed_ratio': closed_norm / (total_norm + EPS),
        'exact_ratio': exact_norm / (total_norm + EPS),
        'harmonic_ratio': harmonic_norm / (total_norm + EPS),
        'defect_ratio': defect_norm / (total_norm + EPS),
        'closed_residual_norm': float(np.linalg.norm(closed_residual)),
        'harmonic_dim_real': int(Q_harm.shape[1]),
        'closed_dim': int(Q_closed.shape[1]),
        'exact_dim': int(Q_exact.shape[1]),
        'deltaK_values': defect,
        'harmonic_vector': harmonic,
        'closed_vector': closed,
        'exact_vector': exact,
    }


def select_boundary_faces(K: GrowthComplex, count: int) -> List[Face]:
    bfaces = K.boundary_faces()
    scored = []
    for f in bfaces:
        births = [K.vertices[v].birth_order for v in f]
        levels = [K.vertices[v].level for v in f]
        # Purely provenance-based ranking, not beta/harmonic/topology-based.
        score = (max(births) - min(births)) + 0.37 * sum(levels) + 0.013 * sum(f)
        scored.append((score, f))
    scored.sort(reverse=True)
    chosen: List[Face] = []
    used: set[int] = set()
    for _, f in scored:
        # Prefer disjoint-ish carriers to avoid one local artifact.
        if len(set(f) & used) <= 1:
            chosen.append(f)
            used.update(f)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for _, f in scored:
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= count:
                break
    return chosen[:count]


def cap_strengths(K: GrowthComplex, face: Face, strict_sym: bool) -> Dict[str, float]:
    if strict_sym:
        return {'record': 0.0, 'live': 0.0, 'full': 0.0}
    vals = {}
    for source in ['record', 'live', 'full']:
        channels = np.array([response_channels(K, v, source) for v in face], dtype=float)
        spread = float(np.linalg.norm(np.max(channels, axis=0) - np.min(channels, axis=0)))
        alt = abs(face_response_k(K, face, source, strict_sym))
        vals[source] = 0.35 + 0.23 * spread + 0.17 * alt
    return vals


def add_hollow_cap(K: GrowthComplex, base_face: Face, partner_face: Face, pair_index: int, strict_sym: bool) -> dict:
    before = topology(K)
    max_level = max(K.vertices[v].level for v in base_face) + 1
    cap_v = K.add_vertex(max_level, base_face, pair_index, base_face, strict_sym)
    for f, _sign in oriented_boundary_faces(tuple(sorted_tuple((*base_face, cap_v)))):
        K.add_face(f)
    strengths = cap_strengths(K, base_face, strict_sym)
    after = topology(K)
    rec = {
        'pair_index': pair_index,
        'event_t': K.vertices[cap_v].birth_time,
        'base_face': tuple(base_face),
        'partner_face': tuple(partner_face),
        'cap_vertex': cap_v,
        'strength_record': strengths['record'],
        'strength_live': strengths['live'],
        'strength_full': strengths['full'],
        'decision_used_delta_beta': False,
        'measured_delta_beta2': after['beta2'] - before['beta2'],
    }
    K.cap_records.append(rec)
    K.pairing_log.append({
        'pair_index': pair_index,
        'event_t': rec['event_t'],
        'face_a': str(list(base_face)),
        'face_b': str(list(partner_face)),
        'cap_vertex': cap_v,
        'applied': True,
        'decision_used_delta_beta': False,
        'measured_delta_beta2': rec['measured_delta_beta2'],
        'strength_record': strengths['record'],
        'strength_live': strengths['live'],
        'strength_full': strengths['full'],
    })
    return rec


def build_growth(max_level: int, strict_sym: bool, use_backreaction: bool, pairings: int) -> GrowthComplex:
    K = GrowthComplex()
    # Seed tetrahedron: geometric carrier, not a spatial ontology.  It is a primal 3-simplex.
    seed = [K.add_vertex(level=0, parent_face=None, sibling_index=i, ancestor_vertices=(), strict_sym=strict_sym) for i in range(4)]
    K.add_tetrahedron(seed)
    sibling_counter = 0
    for level in range(1, max_level + 1):
        frontier = K.boundary_faces()
        # Freeze the frontier per level; sequential births still update live responses.
        for f in frontier:
            sibling_counter += 1
            new_v = K.add_vertex(level=level, parent_face=f, sibling_index=sibling_counter, ancestor_vertices=f, strict_sym=strict_sym)
            K.add_tetrahedron((*f, new_v))
            if use_backreaction:
                K.apply_backreaction(new_v, f, strength=0.31 / float(level + 1))
            K.birth_log.append({
                'event_t': K.vertices[new_v].birth_time,
                'level': level,
                'new_vertex': new_v,
                'parent_face': str(list(f)),
                'sibling_index': sibling_counter,
                'used_backreaction': use_backreaction,
                'strict_symmetrized': strict_sym,
            })
    if pairings > 0 and not strict_sym:
        chosen = select_boundary_faces(K, pairings)
        partners = list(reversed(chosen))
        for i, f in enumerate(chosen):
            add_hollow_cap(K, f, partners[i], i, strict_sym)
    return K


def top_faces_rows(K: GrowthComplex, vec: np.ndarray, harmonic: np.ndarray, closed: np.ndarray, scalar: np.ndarray, topn: int) -> List[dict]:
    faces = K.faces()
    rows = []
    for i, f in enumerate(faces):
        val = float(vec[i])
        h = float(harmonic[i])
        c = float(closed[i])
        s = float(scalar[i])
        if abs(val) <= 1e-14 and abs(h) <= 1e-14 and abs(c) <= 1e-14:
            continue
        rows.append({
            'face': str(list(f)),
            'K2_value': val,
            'abs_K2_scalar': s,
            'closed_component': c,
            'harmonic_component': h,
            'birth_orders': str([K.vertices[v].birth_order for v in f]),
            'levels': str([K.vertices[v].level for v in f]),
        })
    rows.sort(key=lambda r: (abs(r['harmonic_component']), abs(r['K2_value'])), reverse=True)
    return rows[:topn]


def defect_rows(K: GrowthComplex, delta: np.ndarray) -> List[dict]:
    tets = chain_data(K)['T']
    rows = []
    for i, t in enumerate(tets):
        v = float(delta[i]) if i < len(delta) else 0.0
        if abs(v) > 1e-14:
            rows.append({
                'tetrahedron': str(list(t)),
                'deltaK_value': v,
                'abs_deltaK': abs(v),
                'birth_orders': str([K.vertices[x].birth_order for x in t]),
                'levels': str([K.vertices[x].level for x in t]),
            })
    rows.sort(key=lambda r: r['abs_deltaK'], reverse=True)
    return rows


def orientation_gauge_check(K: GrowthComplex, vec: np.ndarray) -> dict:
    dec = decompose_2cochain(K, vec)
    flip = decompose_2cochain(K, -vec)
    return {
        'flip_closed_ratio_absdiff': abs(dec['closed_ratio'] - flip['closed_ratio']),
        'flip_harmonic_ratio_absdiff': abs(dec['harmonic_ratio'] - flip['harmonic_ratio']),
        'flip_defect_ratio_absdiff': abs(dec['defect_ratio'] - flip['defect_ratio']),
        'flip_total_norm_absdiff': abs(dec['total_norm'] - flip['total_norm']),
    }


def relabel_complex(K: GrowthComplex) -> GrowthComplex:
    vs = K.vertex_list()
    # Deterministic non-monotone permutation, independent of geometry.
    perm_values = list(reversed(vs[::2])) + list(reversed(vs[1::2]))
    mapping = {v: perm_values[i] for i, v in enumerate(vs)}
    R = GrowthComplex()
    R.next_vertex = max(mapping.values()) + 1 if mapping else 0
    for old, vd in K.vertices.items():
        new = mapping[old]
        pf = tuple(sorted(mapping[x] for x in vd.parent_face)) if vd.parent_face is not None else None
        av = tuple(mapping[x] for x in vd.ancestor_vertices if x in mapping)
        R.vertices[new] = VertexData(
            birth_time=vd.birth_time,
            birth_order=vd.birth_order,
            level=vd.level,
            parent_face=pf,
            sibling_index=vd.sibling_index,
            ancestor_vertices=av,
            record=vd.record.copy(),
            live=vd.live.copy(),
        )
    R.tetrahedra = {tuple(sorted(mapping[x] for x in t)) for t in K.tetrahedra}
    R.extra_faces = {tuple(sorted(mapping[x] for x in f)) for f in K.extra_faces}
    R.birth_log = [dict(x) for x in K.birth_log]
    for rec in K.cap_records:
        nr = dict(rec)
        nr['base_face'] = tuple(sorted(mapping[x] for x in rec['base_face']))
        nr['partner_face'] = tuple(sorted(mapping[x] for x in rec['partner_face']))
        nr['cap_vertex'] = mapping[rec['cap_vertex']]
        R.cap_records.append(nr)
    R.pairing_log = [dict(x) for x in K.pairing_log]
    return R


def relabel_robustness_check(K: GrowthComplex, source: str, mode: str, strict_sym: bool) -> dict:
    vec, scalar, _ = cochain_K2(K, source, mode, strict_sym)
    d0 = decompose_2cochain(K, vec)
    topo0 = topology(K)
    R = relabel_complex(K)
    vec_r, scalar_r, _ = cochain_K2(R, source, mode, strict_sym)
    dr = decompose_2cochain(R, vec_r)
    topor = topology(R)
    return {
        'relabel_beta_match': [topo0[f'beta{i}'] for i in range(4)] == [topor[f'beta{i}'] for i in range(4)],
        'relabel_total_norm_rel_absdiff': abs(d0['total_norm'] - dr['total_norm']) / (abs(d0['total_norm']) + EPS),
        'relabel_closed_ratio_absdiff': abs(d0['closed_ratio'] - dr['closed_ratio']),
        'relabel_harmonic_ratio_absdiff': abs(d0['harmonic_ratio'] - dr['harmonic_ratio']),
        'relabel_defect_ratio_absdiff': abs(d0['defect_ratio'] - dr['defect_ratio']),
    }


def run_case(case: dict, args: argparse.Namespace, out: Path) -> dict:
    variant = case['variant']
    vout = out / variant
    vout.mkdir(parents=True, exist_ok=True)
    K = build_growth(
        max_level=args.max_level,
        strict_sym=case['strict_sym'],
        use_backreaction=case['use_backreaction'],
        pairings=case['pairings'],
    )
    topo = topology(K)
    vec, scalar, pair_rows = cochain_K2(K, case['source'], case['mode'], case['strict_sym'])
    dec = decompose_2cochain(K, vec)
    scalar_dec = decompose_2cochain(K, scalar)
    gauge = orientation_gauge_check(K, vec)
    relabel = relabel_robustness_check(K, case['source'], case['mode'], case['strict_sym']) if args.relabel_check else {}
    faces_top = top_faces_rows(K, vec, dec['harmonic_vector'], dec['closed_vector'], scalar, args.keep_top_faces)
    drows = defect_rows(K, dec['deltaK_values'])[:args.keep_top_defects]
    write_csv(vout / 'birth_geometry_log.csv', K.birth_log)
    write_csv(vout / 'pairing_cap_log.csv', K.pairing_log)
    write_csv(vout / 'pairing_2form_rows.csv', pair_rows)
    write_csv(vout / 'top_2form_faces.csv', faces_top)
    write_csv(vout / 'top_3form_defects.csv', drows)
    chain = chain_data(K)
    summary = {
        'variant': variant,
        'model_label': 'CNNA deterministic growing primal simplicial complex; provenance tree as birth-history; NGF/CQNM-like only as comparison; 2-form closure / 3-form defect diagnostic',
        'max_level': args.max_level,
        'source': case['source'],
        'mode': case['mode'],
        'strict_symmetrized': case['strict_sym'],
        'use_backreaction': case['use_backreaction'],
        'pairings_requested': case['pairings'],
        'topology': topo,
        'applied_pair_count': sum(1 for x in K.pairing_log if x.get('applied')),
        'decision_used_delta_beta_any': any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows),
        'measured_delta_beta2_sum': sum(int(x.get('measured_delta_beta2', 0) or 0) for x in K.pairing_log),
        'chain_dimensions': {'C0': len(chain['V']), 'C1': len(chain['E']), 'C2': len(chain['F']), 'C3': len(chain['T'])},
        'K2_decomposition': {k: v for k, v in dec.items() if not isinstance(v, np.ndarray)},
        'scalar_absK_decomposition': {k: v for k, v in scalar_dec.items() if not isinstance(v, np.ndarray)},
        'orientation_gauge_check': gauge,
        'relabel_robustness_check': relabel,
        'interpretation_flags': {
            'beta2_positive': topo['beta2'] > 0,
            'closed_ratio_positive': dec['closed_ratio'] > args.positive_threshold,
            'harmonic_ratio_positive': dec['harmonic_ratio'] > args.positive_threshold,
            'three_form_defect_large': dec['defect_ratio'] > args.large_defect_threshold,
            'strict_sym_killed': case['strict_sym'] and topo['beta2'] == 0 and dec['total_norm'] <= args.zero_threshold,
            'decision_used_delta_beta_any': any(str(x.get('decision_used_delta_beta', '')).lower() == 'true' for x in K.pairing_log + pair_rows),
            'label_orientation_robust_under_diagnostics': (
                gauge['flip_harmonic_ratio_absdiff'] < 1e-9 and
                (not relabel or (relabel.get('relabel_beta_match', False) and relabel.get('relabel_harmonic_ratio_absdiff', 1.0) < 1e-8))
            ),
        },
    }
    (vout / 'variant_2form_closure_3form_defect_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def comparative_rows(rows: List[dict]) -> List[dict]:
    flat = []
    for r in rows:
        topo = r['topology']
        d = r['K2_decomposition']
        sd = r['scalar_absK_decomposition']
        flat.append({
            'variant': r['variant'],
            'source': r['source'],
            'mode': r['mode'],
            'beta0': topo['beta0'],
            'beta1': topo['beta1'],
            'beta2': topo['beta2'],
            'beta3': topo['beta3'],
            'applied_pair_count': r['applied_pair_count'],
            'K_total_norm': d['total_norm'],
            'K_closed_ratio': d['closed_ratio'],
            'K_exact_ratio': d['exact_ratio'],
            'K_harmonic_ratio': d['harmonic_ratio'],
            'K_defect_ratio': d['defect_ratio'],
            'K_defect_norm': d['defect_norm'],
            'scalar_absK_harmonic_ratio': sd['harmonic_ratio'],
            'harmonic_dim_real': d['harmonic_dim_real'],
            'closed_dim': d['closed_dim'],
            'exact_dim': d['exact_dim'],
            'decision_used_delta_beta_any': r['decision_used_delta_beta_any'],
            'measured_delta_beta2_sum': r['measured_delta_beta2_sum'],
            'flip_harmonic_ratio_absdiff': r['orientation_gauge_check']['flip_harmonic_ratio_absdiff'],
            'relabel_harmonic_ratio_absdiff': r['relabel_robustness_check'].get('relabel_harmonic_ratio_absdiff', ''),
        })
    return flat


def make_docs(summary: dict) -> Tuple[str, str, str, str]:
    rows = summary['variant_rows']
    flat = comparative_rows(rows)
    table_lines = [
        '| variant | source | mode | beta | pairings | K closed | K exact | K harmonic | δK defect | H² dim | |K| harmonic | used Δβ? |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in flat:
        table_lines.append(
            f"| {r['variant']} | {r['source']} | {r['mode']} | ({r['beta0']},{r['beta1']},{r['beta2']},{r['beta3']}) | "
            f"{r['applied_pair_count']} | {r['K_closed_ratio']:.6g} | {r['K_exact_ratio']:.6g} | {r['K_harmonic_ratio']:.6g} | "
            f"{r['K_defect_ratio']:.6g} | {r['harmonic_dim_real']} | {r['scalar_absK_harmonic_ratio']:.6g} | {r['decision_used_delta_beta_any']} |"
        )
    table = '\n'.join(table_lines)
    smd = f"""# SUMMARY — 2-form closure and 3-form defect gate

## Model label

CNNA deterministic growing primal simplicial complex with provenance/birth-history bookkeeping.  The tree-like birth process is a provenance register, not space.  NGF/CQNM is only a comparison frame.  This package is not SG/ST geometry, not a finished CQNM model, and not a complex/J derivation.

## Gate

This test keeps the next object real:

```text
K ∈ C² on triangular faces
δK ∈ C³ on tetrahedra
```

It asks whether the pairing-carried real 2-cochain has a closed/harmonic carrier or whether it has a large local 3-form defect on filled tetrahedra.

## Comparative result

{table}

## Conservative reading

Positive β₂ and positive harmonic K² support mean only that the generated primal complex carries a real 2-cochain sector.  They do not derive `J`, `i`, a sign of `J`, a Hodge star, a norm, positivity, spin, or a complex structure.

The new structure-package warning is encoded as a test condition: no claim is made that a sign flip is a convention unless all dependent structure is transformed.  This package therefore does not use branch choices, upper/lower half-planes, Fourier sign conventions, positivity, or analytic square-root/logarithm data.
"""
    rmd = f"""# RESULTS — 2-form closure and 3-form defect gate

## Comparative table

{table}

## Interpretation by gate

```text
strict_symmetrized_control:
  should kill β₂ and K-support.

real_growth_*:
  checks whether nonlinear asymmetry-gated cap/pair carriers produce β₂ and H² support.

no_backreaction_record:
  tests whether the carrier depends on live backreaction or already follows from sequential provenance asymmetry.

δK defect:
  if large, K is not a clean closed 2-form; the obstruction lives in the tetrahedral 3-form defect.
```

## Anti-smuggling checks

- `decision_used_delta_beta_any` must remain false.
- Orientation is treated only as a finite cochain gauge needed to compute incidence matrices.
- Sign-flip diagnostics compare scalar ratios under `K -> -K`.
- Relabel diagnostics compare the same scalar ratios after a deterministic vertex relabeling.
- No `i`, `J`, Hodge star, positivity, norm-as-axiom, branch cut, logarithm, square root, spin, or Fourier convention is used as input.

## Files

- `comparative_2form_closure_3form_defect_summary.csv`
- `comparative_summary.json`
- per-variant `variant_2form_closure_3form_defect_summary.json`
- per-variant `top_2form_faces.csv`
- per-variant `top_3form_defects.csv`
- per-variant `pairing_cap_log.csv`

## Next test

`test_real_operator_composition_from_closed_2sector_gate.py`: build the smallest real operator family from the closed/harmonic 2-sector and pair/cap transport records, then test closure under composition and an involution candidate without importing `*`, positivity, or a C*-norm.
"""
    audit = """# SOURCE AND METHODOLOGY AUDIT

## Inherited positive path

- Sequential provenance growth creates sibling/order asymmetry.
- Nonlinear asymmetry-gated complement/cap pairing can open β₂.
- Strict symmetrization is the required kill-control.
- `decision_used_delta_beta` must stay false: β₂ may be measured only after the move.

## What this package deliberately does not do

- It does not derive `J`.
- It does not select `J` versus `-J`.
- It does not derive complex scalar multiplication.
- It does not define a Hodge star.
- It does not define positivity or a C*-norm.
- It does not use branch choices for log/sqrt, upper/lower half-planes, Fourier signs, or analytic continuation conventions.

## Orientation note

Simplicial cochain computations require signed incidence matrices.  Here the orientation is only a computational gauge from sorted vertex order.  The interpreted metrics are scalar ratios and are checked under sign flip and deterministic relabeling.  A future failure of these checks would be an obstruction, not a convention.
"""
    readme = """# CNNA 2-form closure / 3-form defect gate

Run:

```bash
python3 test_2form_closure_and_3form_defect_gate.py
```

Default output:

```text
2form_closure_3form_defect_out_L2/
cnna_2form_closure_3form_defect_gate_pkg_L2.zip
```

Recommended quick rerun:

```bash
python3 test_2form_closure_and_3form_defect_gate.py \
  --max-level 2 \
  --out 2form_closure_3form_defect_out_L2 \
  --zip cnna_2form_closure_3form_defect_gate_pkg_L2.zip
```

Model class: CNNA deterministic growing primal simplicial complex with provenance/birth-history bookkeeping.  It is not a SG/ST global geometry and not a finished CQNM model.
"""
    return smd, rmd, audit, readme


def package(out: Path, zip_path: Path) -> None:
    files = [Path(__file__).name]
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
    ap.add_argument('--out', default='2form_closure_3form_defect_out_L2')
    ap.add_argument('--zip', default='cnna_2form_closure_3form_defect_gate_pkg_L2.zip')
    ap.add_argument('--keep-top-faces', type=int, default=80)
    ap.add_argument('--keep-top-defects', type=int, default=80)
    ap.add_argument('--positive-threshold', type=float, default=1e-4)
    ap.add_argument('--large-defect-threshold', type=float, default=0.5)
    ap.add_argument('--zero-threshold', type=float, default=1e-10)
    ap.add_argument('--relabel-check', action='store_true', default=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cases = [
        {'variant': 'real_growth_live_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'live', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_record_only_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'record', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_record_plus_live_pair_plus_response', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'full', 'mode': 'pair_plus_response'},
        {'variant': 'real_growth_live_pair_only', 'strict_sym': False, 'use_backreaction': True, 'pairings': 2, 'source': 'live', 'mode': 'pair_only'},
        {'variant': 'strict_symmetrized_control', 'strict_sym': True, 'use_backreaction': False, 'pairings': 0, 'source': 'record', 'mode': 'pair_plus_response'},
        {'variant': 'no_backreaction_record_pair_plus_response', 'strict_sym': False, 'use_backreaction': False, 'pairings': 2, 'source': 'record', 'mode': 'pair_plus_response'},
    ]
    rows = [run_case(c, args, out) for c in cases]
    summary = {'args': vars(args), 'variant_rows': rows}
    (out / 'comparative_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(out / 'comparative_2form_closure_3form_defect_summary.csv', comparative_rows(rows))
    smd, rmd, audit, readme = make_docs(summary)
    (out / 'SUMMARY.md').write_text(smd, encoding='utf-8')
    (out / 'RESULTS.md').write_text(rmd, encoding='utf-8')
    (out / 'SOURCE_AUDIT.md').write_text(audit, encoding='utf-8')
    (out / 'README.md').write_text(readme, encoding='utf-8')
    package(out, Path(args.zip))
    print(json.dumps({
        'zip': args.zip,
        'out': args.out,
        'summary': comparative_rows(rows),
    }, indent=2))


if __name__ == '__main__':
    main()
