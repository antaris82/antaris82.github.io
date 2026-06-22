#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

EPS = 1e-12
Vertex = int
Edge = Tuple[int, int]
Face = Tuple[int, int, int]
Tet = Tuple[int, int, int, int]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def frame_from_radial(radial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = unit(radial)
    seed = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(seed, r))) > 0.92:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = unit(np.cross(r, seed))
    e2 = unit(np.cross(r, e1))
    return r, e1, e2


def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def skew(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - M.T)


def axial(S: np.ndarray) -> np.ndarray:
    return np.array([
        0.5 * (S[2, 1] - S[1, 2]),
        0.5 * (S[0, 2] - S[2, 0]),
        0.5 * (S[1, 0] - S[0, 1]),
    ], dtype=float)


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


def nullspace(A: np.ndarray, ncols: Optional[int] = None, tol: float = 1e-10) -> np.ndarray:
    if ncols is None:
        ncols = A.shape[1] if A.ndim == 2 else 0
    if ncols == 0:
        return np.zeros((0, 0), dtype=float)
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(ncols, dtype=float)
    _U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > max(tol, tol * float(s[0])))) if len(s) else 0
    return Vt[rank:].T.copy()


def colspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros((A.shape[0] if A.ndim == 2 else 0, 0), dtype=float)
    U, s, _Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > max(tol, tol * float(s[0])) if len(s) else []
    return U[:, keep]


def project(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    if Q.size == 0 or Q.shape[1] == 0:
        return np.zeros_like(x)
    return Q @ (Q.T @ x)


def sorted_tuple(xs: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in xs))


def oriented_boundary_faces(simplex: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
    s = tuple(simplex)
    out = []
    for i in range(len(s)):
        face = s[:i] + s[i + 1:]
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


@dataclass
class Node:
    id: int
    parent: Optional[int]
    level: int
    birth_order: int
    birth_time: int
    birth_g: float
    g: float
    pos: np.ndarray
    radial: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    children: List[int] = field(default_factory=list)


class GrowthModel:
    def __init__(self, growth_rule: str, response_mode: str, branching: int = 3):
        if branching != 3:
            raise ValueError('this test is ternary because the challenged code uses a Z3 birth fan')
        if growth_rule not in {'real_growth', 'strict_symmetrized_control', 'no_backreaction'}:
            raise ValueError(growth_rule)
        self.growth_rule = growth_rule
        self.response_mode = response_mode
        self.branching = branching
        self.base = 1.0
        self.alpha_env = 0.22
        self.ancestor_decay = 0.55
        self.br_ancestor = 0.0 if growth_rule == 'no_backreaction' else 0.045
        self.br_sibling = 0.0 if growth_rule == 'no_backreaction' else 0.035
        self.transverse_amp = 0.42
        self.radial_step = 1.0
        self.nodes: Dict[int, Node] = {}
        self.directed_edges: Dict[Tuple[int, int], float] = defaultdict(float)
        self.birth_events: List[dict] = []
        self.t = 0
        self.next_id = 0
        r, e1, e2 = frame_from_radial(np.array([0.0, 0.0, 1.0]))
        root = self._new_node(None, 0, 0, 1.0, np.zeros(3), r, e1, e2)
        self.root = root.id

    def _new_node(self, parent: Optional[int], level: int, order: int, birth_g: float, pos: np.ndarray, radial: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> Node:
        n = Node(self.next_id, parent, level, order, self.t, birth_g, birth_g, pos, radial, e1, e2)
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def parent_line(self, parent: int) -> List[int]:
        out = []
        cur: Optional[int] = parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def address_tuple(self, node: int) -> Tuple[int, ...]:
        out: List[int] = []
        cur: Optional[int] = node
        while cur is not None and self.nodes[cur].parent is not None:
            out.append(self.nodes[cur].birth_order)
            cur = self.nodes[cur].parent
        return tuple(reversed(out))

    def birth_environment_load(self, parent: int, older_siblings: List[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_decay ** (d - 1))
        if self.growth_rule != 'strict_symmetrized_control':
            for s in older_siblings:
                env += self.nodes[s].g
        return env

    def child_conductance_from_env(self, env_load: float) -> float:
        if self.response_mode == 'linear':
            return self.base + self.alpha_env * env_load
        if self.response_mode == 'log':
            return self.base + self.alpha_env * math.log1p(env_load)
        if self.response_mode == 'saturating':
            return self.base + self.alpha_env * (env_load / (1.0 + env_load))
        if self.response_mode == 'power_saturating':
            x = env_load / (1.0 + env_load)
            return self.base + self.alpha_env * (x ** 1.35)
        raise ValueError(self.response_mode)

    def child_position(self, parent: int, order: int, older_siblings: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.nodes[parent]
        effective_order = 2 if self.growth_rule == 'strict_symmetrized_control' else order
        theta = 2.0 * math.pi * (effective_order - 1) / 3.0
        twist = 0.37 * sum((i + 1) * x for i, x in enumerate(self.address_tuple(parent)))
        transverse = math.cos(theta + twist) * p.e1 + math.sin(theta + twist) * p.e2
        older_push = np.zeros(3)
        if self.growth_rule != 'strict_symmetrized_control':
            for s in older_siblings:
                older_push += unit(p.pos - self.nodes[s].pos)
        direction = unit(p.radial + self.transverse_amp * transverse + 0.08 * older_push)
        step = self.radial_step * (1.0 + (0.0 if self.growth_rule == 'strict_symmetrized_control' else 0.06 * (order - 2)))
        pos = p.pos + step * direction
        r, e1, e2 = frame_from_radial(pos if np.linalg.norm(pos) > EPS else direction)
        return pos, r, e1, e2

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env)
        pos, r, e1, e2 = self.child_position(parent, order, older)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g, pos, r, e1, e2)
        c = child.id
        total_env = env + EPS
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            self.directed_edges[(a, c)] += self.alpha_env * contrib / total_env * birth_g
        if self.growth_rule != 'strict_symmetrized_control':
            for s in older:
                contrib = self.nodes[s].g
                self.directed_edges[(s, c)] += self.alpha_env * contrib / total_env * birth_g
        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * birth_g / float(d * d)
            self.nodes[a].g += delta
            self.directed_edges[(c, a)] += delta
        if self.growth_rule != 'strict_symmetrized_control':
            for s in older:
                delta = self.br_sibling * birth_g
                self.nodes[s].g += delta
                self.directed_edges[(c, s)] += delta
        self.birth_events.append({
            't': self.t,
            'parent': parent,
            'child': c,
            'order': order,
            'older_siblings': list(older),
            'env_load': env,
            'birth_g': birth_g,
            'level': child.level,
            'growth_rule': self.growth_rule,
        })
        return c

    def grow(self, max_level: int) -> None:
        frontier = [self.root]
        for _ in range(max_level):
            nxt = []
            for p in frontier:
                for k in range(1, 4):
                    nxt.append(self.add_child(p, k))
            frontier = nxt

    def child_ids_ordered(self, parent: int) -> List[int]:
        return sorted(self.nodes[parent].children, key=lambda c: self.nodes[c].birth_order)


@dataclass
class SimplicialComplex:
    tets: set[Tet] = field(default_factory=set)
    extra_faces: set[Face] = field(default_factory=set)

    def add_tet(self, tet: Iterable[int]) -> None:
        t = tuple(sorted_tuple(tet))
        if len(t) == 4:
            self.tets.add(t)

    def add_face(self, face: Iterable[int]) -> None:
        f = tuple(sorted_tuple(face))
        if len(f) == 3:
            self.extra_faces.add(f)

    def faces(self) -> List[Face]:
        fs = set(self.extra_faces)
        for t in self.tets:
            for f in itertools.combinations(t, 3):
                fs.add(tuple(f))
        return sorted(fs)

    def edges(self) -> List[Edge]:
        es = set()
        for f in self.faces():
            for e in itertools.combinations(f, 2):
                es.add(tuple(e))
        return sorted(es)

    def vertices(self) -> List[int]:
        vs = set()
        for f in self.faces():
            vs.update(f)
        for t in self.tets:
            vs.update(t)
        return sorted(vs)

    def face_occupancy(self) -> Dict[Face, int]:
        occ: Dict[Face, int] = defaultdict(int)
        for t in self.tets:
            for f in itertools.combinations(t, 3):
                occ[tuple(f)] += 1
        return dict(occ)

    def boundary_faces(self) -> List[Face]:
        occ = self.face_occupancy()
        return sorted([f for f, c in occ.items() if c == 1])


def chain_data(K: SimplicialComplex) -> dict:
    V = [(v,) for v in K.vertices()]
    E = [tuple(e) for e in K.edges()]
    F = [tuple(f) for f in K.faces()]
    T = [tuple(t) for t in sorted(K.tets)]
    B1 = boundary_matrix(V, E)
    B2 = boundary_matrix(E, F)
    B3 = boundary_matrix(F, T)
    return {'V': V, 'E': E, 'F': F, 'T': T, 'B1': B1, 'B2': B2, 'B3': B3}


def topology(K: SimplicialComplex) -> dict:
    cd = chain_data(K)
    n0, n1, n2, n3 = len(cd['V']), len(cd['E']), len(cd['F']), len(cd['T'])
    r1, r2, r3 = matrix_rank(cd['B1']), matrix_rank(cd['B2']), matrix_rank(cd['B3'])
    return {
        'n0': n0, 'n1': n1, 'n2': n2, 'n3': n3,
        'rank_boundary_1': r1, 'rank_boundary_2': r2, 'rank_boundary_3': r3,
        'beta0': n0 - r1,
        'beta1': n1 - r1 - r2,
        'beta2': n2 - r2 - r3,
        'beta3': n3 - r3,
    }


def choose_boundary_face_for_parent(model: GrowthModel, K: SimplicialComplex, parent: int) -> Optional[Face]:
    candidates = [f for f in K.boundary_faces() if parent in f]
    if not candidates:
        return None
    pr = model.nodes[parent].radial
    def score(f: Face) -> Tuple[float, int]:
        centroid = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
        outward = unit(centroid - model.nodes[parent].pos)
        return (float(np.dot(outward, pr)), -sum(model.nodes[v].birth_time for v in f))
    return max(candidates, key=score)


def build_primal_complex(model: GrowthModel, pairings: int, strict_sym: bool) -> Tuple[SimplicialComplex, List[dict]]:
    K = SimplicialComplex()
    root_ch = model.child_ids_ordered(model.root)
    if len(root_ch) >= 3:
        K.add_tet((model.root, root_ch[0], root_ch[1], root_ch[2]))
    for ev in model.birth_events:
        child = int(ev['child'])
        parent = int(ev['parent'])
        if child in K.vertices():
            continue
        face = choose_boundary_face_for_parent(model, K, parent)
        if face is None:
            continue
        K.add_tet((*face, child))
    cap_log: List[dict] = []
    if pairings <= 0 or strict_sym:
        return K, cap_log
    scored = []
    for f in K.boundary_faces():
        births = [model.nodes[v].birth_order for v in f]
        levels = [model.nodes[v].level for v in f]
        score = (max(births) - min(births)) + 0.37 * sum(levels) + 0.013 * sum(f)
        scored.append((score, f))
    scored.sort(reverse=True)
    chosen = [f for _, f in scored[:pairings]]
    partners = list(reversed(chosen))
    for i, f in enumerate(chosen):
        cap_v = max(model.nodes) + 1
        while cap_v in model.nodes:
            cap_v += 1
        # Cap vertices are real geometric/provenance markers for a hollow 2-boundary;
        # they are not tetrahedra, so they can open beta2.  They receive no free phase.
        centroid = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
        radial = unit(centroid)
        r, e1, e2 = frame_from_radial(radial)
        birth_g = float(np.mean([model.nodes[v].birth_g for v in f]))
        model.nodes[cap_v] = Node(cap_v, None, max(model.nodes[v].level for v in f) + 1, 0, model.t + i + 1, birth_g, birth_g, centroid + 0.19 * radial, r, e1, e2)
        ghost = tuple(sorted_tuple((*f, cap_v)))
        for face, _s in oriented_boundary_faces(ghost):
            K.add_face(face)
        cap_log.append({
            'pair_index': i,
            'base_face': str(list(f)),
            'partner_face': str(list(partners[i])) if partners else '',
            'cap_vertex': cap_v,
            'decision_used_delta_beta': False,
            'measured_delta_beta2': '',
        })
    return K, cap_log


def harmonic_basis_faces(K: SimplicialComplex) -> Tuple[np.ndarray, np.ndarray]:
    cd = chain_data(K)
    B2, B3 = cd['B2'], cd['B3']
    n2 = len(cd['F'])
    if n2 == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    L2 = B2.T @ B2 + B3 @ B3.T
    vals, vecs = np.linalg.eigh(L2)
    mask = vals < 1e-9
    return vecs[:, mask], vals


def face_normal(model: GrowthModel, f: Face, mode: str) -> np.ndarray:
    pts = [model.nodes[v].pos for v in f]
    if mode == 'birth_order':
        order = sorted(f, key=lambda v: (model.nodes[v].birth_time, v))
        pts = [model.nodes[v].pos for v in order]
    a, b, c = pts
    n = np.cross(b - a, c - a)
    nn = float(np.linalg.norm(n))
    if nn < EPS:
        return np.zeros(3, dtype=float)
    n = n / nn
    centroid = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
    root = model.nodes[model.root].pos
    if float(np.dot(n, centroid - root)) < 0.0:
        n = -n
    return n


def face_area(model: GrowthModel, f: Face) -> float:
    a, b, c = [model.nodes[v].pos for v in f]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def vertex_operator(model: GrowthModel, node: int, source: str, op_mode: str, phase_sign: int, antisym_eta: float) -> np.ndarray:
    n = model.nodes[node]
    r, e1, e2 = n.radial, n.e1, n.e2
    if n.parent is None or n.birth_order == 0:
        order_phase = 0.0
    else:
        effective_order = 2 if model.growth_rule == 'strict_symmetrized_control' else n.birth_order
        order_phase = float(phase_sign) * 2.0 * math.pi * (effective_order - 1) / 3.0
    q = math.cos(order_phase) * e1 + math.sin(order_phase) * e2
    h = unit(0.7 * r + 0.3 * q)
    birth = n.birth_g
    live = n.g
    aging = max(0.0, live - birth)
    if source == 'record':
        a, b, c = birth, 0.22 * birth, 0.08 * birth
    elif source == 'live':
        a, b, c = live, 0.25 * birth + 0.55 * aging, 0.12 * live
    elif source == 'full':
        a, b, c = 0.5 * (birth + live), 0.235 * birth + 0.275 * aging, 0.1 * live
    else:
        raise ValueError(source)
    metric_part = a * np.outer(r, r) + b * np.outer(q, q) + c * np.outer(h, h) + 0.04 * birth * np.eye(3)
    transport_part = antisym_eta * (0.5 * b + c + 0.07 * birth) * (np.outer(q, h) - np.outer(h, q))
    if op_mode == 'legacy_sym_metric':
        return sym(metric_part)
    if op_mode == 'legacy_raw_metric_no_final_sym':
        return metric_part
    if op_mode == 'antisym_birth_transport':
        return metric_part + transport_part
    if op_mode == 'antisym_then_sym_control':
        return sym(metric_part + transport_part)
    raise ValueError(op_mode)


def face_K(model: GrowthModel, face: Face, source: str, op_mode: str, phase_sign: int, antisym_eta: float) -> np.ndarray:
    a, b, c = face
    Sa = vertex_operator(model, a, source, op_mode, phase_sign, antisym_eta)
    Sb = vertex_operator(model, b, source, op_mode, phase_sign, antisym_eta)
    Sc = vertex_operator(model, c, source, op_mode, phase_sign, antisym_eta)
    Aab = Sb - Sa
    Abc = Sc - Sb
    return skew(Aab @ Abc - Abc @ Aab)


def orientation_metrics(model: GrowthModel, K: SimplicialComplex, source: str, op_mode: str, phase_sign: int, antisym_eta: float) -> Tuple[dict, List[dict]]:
    faces = K.faces()
    topo = topology(K)
    if not faces:
        return {'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'], 'harmonic_dim_real': 0}, []
    W = np.array([axial(face_K(model, f, source, op_mode, phase_sign, antisym_eta)) for f in faces], dtype=float)
    total = float(np.linalg.norm(W)) + EPS
    H, vals = harmonic_basis_faces(K)
    Wh = H @ (H.T @ W) if H.size and H.shape[1] else np.zeros_like(W)
    hn = np.linalg.norm(Wh, axis=1)
    htotal = float(np.linalg.norm(Wh))
    mask = hn > max(1e-10, 1e-8 * (float(np.max(hn)) if len(hn) else 1.0))
    if np.any(mask):
        unit_vecs = Wh[mask] / (hn[mask][:, None] + EPS)
        coherence = float(np.linalg.norm(np.mean(unit_vecs, axis=0)))
    else:
        coherence = 0.0
    normals_out = np.array([face_normal(model, f, 'outward') for f in faces], dtype=float)
    normals_birth = np.array([face_normal(model, f, 'birth_order') for f in faces], dtype=float)
    areas = np.array([max(face_area(model, f), EPS) for f in faces], dtype=float)
    denom = float(np.sum(hn * areas)) + EPS
    dot_out = np.einsum('ij,ij->i', Wh, normals_out)
    dot_birth = np.einsum('ij,ij->i', Wh, normals_birth)
    signed_out = float(np.sum(dot_out * areas)) / denom
    abs_out = float(np.sum(np.abs(dot_out) * areas)) / denom
    signed_birth = float(np.sum(dot_birth * areas)) / denom
    abs_birth = float(np.sum(np.abs(dot_birth) * areas)) / denom
    rows = []
    for i, f in enumerate(faces):
        rows.append({
            'face': str(list(f)),
            'birth_orders': str([model.nodes[v].birth_order for v in f]),
            'levels': str([model.nodes[v].level for v in f]),
            'K_axial_norm': float(np.linalg.norm(W[i])),
            'H_axial_norm': float(hn[i]),
            'H_dot_birth_normal': float(dot_birth[i]),
            'H_dot_outward_normal': float(dot_out[i]),
            'area': float(areas[i]),
        })
    rows.sort(key=lambda r: (r['H_axial_norm'], r['K_axial_norm']), reverse=True)
    metrics = {
        'beta0': topo['beta0'], 'beta1': topo['beta1'], 'beta2': topo['beta2'], 'beta3': topo['beta3'],
        'n_faces': len(faces), 'n_tets': len(K.tets),
        'harmonic_dim_real': int(H.shape[1]) if H.ndim == 2 else 0,
        'K_axial_total_norm': total - EPS,
        'harmonic_axial_norm': htotal,
        'harmonic_axial_ratio': htotal / total,
        'orientation_coherence': coherence,
        'normal_flux_signed_ratio': signed_out,
        'normal_flux_abs_ratio': abs_out,
        'birth_normal_flux_signed_ratio': signed_birth,
        'birth_normal_flux_abs_ratio': abs_birth,
        'kappa_orientation_ratio': abs(signed_out) / (abs_out + EPS),
        'kappa_birth_orientation_ratio': abs(signed_birth) / (abs_birth + EPS),
        'support_fraction': float(np.mean(mask)) if len(mask) else 0.0,
        'laplacian_zero_eigs': int(H.shape[1]) if H.ndim == 2 else 0,
    }
    return metrics, rows


def run_case(case: dict, args: argparse.Namespace, out: Path) -> dict:
    variant = case['variant']
    vout = out / variant
    vout.mkdir(parents=True, exist_ok=True)
    model = GrowthModel(case['growth_rule'], args.response_mode)
    model.grow(args.max_level)
    K, cap_log = build_primal_complex(model, pairings=case['pairings'], strict_sym=case['growth_rule'] == 'strict_symmetrized_control')
    rows_by_mode = []
    summaries = []
    for op_mode in args.operator_modes:
        met, face_rows = orientation_metrics(model, K, case['source'], op_mode, +1, args.antisym_eta)
        met_k, _ = orientation_metrics(model, K, case['source'], op_mode, -1, args.antisym_eta)
        same_as_legacy_diff = None
        if op_mode != 'legacy_sym_metric':
            W0 = np.array([axial(face_K(model, f, case['source'], 'legacy_sym_metric', +1, args.antisym_eta)) for f in K.faces()], dtype=float)
            W1 = np.array([axial(face_K(model, f, case['source'], op_mode, +1, args.antisym_eta)) for f in K.faces()], dtype=float)
            same_as_legacy_diff = float(np.linalg.norm(W1 - W0) / (np.linalg.norm(W0) + EPS))
        row = {
            'variant': variant,
            'growth_rule': case['growth_rule'],
            'source': case['source'],
            'operator_mode': op_mode,
            **met,
            'kappa_mirror_harmonic_axial_ratio': met_k['harmonic_axial_ratio'],
            'kappa_mirror_birth_signed_ratio': met_k['birth_normal_flux_signed_ratio'],
            'kappa_flip_birth_signed_sum_abs': abs(met['birth_normal_flux_signed_ratio'] + met_k['birth_normal_flux_signed_ratio']),
            'kappa_flip_birth_signed_product': met['birth_normal_flux_signed_ratio'] * met_k['birth_normal_flux_signed_ratio'],
            'relative_axial_field_change_vs_legacy_sym_metric': same_as_legacy_diff if same_as_legacy_diff is not None else 0.0,
            'decision_used_delta_beta_any': False,
        }
        summaries.append(row)
        write_csv(vout / f'{op_mode}_top_harmonic_faces.csv', face_rows[:args.keep_top_faces])
        rows_by_mode.extend([{**r, 'operator_mode': op_mode} for r in face_rows[:args.keep_top_faces]])
    write_csv(vout / 'birth_events.csv', model.birth_events)
    write_csv(vout / 'hollow_cap_pairing_log.csv', cap_log)
    write_csv(vout / 'operator_mode_summary.csv', summaries)
    summary = {
        'variant': variant,
        'model_label': 'CNNA diagnostic regression: deterministic ternary provenance growth + growing primal simplicial complex; compares legacy symmetrized metric vertex operator against derived birth-order antisymmetric transport operator; NGF/CQNM only comparison, not input',
        'max_level': args.max_level,
        'source': case['source'],
        'topology': topology(K),
        'cap_pairings_applied': len(cap_log),
        'operator_mode_rows': summaries,
        'anti_smuggling_flags': {
            'uses_i_or_complex_scalar': False,
            'uses_hodge_star_or_adjoint_as_input': False,
            'uses_positivity_or_norm_as_selection_rule': False,
            'uses_delta_beta_in_pairing_decision': False,
            'antisymmetric_term_is_birth_order_derived': True,
            'strict_symmetrized_control_collapses_birth_phase': case['growth_rule'] == 'strict_symmetrized_control',
        },
    }
    (vout / 'variant_birth_phase_antisym_regression_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def make_docs(summary: dict) -> Tuple[str, str, str, str]:
    flat = []
    for v in summary['variant_rows']:
        flat.extend(v['operator_mode_rows'])
    table_lines = [
        '| variant | mode | beta | Hdim | K total | harmonic axial | kappa birth | mirror signed sum | Δ vs legacy | used Δβ? |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in flat:
        table_lines.append(
            f"| {r['variant']} | {r['operator_mode']} | ({r['beta0']},{r['beta1']},{r['beta2']},{r['beta3']}) | "
            f"{r['harmonic_dim_real']} | {r['K_axial_total_norm']:.6g} | {r['harmonic_axial_ratio']:.6g} | "
            f"{r['kappa_birth_orientation_ratio']:.6g} | {r['kappa_flip_birth_signed_sum_abs']:.6g} | "
            f"{r['relative_axial_field_change_vs_legacy_sym_metric']:.6g} | {r['decision_used_delta_beta_any']} |"
        )
    table = '\n'.join(table_lines)
    summary_md = f"""# SUMMARY — birth-phase antisymmetric operator regression gate

This package tests the implementation-level objection that the existing vertex operator encodes the ternary birth phase in a symmetric metric channel and then returns `sym(M)`.  The target is not a J derivation.  The target is a falsifiable operator regression:

```text
legacy_sym_metric
  M = metric terms using q⊗q and h⊗h, then sym(M)

legacy_raw_metric_no_final_sym
  same metric terms, no final sym(M)

antisym_birth_transport
  same metric terms plus a real birth-order-derived antisymmetric transport term
  η · s · (q⊗h - h⊗q)

antisym_then_sym_control
  antisymmetric term inserted but immediately symmetrized away
```

{table}

Conservative reading: a positive result is only meaningful if the antisymmetric birth-transport mode differs from legacy, while strict symmetrization remains zero or trivial.  The antisymmetric term is not a complex structure; it is a real directed transport diagnostic derived from birth order.
"""
    results_md = f"""# RESULTS — birth-phase antisymmetric operator regression gate

## Comparative table

{table}

## Gate logic

The test separates two possible failure modes:

```text
A. Removing final sym(M) alone changes nothing.
   Then q⊗q / h⊗h were already symmetric and phase-even.

B. A birth-order-derived antisymmetric term changes the axial harmonic sector.
   Then the old gate measured the wrong object for directed growth transport.
```

## Critical restrictions

- No `i`, no complex scalar, no J, no Hodge star, no adjoint, no positivity, no norm is used as generator input.
- Norms are output diagnostics only.
- The hollow cap/pairing rule does not use β or harmonic data in its decision.
- The κ-mirror is diagnostic: it reverses the real ternary birth fan phase sign and checks whether signed birth-normal coupling flips.

## Current interpretation

If `legacy_raw_metric_no_final_sym` is numerically identical to `legacy_sym_metric`, the final `sym(M)` is not the only problem: the phase was encoded through phase-even symmetric tensors.  If `antisym_then_sym_control` collapses back to legacy, final symmetrization is confirmed as a hard output kill for directed transport.
"""
    audit_md = """# SOURCE AUDIT — why this test exists

The regression is motivated by package 50's `vertex_operator` structure:

```text
order_phase -> q -> h
M = a rr + b qq + c hh + scalar I
return sym(M)
```

This makes the birth phase available, but encodes it in symmetric metric tensors and then forces a symmetric output.  That is a legitimate metric/DtN-response choice, but it is not a directed growth-transport operator.

This package therefore treats the old operator as the `legacy_sym_metric` branch and tests a separate real transport branch.  The transport branch is not declared ontic truth; it is a regression gate.  It is derived only in the narrow sense that its direction is computed from existing real birth-order fan data.  It is not a J, not a Hodge star, not a complex phase and not a C*-adjoint.

Next test if positive:

```text
test_directed_transport_operator_closure_gate.py
```

Goal: test whether the antisymmetric birth-transport operator family closes under composition on the H² carrier without saturating to the full matrix algebra and without importing an adjoint/positivity package.
"""
    readme_md = """# Birth-phase antisymmetric operator regression gate

Run:

```bash
python3 test_birth_phase_antisym_operator_regression_gate.py
```

Optional:

```bash
python3 test_birth_phase_antisym_operator_regression_gate.py --max-level 2 --antisym-eta 1.0
```

Outputs include JSON, CSV, SUMMARY.md, RESULTS.md, SOURCE_AUDIT.md and a ZIP package.
"""
    return summary_md, results_md, audit_md, readme_md


def write_comparative(out: Path, rows: List[dict]) -> None:
    flat = []
    for v in rows:
        for r in v['operator_mode_rows']:
            flat.append(r)
    write_csv(out / 'comparative_birth_phase_antisym_operator_summary.csv', flat)


def package(out: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).name, Path(__file__).name)
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.resolve().relative_to(Path.cwd()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-level', type=int, default=2)
    ap.add_argument('--response-mode', default='power_saturating', choices=['linear', 'log', 'saturating', 'power_saturating'])
    ap.add_argument('--source', default='live', choices=['record', 'live', 'full'])
    ap.add_argument('--pairings', type=int, default=2)
    ap.add_argument('--antisym-eta', type=float, default=1.0)
    ap.add_argument('--operator-modes', nargs='*', default=['legacy_sym_metric', 'legacy_raw_metric_no_final_sym', 'antisym_birth_transport', 'antisym_then_sym_control'])
    ap.add_argument('--keep-top-faces', type=int, default=80)
    ap.add_argument('--out', default='birth_phase_antisym_operator_regression_out_L2')
    ap.add_argument('--zip', default='cnna_birth_phase_antisym_operator_regression_gate_pkg_L2.zip')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cases = [
        {'variant': 'real_growth_live', 'growth_rule': 'real_growth', 'source': args.source, 'pairings': args.pairings},
        {'variant': 'strict_symmetrized_control', 'growth_rule': 'strict_symmetrized_control', 'source': args.source, 'pairings': 0},
        {'variant': 'no_backreaction_live', 'growth_rule': 'no_backreaction', 'source': args.source, 'pairings': args.pairings},
    ]
    rows = [run_case(c, args, out) for c in cases]
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
                'mode': m['operator_mode'],
                'beta': [m[f'beta{i}'] for i in range(4)],
                'Hdim': m['harmonic_dim_real'],
                'K_total': m['K_axial_total_norm'],
                'harmonic_axial_ratio': m['harmonic_axial_ratio'],
                'kappa_birth_orientation_ratio': m['kappa_birth_orientation_ratio'],
                'mirror_signed_sum_abs': m['kappa_flip_birth_signed_sum_abs'],
                'delta_vs_legacy': m['relative_axial_field_change_vs_legacy_sym_metric'],
            }
            for r in rows for m in r['operator_mode_rows']
        ]
    }, indent=2))


if __name__ == '__main__':
    main()
