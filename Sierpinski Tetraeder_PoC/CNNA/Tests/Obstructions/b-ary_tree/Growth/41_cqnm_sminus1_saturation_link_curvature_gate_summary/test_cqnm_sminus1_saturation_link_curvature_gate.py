#!/usr/bin/env python3
"""
test_cqnm_sminus1_saturation_link_curvature_gate.py

Exploratory CNNA geometry-gate test after the "Provenienz ist nicht die
Geometrie" obstruction.

The script keeps four layers separate:

P  provenance / birth records: order, parent, sibling order, record/live weights
G  primal simplicial complex: tetrahedra, faces, occupancy, links, Betti numbers
R  response cochain: a diagnostic skew-sector K on triangular faces, derived only
   from available provenance fields in this toy test
J  diagnostics: weak local J-plane statistics only after topology/cochain tests

It compares:

A  SG-like inward stellar subdivision of a tetrahedral ball
B  naive outward NGF attachment to boundary faces
C  CQNM/s=-1 saturated closed primal complex, implemented as a periodic
   Freudenthal triangulation of T^3 with a dual-spanning-tree birth ordering
D  randomized saturation control with the same tetrahedron count as C

The purpose is not to prove CNNA physics. The purpose is to falsify the old
geometry reading early: local face K is not enough; a serious candidate must
survive saturation, link-cycle curvature, and exactness/harmonic gates on the
primal simplicial geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Vertex = int
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, Vertex, Vertex]
Tet = Tuple[Vertex, Vertex, Vertex, Vertex]


@dataclass(frozen=True)
class ProvenanceRecord:
    birth_id: int
    simplex: Tet
    parent_birth: int
    attached_face: Optional[Face]
    sibling_order: int
    depth: int
    record_weight: float
    live_weight: float
    model_tag: str


@dataclass
class ModelRun:
    name: str
    description: str
    tets: List[Tet]
    records: List[ProvenanceRecord]


def sorted_tuple(xs: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(xs))


def tet_faces(tet: Tet) -> List[Face]:
    return [tuple(c) for c in combinations(tet, 3)]


def tet_edges(tet: Tet) -> List[Edge]:
    return [tuple(c) for c in combinations(tet, 2)]


def face_edges(face: Face) -> List[Edge]:
    return [tuple(c) for c in combinations(face, 2)]


def canonical_tet(xs: Iterable[int]) -> Tet:
    t = tuple(sorted(xs))
    if len(t) != 4 or len(set(t)) != 4:
        raise ValueError(f"degenerate tetrahedron: {xs}")
    return t  # type: ignore[return-value]


def canonical_face(xs: Iterable[int]) -> Face:
    f = tuple(sorted(xs))
    if len(f) != 3 or len(set(f)) != 3:
        raise ValueError(f"degenerate face: {xs}")
    return f  # type: ignore[return-value]


def deterministic_energy(simplex: Sequence[int], salt: int = 0) -> float:
    acc = 1469598103934665603 ^ (salt + 0x9E3779B97F4A7C15)
    for x in simplex:
        acc ^= (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        acc = (acc * 1099511628211) & ((1 << 64) - 1)
    return ((acc % 1000003) / 1000003.0) + 1e-6


def ngf_face_weight(face: Face, occupancy: int, flavor: int, beta: float) -> float:
    n_alpha = occupancy - 1
    raw = 1.0 + flavor * n_alpha
    if raw <= 0.0:
        return 0.0
    return raw * math.exp(-beta * deterministic_energy(face, salt=17))


def weighted_choice(items: Sequence[Face], weights: Sequence[float], rng: random.Random) -> Face:
    total = float(sum(weights))
    if total <= 0.0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += weight
        if acc >= r:
            return item
    return items[-1]


def build_sg_like_subdivision(levels: int = 3) -> ModelRun:
    tets: List[Tet] = [canonical_tet((0, 1, 2, 3))]
    records: List[ProvenanceRecord] = [
        ProvenanceRecord(0, tets[0], -1, None, 0, 0, 1.0, 1.0, "sg_seed")
    ]
    next_vertex = 4
    next_birth = 1
    parent_births = [0]

    for depth in range(1, levels + 1):
        new_tets: List[Tet] = []
        new_records: List[ProvenanceRecord] = []
        new_parent_births: List[int] = []
        for parent_tet, parent_birth in zip(tets, parent_births):
            c = next_vertex
            next_vertex += 1
            faces = tet_faces(parent_tet)
            for sibling_order, face in enumerate(faces):
                child = canonical_tet((*face, c))
                rw = 1.0 / (1.0 + depth)
                lw = rw + (sibling_order + 1.0) / (10.0 + depth)
                rec = ProvenanceRecord(
                    next_birth,
                    child,
                    parent_birth,
                    face,
                    sibling_order,
                    depth,
                    rw,
                    lw,
                    "sg_stellar_child",
                )
                new_tets.append(child)
                new_records.append(rec)
                new_parent_births.append(next_birth)
                next_birth += 1
        tets = new_tets
        records = new_records
        parent_births = new_parent_births

    return ModelRun(
        name="A_sg_like_subdivision",
        description="inward stellar subdivision of a tetrahedral 3-ball; topology should remain trivial",
        tets=tets,
        records=records,
    )


def build_naive_ngf_attachment(num_tets: int, beta: float, flavor: int, seed: int) -> ModelRun:
    rng = random.Random(seed)
    tets: List[Tet] = [canonical_tet((0, 1, 2, 3))]
    records: List[ProvenanceRecord] = [
        ProvenanceRecord(0, tets[0], -1, None, 0, 0, 1.0, 1.0, "naive_seed")
    ]
    next_vertex = 4
    face_to_births: Dict[Face, List[int]] = defaultdict(list)
    for f in tet_faces(tets[0]):
        face_to_births[f].append(0)
    children_of: Counter[int] = Counter()

    while len(tets) < num_tets:
        boundary_faces = sorted([f for f, bs in face_to_births.items() if len(bs) == 1])
        if not boundary_faces:
            break
        weights = [ngf_face_weight(f, len(face_to_births[f]), flavor, beta) for f in boundary_faces]
        face = weighted_choice(boundary_faces, weights, rng)
        parent_birth = face_to_births[face][0]
        sibling_order = children_of[parent_birth]
        children_of[parent_birth] += 1
        depth = records[parent_birth].depth + 1 if parent_birth >= 0 else 0
        new_tet = canonical_tet((*face, next_vertex))
        next_vertex += 1
        birth_id = len(records)
        rw = 1.0 + deterministic_energy(face, salt=31)
        lw = rw + (1.0 + sibling_order) / (2.0 + depth)
        rec = ProvenanceRecord(
            birth_id,
            new_tet,
            parent_birth,
            face,
            sibling_order,
            depth,
            rw,
            lw,
            "naive_outward_ngf_birth",
        )
        tets.append(new_tet)
        records.append(rec)
        for f in tet_faces(new_tet):
            face_to_births[f].append(birth_id)

    return ModelRun(
        name="B_naive_outward_ngf",
        description="one tetrahedron attached to one boundary triangle per birth; outward but usually ball-like",
        tets=tets,
        records=records,
    )


def freudenthal_periodic_t3_tets(n: int) -> List[Tet]:
    if n < 2:
        raise ValueError("periodic T^3 grid needs n >= 2")

    def vid(i: int, j: int, k: int) -> int:
        return (i % n) * n * n + (j % n) * n + (k % n)

    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tets: List[Tet] = []
    seen = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                base = (i, j, k)
                end = (i + 1, j + 1, k + 1)
                for p in permutations(range(3)):
                    cur = list(base)
                    pts = [base]
                    for idx in p[:2]:
                        cur = [cur[a] + axes[idx][a] for a in range(3)]
                        pts.append(tuple(cur))
                    pts.append(end)
                    tet = canonical_tet(vid(*pt) for pt in pts)
                    if tet not in seen:
                        seen.add(tet)
                        tets.append(tet)
    return tets


def order_tets_by_dual_bfs(tets: Sequence[Tet], model_tag: str) -> Tuple[List[Tet], List[ProvenanceRecord]]:
    face_to_tets: Dict[Face, List[int]] = defaultdict(list)
    for idx, tet in enumerate(tets):
        for face in tet_faces(tet):
            face_to_tets[face].append(idx)

    neighbors: Dict[int, List[Tuple[int, Face]]] = defaultdict(list)
    for face, ids in face_to_tets.items():
        for a in ids:
            for b in ids:
                if a != b:
                    neighbors[a].append((b, face))

    old_to_new: Dict[int, int] = {}
    parent_old: Dict[int, int] = {}
    attach_face_old: Dict[int, Optional[Face]] = {}
    order: List[int] = []
    q = deque([0])
    old_to_new[0] = 0
    parent_old[0] = -1
    attach_face_old[0] = None

    while q:
        u = q.popleft()
        order.append(u)
        for v, face in sorted(neighbors[u], key=lambda x: (x[0], x[1])):
            if v not in old_to_new:
                old_to_new[v] = len(old_to_new)
                parent_old[v] = u
                attach_face_old[v] = face
                q.append(v)

    if len(order) != len(tets):
        raise ValueError("dual graph is disconnected; cannot create one provenance birth tree")

    new_tets = [tets[i] for i in order]
    old_index_to_birth = {old: birth for birth, old in enumerate(order)}
    children_of: Counter[int] = Counter()
    depths: Dict[int, int] = {}
    records: List[ProvenanceRecord] = []

    for birth_id, old in enumerate(order):
        parent = parent_old[old]
        parent_birth = old_index_to_birth[parent] if parent >= 0 else -1
        if parent_birth >= 0:
            sibling_order = children_of[parent_birth]
            children_of[parent_birth] += 1
            depth = depths[parent_birth] + 1
        else:
            sibling_order = 0
            depth = 0
        depths[birth_id] = depth
        face = attach_face_old[old]
        rw = 1.0 + deterministic_energy(new_tets[birth_id], salt=43)
        lw = rw + (1.0 + sibling_order) / (3.0 + depth)
        records.append(
            ProvenanceRecord(
                birth_id,
                new_tets[birth_id],
                parent_birth,
                face,
                sibling_order,
                depth,
                rw,
                lw,
                model_tag,
            )
        )
    return new_tets, records


def build_cqnm_sminus1_saturated_t3(periodic_n: int) -> ModelRun:
    raw_tets = freudenthal_periodic_t3_tets(periodic_n)
    tets, records = order_tets_by_dual_bfs(raw_tets, "cqnm_sminus1_saturated_t3_birth")
    return ModelRun(
        name="C_cqnm_sminus1_saturated_T3",
        description="closed face-saturated periodic primal 3-complex; T^3 control with nontrivial H1/H2",
        tets=tets,
        records=records,
    )


def find_closing_candidates_limited(
    tets: Sequence[Tet], rng: random.Random, max_trials: int = 750
) -> List[Tuple[int, Tet, List[Face]]]:
    face_occ: Counter[Face] = Counter()
    existing_tets = set(tets)
    for tet in tets:
        for face in tet_faces(tet):
            face_occ[face] += 1
    boundary = [f for f, c in face_occ.items() if c == 1]
    if len(boundary) < 2:
        return []

    candidates: Dict[Tet, Tuple[int, Tet, List[Face]]] = {}
    trials = min(max_trials, max(1, len(boundary) * 8))
    for _ in range(trials):
        f1, f2 = rng.sample(boundary, 2)
        quad = tuple(sorted(set(f1) | set(f2)))
        if len(quad) != 4:
            continue
        tet = canonical_tet(quad)
        if tet in existing_tets:
            continue
        faces = tet_faces(tet)
        if any(face_occ.get(f, 0) >= 2 for f in faces):
            continue
        shared = [f for f in faces if face_occ.get(f, 0) == 1]
        if len(shared) >= 2:
            old = candidates.get(tet)
            item = (len(shared), tet, shared)
            if old is None or item[0] > old[0]:
                candidates[tet] = item

    return sorted(candidates.values(), key=lambda x: (-x[0], x[1]))

def build_random_saturation_control(num_tets: int, seed: int, close_probability: float = 0.55) -> ModelRun:
    rng = random.Random(seed)
    tets: List[Tet] = [canonical_tet((0, 1, 2, 3))]
    records: List[ProvenanceRecord] = [
        ProvenanceRecord(0, tets[0], -1, None, 0, 0, 1.0, 1.0, "random_seed")
    ]
    next_vertex = 4
    face_to_births: Dict[Face, List[int]] = defaultdict(list)
    for f in tet_faces(tets[0]):
        face_to_births[f].append(0)
    children_of: Counter[int] = Counter()

    while len(tets) < num_tets:
        do_close = rng.random() < close_probability
        new_tet: Optional[Tet] = None
        attached_face: Optional[Face] = None
        parent_birth = -1
        if do_close:
            candidates = find_closing_candidates_limited(tets, rng)
            if candidates:
                top = candidates[: min(12, len(candidates))]
                _, new_tet, shared_faces = rng.choice(top)
                attached_face = rng.choice(shared_faces)
                parent_birth = face_to_births[attached_face][0]
        if new_tet is None:
            boundary_faces = sorted([f for f, bs in face_to_births.items() if len(bs) == 1])
            if not boundary_faces:
                break
            attached_face = rng.choice(boundary_faces)
            parent_birth = face_to_births[attached_face][0]
            new_tet = canonical_tet((*attached_face, next_vertex))
            next_vertex += 1
        birth_id = len(records)
        sibling_order = children_of[parent_birth]
        children_of[parent_birth] += 1
        depth = records[parent_birth].depth + 1 if parent_birth >= 0 else 0
        rw = 1.0 + deterministic_energy(new_tet, salt=59)
        lw = rw + rng.random() * (1.0 + sibling_order) / (2.0 + depth)
        rec = ProvenanceRecord(
            birth_id,
            new_tet,
            parent_birth,
            attached_face,
            sibling_order,
            depth,
            rw,
            lw,
            "random_saturation_control_birth",
        )
        tets.append(new_tet)
        records.append(rec)
        for f in tet_faces(new_tet):
            face_to_births[f].append(birth_id)
            if len(face_to_births[f]) > 2:
                raise RuntimeError("random control violated s=-1 occupancy bound")

    return ModelRun(
        name="D_random_saturation_control",
        description="same size order as C but random closing/outward moves; wrong provenance/control geometry",
        tets=tets,
        records=records,
    )


def collect_simplices(tets: Sequence[Tet]) -> Tuple[List[int], List[Edge], List[Face], List[Tet]]:
    vertices = sorted({v for tet in tets for v in tet})
    edges = sorted({e for tet in tets for e in tet_edges(tet)})
    faces = sorted({f for tet in tets for f in tet_faces(tet)})
    uniq_tets = sorted(set(tets))
    return vertices, edges, faces, uniq_tets


def boundary_matrix_z2(high: Sequence[Tuple[int, ...]], low: Sequence[Tuple[int, ...]]) -> np.ndarray:
    low_index = {s: i for i, s in enumerate(low)}
    mat = np.zeros((len(low), len(high)), dtype=np.uint8)
    for j, simplex in enumerate(high):
        for face in combinations(simplex, len(simplex) - 1):
            mat[low_index[tuple(face)], j] ^= 1
    return mat


def rank_mod2(mat: np.ndarray) -> int:
    a = np.array(mat, dtype=np.uint8, copy=True) & 1
    m, n = a.shape
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]
        for r in range(m):
            if r != row and a[r, col]:
                a[r, :] ^= a[row, :]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def betti_numbers_z2(tets: Sequence[Tet]) -> Dict[str, int]:
    vertices, edges, faces, uniq_tets = collect_simplices(tets)
    v_s = [(v,) for v in vertices]
    b1 = boundary_matrix_z2(edges, v_s) if edges else np.zeros((len(v_s), 0), dtype=np.uint8)
    b2 = boundary_matrix_z2(faces, edges) if faces else np.zeros((len(edges), 0), dtype=np.uint8)
    b3 = boundary_matrix_z2(uniq_tets, faces) if uniq_tets else np.zeros((len(faces), 0), dtype=np.uint8)
    r1 = rank_mod2(b1)
    r2 = rank_mod2(b2)
    r3 = rank_mod2(b3)
    c0, c1, c2, c3 = len(vertices), len(edges), len(faces), len(uniq_tets)
    return {
        "beta0": int(c0 - r1),
        "beta1": int(c1 - r1 - r2),
        "beta2": int(c2 - r2 - r3),
        "beta3": int(c3 - r3),
    }


def oriented_boundary_matrix(high: Sequence[Tuple[int, ...]], low: Sequence[Tuple[int, ...]]) -> np.ndarray:
    low_index = {s: i for i, s in enumerate(low)}
    mat = np.zeros((len(low), len(high)), dtype=float)
    for j, simplex in enumerate(high):
        simplex = tuple(simplex)
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            mat[low_index[face], j] += -1.0 if (i % 2) else 1.0
    return mat


def components_of_graph(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> int:
    if not nodes:
        return 0
    parent = {v: v for v in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)
    return len({find(v) for v in nodes})


def edge_link_graph(edge: Edge, incident_tets: Sequence[Tet]) -> Tuple[List[int], List[Edge]]:
    nodes = set()
    link_edges = set()
    e_set = set(edge)
    for tet in incident_tets:
        if not e_set.issubset(tet):
            continue
        opp = sorted(set(tet) - e_set)
        if len(opp) == 2:
            nodes.update(opp)
            link_edges.add((opp[0], opp[1]))
    return sorted(nodes), sorted(link_edges)


def complex_topology_metrics(model: ModelRun) -> Dict[str, object]:
    vertices, edges, faces, uniq_tets = collect_simplices(model.tets)
    face_occ: Counter[Face] = Counter()
    edge_to_tets: Dict[Edge, List[Tet]] = defaultdict(list)
    face_to_tets: Dict[Face, List[Tet]] = defaultdict(list)
    for tet in uniq_tets:
        for face in tet_faces(tet):
            face_occ[face] += 1
            face_to_tets[face].append(tet)
        for edge in tet_edges(tet):
            edge_to_tets[edge].append(tet)

    occ_counts = Counter(face_occ.values())
    boundary_faces = [f for f, c in face_occ.items() if c == 1]
    saturated_faces = [f for f, c in face_occ.items() if c == 2]
    overfilled_faces = [f for f, c in face_occ.items() if c > 2]
    betti = betti_numbers_z2(uniq_tets)

    theta = math.acos(1.0 / 3.0)
    link_betas: List[int] = []
    link_cycle_edges: List[Edge] = []
    defects_all: List[float] = []
    defects_cycle: List[float] = []
    link_degree_errors: List[float] = []

    for edge in edges:
        inc = edge_to_tets[edge]
        nodes, l_edges = edge_link_graph(edge, inc)
        comps = components_of_graph(nodes, l_edges)
        beta1 = len(l_edges) - len(nodes) + comps if nodes else 0
        link_betas.append(beta1)
        degrees = Counter()
        for a, b in l_edges:
            degrees[a] += 1
            degrees[b] += 1
        degree_error = float(sum(abs(degrees[v] - 2) for v in nodes)) / max(1, len(nodes))
        link_degree_errors.append(degree_error)
        q = len(inc)
        defect = 2.0 * math.pi - q * theta
        defects_all.append(defect)
        if beta1 >= 1 and comps == 1 and degree_error == 0.0:
            link_cycle_edges.append(edge)
            defects_cycle.append(defect)

    return {
        "counts": {
            "vertices": len(vertices),
            "edges": len(edges),
            "faces": len(faces),
            "tets": len(uniq_tets),
        },
        "face_occupancy_counts": {str(k): int(v) for k, v in sorted(occ_counts.items())},
        "boundary_face_count": len(boundary_faces),
        "boundary_fraction": float(len(boundary_faces) / max(1, len(faces))),
        "saturated_face_count": len(saturated_faces),
        "saturated_face_fraction": float(len(saturated_faces) / max(1, len(faces))),
        "overfilled_face_count": len(overfilled_faces),
        "manifold_face_ok_fraction": float(
            sum(1 for c in face_occ.values() if c in (1, 2)) / max(1, len(faces))
        ),
        "betti_z2": betti,
        "edge_link_cycle_count": len(link_cycle_edges),
        "edge_link_cycle_fraction": float(len(link_cycle_edges) / max(1, len(edges))),
        "edge_link_beta1_mean": float(np.mean(link_betas)) if link_betas else 0.0,
        "edge_link_degree_error_mean": float(np.mean(link_degree_errors)) if link_degree_errors else 0.0,
        "regge_regular_tet_defect_abs_mean_all_edges": float(np.mean(np.abs(defects_all))) if defects_all else 0.0,
        "regge_regular_tet_defect_abs_mean_cycle_edges": float(np.mean(np.abs(defects_cycle))) if defects_cycle else 0.0,
    }


def orthonormal_colspace(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if a.size == 0:
        return np.zeros((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if s.size == 0:
        return np.zeros((a.shape[0], 0))
    cutoff = tol * max(a.shape) * max(float(s[0]), 1.0)
    r = int(np.sum(s > cutoff))
    return u[:, :r]


def orthonormal_kernel(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if a.size == 0:
        return np.eye(a.shape[1])
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    cutoff = tol * max(a.shape) * max(float(s[0]) if s.size else 1.0, 1.0)
    r = int(np.sum(s > cutoff))
    return vt[r:, :].T.copy()


def project(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return np.zeros_like(x)
    return q @ (q.T @ x)


def record_vector(record: ProvenanceRecord, mode: str) -> np.ndarray:
    if mode == "diagonal_trace_kill":
        return np.zeros(3)
    depth = float(record.depth + 1)
    sibling = float(record.sibling_order + 1)
    live_gap = record.live_weight - record.record_weight
    if mode == "no_backreaction":
        live_gap = 0.0
    if mode == "symmetrized_birth":
        sibling = 1.0
        live_gap = 0.0
    v = np.array(
        [
            ((record.birth_id % 3) - 1.0) * sibling / depth,
            ((-1.0) ** record.sibling_order) / (1.0 + depth),
            live_gap / (1.0 + abs(record.record_weight) + abs(record.live_weight)),
        ],
        dtype=float,
    )
    if mode == "symmetrized_birth":
        v[0] = 0.0
        v[2] = 0.0
    return v


def k_cochain_on_faces(model: ModelRun, faces: Sequence[Face], tets: Sequence[Tet], mode: str) -> np.ndarray:
    face_index = {f: i for i, f in enumerate(faces)}
    tet_to_record: Dict[Tet, ProvenanceRecord] = {rec.simplex: rec for rec in model.records}
    k = np.zeros((len(faces), 3), dtype=float)
    for tet in tets:
        rec = tet_to_record.get(tet)
        if rec is None:
            continue
        vec = record_vector(rec, mode)
        simplex = tuple(tet)
        for i in range(4):
            face = simplex[:i] + simplex[i + 1 :]
            sign = -1.0 if (i % 2) else 1.0
            amp = 1.0
            if rec.attached_face is not None and tuple(face) == rec.attached_face:
                amp = 1.35
            k[face_index[tuple(face)], :] += sign * amp * vec
    return k


def cochain_metrics(model: ModelRun, mode: str) -> Dict[str, float]:
    vertices, edges, faces, tets = collect_simplices(model.tets)
    b2 = oriented_boundary_matrix(faces, edges) if faces and edges else np.zeros((len(edges), len(faces)))
    b3 = oriented_boundary_matrix(tets, faces) if tets and faces else np.zeros((len(faces), len(tets)))
    delta1 = b2.T
    delta2 = b3.T
    k = k_cochain_on_faces(model, faces, tets, mode)
    norm_k = float(np.linalg.norm(k))
    if norm_k <= 1e-14:
        return {
            "K_norm": 0.0,
            "nonzero_face_fraction": 0.0,
            "exact_residual_ratio": 0.0,
            "closed_residual_ratio": 0.0,
            "closed_projection_ratio": 0.0,
            "harmonic_projection_ratio": 0.0,
            "edge_flux_norm_ratio": 0.0,
            "cycle_edge_flux_norm_mean": 0.0,
            "weak_local_J_plane_residual": 0.0,
        }

    q_exact = orthonormal_colspace(delta1)
    q_closed = orthonormal_kernel(delta2)
    k_exact = project(q_exact, k)
    k_closed = project(q_closed, k)
    k_harm = k_closed - project(q_exact, k_closed)
    exact_res = float(np.linalg.norm(k - k_exact) / norm_k)
    closed_res = float(np.linalg.norm(delta2 @ k) / norm_k)
    closed_ratio = float(np.linalg.norm(k_closed) / norm_k)
    harmonic_ratio = float(np.linalg.norm(k_harm) / norm_k)

    edge_flux = b2 @ k if b2.size else np.zeros((len(edges), 3))
    edge_flux_norm_ratio = float(np.linalg.norm(edge_flux) / norm_k)

    edge_to_tets: Dict[Edge, List[Tet]] = defaultdict(list)
    for tet in tets:
        for edge in tet_edges(tet):
            edge_to_tets[edge].append(tet)
    cycle_edge_fluxes = []
    for edge_idx, edge in enumerate(edges):
        nodes, l_edges = edge_link_graph(edge, edge_to_tets[edge])
        comps = components_of_graph(nodes, l_edges)
        beta1 = len(l_edges) - len(nodes) + comps if nodes else 0
        degrees = Counter()
        for a, b in l_edges:
            degrees[a] += 1
            degrees[b] += 1
        degree_error = sum(abs(degrees[v] - 2) for v in nodes)
        if beta1 >= 1 and comps == 1 and degree_error == 0:
            cycle_edge_fluxes.append(float(np.linalg.norm(edge_flux[edge_idx])))

    face_norms = np.linalg.norm(k, axis=1)
    nonzero_fraction = float(np.mean(face_norms > 1e-12))
    local_residuals = []
    for v in k:
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
            continue
        u = v / nv
        skew = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        projector_plane = np.eye(3) - np.outer(u, u)
        local_residuals.append(float(np.linalg.norm(skew @ skew + projector_plane)))

    return {
        "K_norm": norm_k,
        "nonzero_face_fraction": nonzero_fraction,
        "exact_residual_ratio": exact_res,
        "closed_residual_ratio": closed_res,
        "closed_projection_ratio": closed_ratio,
        "harmonic_projection_ratio": harmonic_ratio,
        "edge_flux_norm_ratio": edge_flux_norm_ratio,
        "cycle_edge_flux_norm_mean": float(np.mean(cycle_edge_fluxes)) if cycle_edge_fluxes else 0.0,
        "weak_local_J_plane_residual": float(np.mean(local_residuals)) if local_residuals else 0.0,
    }


def gate_report(summary: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    gates: Dict[str, object] = {}
    for name, data in summary.items():
        topo = data["topology"]  # type: ignore[index]
        betti = topo["betti_z2"]  # type: ignore[index]
        gates[name] = {
            "no_overfilled_faces": topo["overfilled_face_count"] == 0,
            "face_manifold_occupancy_ok": topo["manifold_face_ok_fraction"] == 1.0,
            "closed_or_saturated": topo["boundary_fraction"] < 1e-12,
            "nontrivial_link_cycles": topo["edge_link_cycle_fraction"] > 0.25,
            "nontrivial_beta2": betti["beta2"] > 0,  # type: ignore[index]
            "real_K_has_harmonic_projection": data["cochains"]["real_growth"]["harmonic_projection_ratio"] > 1e-8,  # type: ignore[index]
            "diagonal_trace_kills_K": data["cochains"]["diagonal_trace_kill"]["K_norm"] == 0.0,  # type: ignore[index]
        }

    gates["expected_pattern"] = {
        "A_B_should_not_be_stage4_positive": (
            not gates["A_sg_like_subdivision"]["closed_or_saturated"]
            and not gates["B_naive_outward_ngf"]["closed_or_saturated"]
        ),
        "C_should_supply_saturated_nontrivial_geometry": (
            gates["C_cqnm_sminus1_saturated_T3"]["closed_or_saturated"]
            and gates["C_cqnm_sminus1_saturated_T3"]["nontrivial_link_cycles"]
            and gates["C_cqnm_sminus1_saturated_T3"]["nontrivial_beta2"]
        ),
        "D_should_remain_a_control_not_a_clean_positive": (
            not (
                gates["D_random_saturation_control"]["closed_or_saturated"]
                and gates["D_random_saturation_control"]["nontrivial_beta2"]
            )
        ),
    }
    return gates


def run_suite(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    c_model = build_cqnm_sminus1_saturated_t3(args.periodic_n)
    target_tets = len(c_model.tets)
    models = [
        build_sg_like_subdivision(args.sg_levels),
        build_naive_ngf_attachment(target_tets, args.beta, args.naive_flavor, args.seed),
        c_model,
        build_random_saturation_control(target_tets, args.seed + 1009, args.random_close_probability),
    ]
    modes = ["real_growth", "symmetrized_birth", "no_backreaction", "diagonal_trace_kill"]
    summary: Dict[str, Dict[str, object]] = {}
    for model in models:
        topology = complex_topology_metrics(model)
        cochains = {mode: cochain_metrics(model, mode) for mode in modes}
        summary[model.name] = {
            "description": model.description,
            "topology": topology,
            "cochains": cochains,
            "provenance_sample": [asdict(r) for r in model.records[: min(5, len(model.records))]],
        }
    summary["__gates__"] = gate_report(summary)  # type: ignore[assignment]
    return summary


def compact_print(summary: Dict[str, Dict[str, object]]) -> None:
    print("\nCNNA CQNM/s=-1 saturation-link-curvature gate\n")
    header = (
        "model",
        "V/E/F/T",
        "bdry",
        "sat",
        "β0β1β2β3",
        "linkcyc",
        "|def|cyc",
        "H_K(real)",
        "exact_res",
    )
    print("{:<34} {:>15} {:>8} {:>8} {:>12} {:>9} {:>10} {:>10} {:>10}".format(*header))
    print("-" * 128)
    for name, data in summary.items():
        if name == "__gates__":
            continue
        topo = data["topology"]  # type: ignore[index]
        counts = topo["counts"]  # type: ignore[index]
        betti = topo["betti_z2"]  # type: ignore[index]
        real = data["cochains"]["real_growth"]  # type: ignore[index]
        count_s = f"{counts['vertices']}/{counts['edges']}/{counts['faces']}/{counts['tets']}"
        beta_s = f"{betti['beta0']}{betti['beta1']}{betti['beta2']}{betti['beta3']}"
        print(
            "{:<34} {:>15} {:8.3f} {:8.3f} {:>12} {:9.3f} {:10.3g} {:10.3g} {:10.3g}".format(
                name,
                count_s,
                topo["boundary_fraction"],
                topo["saturated_face_fraction"],
                beta_s,
                topo["edge_link_cycle_fraction"],
                topo["regge_regular_tet_defect_abs_mean_cycle_edges"],
                real["harmonic_projection_ratio"],
                real["exact_residual_ratio"],
            )
        )
    print("\nGate pattern:")
    gates = summary["__gates__"]["expected_pattern"]  # type: ignore[index]
    for key, value in gates.items():
        print(f"  {key}: {value}")
    print("\nInterpretation rule:")
    print("  K≠0 alone is weak. A serious stage-4 candidate needs saturated/link-cyclic geometry")
    print("  plus non-exact/harmonic response content; otherwise the local J-like sector is still a coboundary/diagnostic artefact.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periodic-n", type=int, default=2, help="periodic grid size for the saturated T^3 control, n>=2")
    parser.add_argument("--sg-levels", type=int, default=3, help="stellar subdivision levels for SG-like negative control")
    parser.add_argument("--beta", type=float, default=1.0, help="NGF beta used in boundary face weights")
    parser.add_argument("--naive-flavor", type=int, default=0, choices=[-1, 0, 1], help="NGF flavor for naive outward control")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--random-close-probability", type=float, default=0.55)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--strict-exit", action="store_true", help="return nonzero if the expected control pattern fails")
    args = parser.parse_args()

    summary = run_suite(args)
    compact_print(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nwrote JSON summary: {args.json_out}")

    if args.strict_exit:
        pattern = summary["__gates__"]["expected_pattern"]  # type: ignore[index]
        if not all(bool(v) for v in pattern.values()):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
