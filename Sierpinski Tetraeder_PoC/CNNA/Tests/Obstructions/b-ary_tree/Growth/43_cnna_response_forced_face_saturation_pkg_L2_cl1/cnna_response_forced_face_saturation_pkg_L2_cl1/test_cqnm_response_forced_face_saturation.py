#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

EPS = 1e-12
Simplex = Tuple[int, ...]
Face = Tuple[int, int, int]
Tet = Tuple[int, int, int, int]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def frame_from_radial(radial: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = unit(radial)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, r))) > 0.92:
        seed = np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(r, seed))
    e2 = unit(np.cross(r, e1))
    return r, e1, e2


def fro(M: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(M, dtype=float), ord="fro"))


def skew(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M - M.T)


def sym(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else 0.0


def perc(xs: Iterable[float], q: float) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.percentile(vals, q)) if vals else 0.0


def axial_from_skew(K: np.ndarray) -> np.ndarray:
    K = skew(K)
    return np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=float)


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


class DynamicProvenanceGrowth:
    def __init__(
        self,
        mode: str = "linear",
        growth_rule: str = "real_growth",
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
        transverse_amp: float = 0.42,
        radial_step: float = 1.0,
    ):
        if branching != 3:
            raise ValueError("This diagnostic uses ternary sibling fans.")
        if growth_rule not in {"real_growth", "symmetrized_birth", "no_backreaction"}:
            raise ValueError(growth_rule)
        self.mode = mode
        self.growth_rule = growth_rule
        self.branching = branching
        self.base = base
        self.alpha_env = alpha_env
        self.ancestor_decay = ancestor_decay
        self.br_ancestor = 0.0 if growth_rule == "no_backreaction" else br_ancestor
        self.br_sibling = 0.0 if growth_rule == "no_backreaction" else br_sibling
        self.transverse_amp = transverse_amp
        self.radial_step = radial_step
        self.nodes: Dict[int, Node] = {}
        self.t = 0
        self.next_id = 0
        self.directed_edges: Dict[Tuple[int, int], float] = defaultdict(float)
        self.birth_events: List[dict] = []
        self.level_rows: List[dict] = []
        r, e1, e2 = frame_from_radial(np.array([0.0, 0.0, 1.0]))
        root = self._new_node(None, 0, 0, 1.0, np.zeros(3), r, e1, e2)
        self.root = root.id

    def _new_node(
        self,
        parent: Optional[int],
        level: int,
        birth_order: int,
        birth_g: float,
        pos: np.ndarray,
        radial: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
    ) -> Node:
        n = Node(self.next_id, parent, level, birth_order, self.t, birth_g, birth_g, pos, radial, e1, e2)
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def parent_line(self, parent: int) -> List[int]:
        line: List[int] = []
        cur: Optional[int] = parent
        while cur is not None:
            line.append(cur)
            cur = self.nodes[cur].parent
        return line

    def birth_environment_load(self, parent: int, older_siblings: List[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_decay ** (d - 1))
        if self.growth_rule != "symmetrized_birth":
            for s in older_siblings:
                env += self.nodes[s].g
        return env

    def child_conductance_from_env(self, env_load: float) -> float:
        if self.mode == "linear":
            return self.base + self.alpha_env * env_load
        if self.mode == "log":
            return self.base + self.alpha_env * math.log1p(env_load)
        if self.mode == "saturating":
            return self.base + self.alpha_env * (env_load / (1.0 + env_load))
        raise ValueError(self.mode)

    def child_position(self, parent: int, order: int, older_siblings: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.nodes[parent]
        theta = 2.0 * math.pi * (order - 1) / 3.0
        twist = 0.37 * sum((i + 1) * x for i, x in enumerate(self.address_tuple(parent)))
        trans = math.cos(theta + twist) * p.e1 + math.sin(theta + twist) * p.e2
        older_push = np.zeros(3)
        for s in older_siblings:
            older_push += unit(p.pos - self.nodes[s].pos)
        direction = unit(p.radial + self.transverse_amp * trans + 0.08 * older_push)
        step = self.radial_step * (1.0 + 0.06 * (order - 2))
        pos = p.pos + step * direction
        r, e1, e2 = frame_from_radial(pos if np.linalg.norm(pos) > EPS else direction)
        return pos, r, e1, e2

    def address_tuple(self, node: int) -> Tuple[int, ...]:
        out: List[int] = []
        cur: Optional[int] = node
        while cur is not None and self.nodes[cur].parent is not None:
            out.append(self.nodes[cur].birth_order)
            cur = self.nodes[cur].parent
        return tuple(reversed(out))

    def add_child(self, parent: int, order: int) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env_load = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env_load)
        pos, r, e1, e2 = self.child_position(parent, order, older)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g, pos, r, e1, e2)
        c = child.id
        total_env = env_load + EPS
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            self.directed_edges[(a, c)] += self.alpha_env * contrib / total_env * birth_g
        if self.growth_rule != "symmetrized_birth":
            for s in older:
                contrib = self.nodes[s].g
                self.directed_edges[(s, c)] += self.alpha_env * contrib / total_env * birth_g
        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * birth_g / (d * d)
            self.nodes[a].g += delta
            self.directed_edges[(c, a)] += delta
        for s in older:
            delta = self.br_sibling * birth_g
            self.nodes[s].g += delta
            self.directed_edges[(c, s)] += delta
        self.birth_events.append({
            "t": self.t,
            "parent": parent,
            "child": c,
            "order": order,
            "older_siblings": list(older),
            "env_load": env_load,
            "birth_g": birth_g,
            "level": child.level,
        })
        return c

    def grow_level(self, frontier: List[int]) -> List[int]:
        nxt: List[int] = []
        for p in frontier:
            for k in range(1, 4):
                nxt.append(self.add_child(p, k))
        return nxt

    def grow(self, max_level: int) -> None:
        frontier = [self.root]
        for _ in range(max_level):
            frontier = self.grow_level(frontier)

    def completed_parent_ids(self) -> List[int]:
        return [i for i, n in self.nodes.items() if len(n.children) == 3]

    def child_ids_ordered(self, parent: int) -> List[int]:
        return sorted(self.nodes[parent].children, key=lambda c: self.nodes[c].birth_order)

    def neutral_for_parent(self, parent: int, current: bool) -> complex:
        ch = self.child_ids_ordered(parent)
        vals = [self.nodes[c].g if current else self.nodes[c].birth_g for c in ch]
        omega = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
        return vals[0] + vals[1] * omega + vals[2] * omega * omega

    def local_cycle_log_bias(self, parent: int) -> float:
        ch = self.child_ids_ordered(parent)
        c1, c2, c3 = ch
        def w(u: int, v: int) -> float:
            return self.directed_edges.get((u, v), 0.0) + EPS
        return math.log((w(c1, c2) * w(c2, c3) * w(c3, c1)) / (w(c1, c3) * w(c3, c2) * w(c2, c1)))


@dataclass
class SimplicialComplex:
    name: str
    vertices: set[int] = field(default_factory=set)
    tets: List[Tet] = field(default_factory=list)
    face_birth: Dict[Face, int] = field(default_factory=dict)

    def add_tet(self, tet: Iterable[int], birth_time: int = 0) -> bool:
        tt = tuple(sorted(set(int(x) for x in tet)))
        if len(tt) != 4:
            return False
        if tt in self.tets:
            return False
        self.tets.append(tt)
        self.vertices.update(tt)
        for f in faces_of_tet(tt):
            self.face_birth.setdefault(f, birth_time)
        return True

    def faces(self) -> List[Face]:
        fs = set()
        for t in self.tets:
            fs.update(faces_of_tet(t))
        return sorted(fs)

    def edges(self) -> List[Tuple[int, int]]:
        es = set()
        for f in self.faces():
            a, b, c = f
            es.add(tuple(sorted((a, b))))
            es.add(tuple(sorted((a, c))))
            es.add(tuple(sorted((b, c))))
        return sorted(es)

    def face_occupancy(self) -> Dict[Face, int]:
        occ: Dict[Face, int] = defaultdict(int)
        for t in self.tets:
            for f in faces_of_tet(t):
                occ[f] += 1
        return dict(occ)


def faces_of_tet(t: Tet) -> List[Face]:
    a, b, c, d = tuple(t)
    return [tuple(sorted(x)) for x in [(b, c, d), (a, c, d), (a, b, d), (a, b, c)]]


def build_parent_fan_tetra_complex(model: DynamicProvenanceGrowth, max_level: int) -> SimplicialComplex:
    K = SimplicialComplex("parent_fan_tetra")
    for p in model.completed_parent_ids():
        if model.nodes[p].level >= max_level:
            continue
        ch = model.child_ids_ordered(p)
        K.add_tet((p, ch[0], ch[1], ch[2]), birth_time=max(model.nodes[c].birth_time for c in ch))
    return K


def choose_boundary_face_for_parent(model: DynamicProvenanceGrowth, faces_by_vertex: Dict[int, set[Face]], occ: Dict[Face, int], parent: int) -> Optional[Face]:
    candidates = [f for f in faces_by_vertex.get(parent, set()) if occ.get(f, 0) == 1]
    if not candidates:
        return None
    pr = model.nodes[parent].radial
    def score(f: Face) -> Tuple[float, int]:
        centroid = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
        outward = unit(centroid - model.nodes[parent].pos)
        return (float(np.dot(outward, pr)), -sum(model.nodes[v].birth_time for v in f))
    return max(candidates, key=score)


def update_face_maps(K: SimplicialComplex, faces_by_vertex: Dict[int, set[Face]], occ: Dict[Face, int], tet: Tet) -> None:
    for f in faces_of_tet(tet):
        occ[f] = occ.get(f, 0) + 1
        for v in f:
            faces_by_vertex.setdefault(v, set()).add(f)


def build_dynamic_outward_ngf_complex(model: DynamicProvenanceGrowth, *, closure_passes: int = 0) -> SimplicialComplex:
    K = SimplicialComplex("dynamic_outward_ngf" if closure_passes <= 0 else "dynamic_sminus1_closure_attempt")
    occ: Dict[Face, int] = {}
    faces_by_vertex: Dict[int, set[Face]] = defaultdict(set)
    root_seeded = False
    for ev in model.birth_events:
        parent = int(ev["parent"])
        child = int(ev["child"])
        if not root_seeded and len(model.nodes[model.root].children) == 3:
            ch = model.child_ids_ordered(model.root)
            tet = tuple(sorted((model.root, ch[0], ch[1], ch[2])))
            if K.add_tet(tet, birth_time=max(model.nodes[c].birth_time for c in ch)):
                update_face_maps(K, faces_by_vertex, occ, tet)
            root_seeded = True
        if child in K.vertices:
            continue
        face = choose_boundary_face_for_parent(model, faces_by_vertex, occ, parent)
        if face is None:
            continue
        tet = tuple(sorted((*face, child)))
        new_faces = faces_of_tet(tet)
        if any(occ.get(f, 0) >= 2 for f in new_faces):
            continue
        if K.add_tet(tet, birth_time=int(ev["t"])):
            update_face_maps(K, faces_by_vertex, occ, tet)
    if closure_passes > 0:
        closure_attempt(model, K, closure_passes)
    return K


def closure_attempt(model: DynamicProvenanceGrowth, K: SimplicialComplex, passes: int) -> None:
    """Heuristic bounded CQNM/s=-1 closure comparison.

    This is deliberately capped. It is not a derived growth law; it asks whether a
    small number of admissible extra tetrahedra can reduce open boundary while
    respecting face occupancy <= 2. Unbounded combinatorial closure would swamp
    the diagnostic and is not methodologically meaningful here.
    """
    max_boundary_faces = 160
    max_vertices_per_face = 18
    max_new_tets_per_pass = 32
    for _ in range(passes):
        changed = False
        occ = K.face_occupancy()
        boundary_all = [f for f, n in occ.items() if n == 1]
        boundary_all.sort(key=lambda f: sum(model.nodes[v].birth_time for v in f))
        boundary = boundary_all[:max_boundary_faces]
        existing = sorted(K.vertices)
        used = set(K.tets)
        candidates: List[Tuple[int, float, Tet]] = []
        for f in boundary:
            fset = set(f)
            fc = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
            fr = unit(fc)
            ranked_vertices: List[Tuple[float, int]] = []
            for x in existing:
                if x in fset:
                    continue
                xr = unit(model.nodes[x].pos)
                radial_ok = float(np.dot(fr, xr))
                dist = float(np.linalg.norm(model.nodes[x].pos - fc))
                ranked_vertices.append((radial_ok - 0.05 * dist, x))
            ranked_vertices.sort(reverse=True)
            for score_x, x in ranked_vertices[:max_vertices_per_face]:
                tet = tuple(sorted((*f, x)))
                if tet in used:
                    continue
                fs = faces_of_tet(tet)
                if any(occ.get(ff, 0) >= 2 for ff in fs):
                    continue
                gain = sum(1 for ff in fs if occ.get(ff, 0) == 1)
                if gain < 2:
                    continue
                candidates.append((gain, score_x, tet))
        candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
        for _, _, tet in candidates[:max_new_tets_per_pass]:
            occ_now = K.face_occupancy()
            if tet in K.tets:
                continue
            if any(occ_now.get(ff, 0) >= 2 for ff in faces_of_tet(tet)):
                continue
            if K.add_tet(tet, birth_time=max(model.nodes[v].birth_time for v in tet)):
                changed = True
        if not changed:
            break


def gf2_rank(A: np.ndarray) -> int:
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if A[r, col]:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def boundary_matrix_mod2(high: List[Simplex], low: List[Simplex]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(low)}
    B = np.zeros((len(low), len(high)), dtype=np.uint8)
    for j, s in enumerate(high):
        for k in range(len(s)):
            face = tuple(s[:k] + s[k + 1:])
            B[idx[tuple(sorted(face))], j] ^= 1
    return B


def boundary_matrix_real(high: List[Simplex], low: List[Simplex]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(low)}
    B = np.zeros((len(low), len(high)), dtype=float)
    for j, s in enumerate(high):
        s = tuple(s)
        for k in range(len(s)):
            face = tuple(s[:k] + s[k + 1:])
            B[idx[tuple(sorted(face))], j] += -1.0 if k % 2 else 1.0
    return B


def topology(K: SimplicialComplex) -> dict:
    verts = sorted(K.vertices)
    edges = K.edges()
    faces = K.faces()
    tets = sorted(K.tets)
    C = [len(verts), len(edges), len(faces), len(tets)]
    B1 = boundary_matrix_mod2([tuple(e) for e in edges], [(v,) for v in verts]) if edges else np.zeros((len(verts), 0), dtype=np.uint8)
    B2 = boundary_matrix_mod2([tuple(f) for f in faces], [tuple(e) for e in edges]) if faces else np.zeros((len(edges), 0), dtype=np.uint8)
    B3 = boundary_matrix_mod2([tuple(t) for t in tets], [tuple(f) for f in faces]) if tets else np.zeros((len(faces), 0), dtype=np.uint8)
    r1, r2, r3 = gf2_rank(B1), gf2_rank(B2), gf2_rank(B3)
    betti = {
        "beta0": C[0] - r1,
        "beta1": C[1] - r1 - r2,
        "beta2": C[2] - r2 - r3,
        "beta3": C[3] - r3,
    }
    occ = K.face_occupancy()
    boundary_faces = sum(1 for n in occ.values() if n == 1)
    saturated_faces = sum(1 for n in occ.values() if n == 2)
    overfull_faces = sum(1 for n in occ.values() if n > 2)
    return {
        "vertices": C[0], "edges": C[1], "faces": C[2], "tets": C[3],
        "euler": C[0] - C[1] + C[2] - C[3],
        **betti,
        "boundary_faces": boundary_faces,
        "saturated_faces": saturated_faces,
        "overfull_faces": overfull_faces,
        "boundary_fraction": boundary_faces / (len(occ) + EPS),
        "saturated_fraction": saturated_faces / (len(occ) + EPS),
        "manifold_face_fraction": (boundary_faces + saturated_faces) / (len(occ) + EPS),
    }


def edge_link_metrics(K: SimplicialComplex) -> dict:
    edge_to_link_edges: Dict[Tuple[int, int], set[Tuple[int, int]]] = defaultdict(set)
    edge_to_link_vertices: Dict[Tuple[int, int], set[int]] = defaultdict(set)
    for t in K.tets:
        tv = list(t)
        for i in range(4):
            for j in range(i + 1, 4):
                e = tuple(sorted((tv[i], tv[j])))
                rest = [x for x in tv if x not in e]
                edge_to_link_vertices[e].update(rest)
                edge_to_link_edges[e].add(tuple(sorted(rest)))
    cycle_edges = 0
    checked = 0
    lengths = []
    for e, les in edge_to_link_edges.items():
        lv = edge_to_link_vertices[e]
        if len(lv) < 3:
            continue
        deg = defaultdict(int)
        for a, b in les:
            deg[a] += 1
            deg[b] += 1
        if set(deg.keys()) != lv:
            continue
        checked += 1
        ok_deg = all(deg[v] == 2 for v in lv)
        if ok_deg:
            start = next(iter(lv))
            seen = {start}
            stack = [start]
            adj = defaultdict(list)
            for a, b in les:
                adj[a].append(b)
                adj[b].append(a)
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if seen == lv:
                cycle_edges += 1
                lengths.append(len(lv))
    return {
        "edge_links_checked": checked,
        "edge_link_cycle_count": cycle_edges,
        "edge_link_cycle_fraction": cycle_edges / (checked + EPS),
        "mean_edge_link_cycle_length": mean(lengths),
    }


def vertex_operator(model: DynamicProvenanceGrowth, node: int, source: str) -> np.ndarray:
    n = model.nodes[node]
    r, e1, e2 = n.radial, n.e1, n.e2
    order_phase = 2.0 * math.pi * (n.birth_order - 1) / 3.0 if n.parent is not None else 0.0
    q = math.cos(order_phase) * e1 + math.sin(order_phase) * e2
    h = unit(0.7 * r + 0.3 * q)
    birth = n.birth_g
    live = n.g
    aging = max(0.0, live - birth)
    if source == "record":
        a, b, c = birth, 0.22 * birth, 0.08 * birth
    elif source == "live":
        a, b, c = live, 0.25 * birth + 0.55 * aging, 0.12 * live
    elif source == "handoff":
        inc = 0.0
        if n.parent is not None:
            inc = model.directed_edges.get((n.parent, node), 0.0) + model.directed_edges.get((node, n.parent), 0.0)
        a, b, c = birth + inc, 0.18 * live + inc, 0.15 * inc + 0.05 * birth
    elif source == "aging":
        a, b, c = aging + 0.1 * birth, 0.6 * aging + 0.03 * birth, 0.3 * aging
    else:
        raise ValueError(source)
    M = (
        a * np.outer(r, r)
        + b * np.outer(q, q)
        + c * np.outer(h, h)
        + 0.04 * birth * np.eye(3)
    )
    return sym(M)


def reduce_ops(Sa: np.ndarray, Sb: np.ndarray, Sc: np.ndarray, reduction: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reduction == "full":
        return Sa, Sb, Sc
    if reduction == "diagonal":
        return np.diag(np.diag(Sa)), np.diag(np.diag(Sb)), np.diag(np.diag(Sc))
    if reduction == "trace_scalar":
        return tuple(np.eye(3) * (float(np.trace(S)) / 3.0) for S in (Sa, Sb, Sc))
    raise ValueError(reduction)


def face_K(model: DynamicProvenanceGrowth, face: Face, source: str, reduction: str) -> np.ndarray:
    a, b, c = face
    Sa = vertex_operator(model, a, source)
    Sb = vertex_operator(model, b, source)
    Sc = vertex_operator(model, c, source)
    Sa, Sb, Sc = reduce_ops(Sa, Sb, Sc, reduction)
    Aab = Sb - Sa
    Abc = Sc - Sb
    return skew(Aab @ Abc - Abc @ Aab)


def project_to_colspace(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros_like(y)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ sol


def nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int((s > tol).sum())
    return vh[rank:].T.copy()


COCHAIN_TOPO_CACHE: Dict[Tuple[Tet, ...], dict] = {}


def cochain_cache(K: SimplicialComplex) -> dict:
    key = tuple(sorted(K.tets))
    hit = COCHAIN_TOPO_CACHE.get(key)
    if hit is not None:
        return hit
    faces = K.faces()
    edges = K.edges()
    tets = sorted(K.tets)
    if faces:
        B2 = boundary_matrix_real([tuple(f) for f in faces], [tuple(e) for e in edges]) if edges else np.zeros((0, len(faces)))
        B3 = boundary_matrix_real([tuple(t) for t in tets], [tuple(f) for f in faces]) if tets else np.zeros((len(faces), 0))
        d1 = B2.T
        d2 = B3.T
        N = nullspace(d2) if d2.size else np.eye(len(faces))
    else:
        d1 = np.zeros((0, 0))
        d2 = np.zeros((0, 0))
        N = np.zeros((0, 0))
    out = {"faces": faces, "edges": edges, "tets": tets, "d1": d1, "d2": d2, "N": N}
    COCHAIN_TOPO_CACHE[key] = out
    return out


def cochain_metrics(K: SimplicialComplex, model: DynamicProvenanceGrowth, source: str, reduction: str) -> dict:
    cache = cochain_cache(K)
    faces = cache["faces"]
    edges = cache["edges"]
    if not faces:
        return {"face_count": 0, "K_mean": 0.0, "K_p95": 0.0, "exact_residual_ratio": 0.0, "closed_residual_ratio": 0.0, "harmonic_ratio": 0.0, "link_flux_mean": 0.0}
    axes = np.vstack([axial_from_skew(face_K(model, f, source, reduction)) for f in faces])
    norms = np.linalg.norm(axes, axis=1)
    d1 = cache["d1"]
    d2 = cache["d2"]
    N = cache["N"]
    exact_res = []
    closed_res = []
    harm = []
    for j in range(3):
        y = axes[:, j]
        yn = float(np.linalg.norm(y)) + EPS
        exact = project_to_colspace(d1, y)
        exact_res.append(float(np.linalg.norm(y - exact) / yn))
        closed_res.append(float(np.linalg.norm(d2 @ y) / yn) if d2.size else 0.0)
        y_closed = N @ (N.T @ y) if N.size else np.zeros_like(y)
        exact_closed = project_to_colspace(d1, y_closed)
        harm.append(float(np.linalg.norm(y_closed - exact_closed) / yn))
    edge_flux = []
    face_axis = {f: axes[i] for i, f in enumerate(faces)}
    edge_to_faces: Dict[Tuple[int, int], List[Face]] = defaultdict(list)
    for f in faces:
        a, b, c = f
        for e in [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]:
            edge_to_faces[e].append(f)
    for _, fs in edge_to_faces.items():
        if len(fs) >= 3:
            edge_flux.append(float(np.linalg.norm(sum((face_axis[f] for f in fs), np.zeros(3)))))
    return {
        "face_count": len(faces),
        "K_mean": float(np.mean(norms)) if len(norms) else 0.0,
        "K_p95": float(np.percentile(norms, 95)) if len(norms) else 0.0,
        "exact_residual_ratio": mean(exact_res),
        "closed_residual_ratio": mean(closed_res),
        "harmonic_ratio": mean(harm),
        "link_flux_mean": mean(edge_flux),
    }


def source_growth_summary(model: DynamicProvenanceGrowth) -> dict:
    completed = model.completed_parent_ids()
    zcur = []
    zbirth = []
    bias = []
    for p in completed:
        zcur.append(abs(model.neutral_for_parent(p, True)))
        zbirth.append(abs(model.neutral_for_parent(p, False)))
        bias.append(abs(model.local_cycle_log_bias(p)))
    return {
        "nodes": len(model.nodes),
        "birth_events": len(model.birth_events),
        "completed_parents": len(completed),
        "directed_edges": len(model.directed_edges),
        "mean_abs_neutral_current": mean(zcur),
        "mean_abs_neutral_birth": mean(zbirth),
        "mean_abs_cycle_log_bias": mean(bias),
    }


def analyze(max_level: int, mode: str, outdir: Path, closure_passes: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    controls = ["real_growth", "symmetrized_birth", "no_backreaction"]
    sources = ["record", "live", "handoff", "aging"]
    reductions = ["full", "diagonal", "trace_scalar"]
    rows: List[dict] = []
    growth_rows: List[dict] = []
    models: Dict[str, DynamicProvenanceGrowth] = {}
    for control in controls:
        model = DynamicProvenanceGrowth(mode=mode, growth_rule=control)
        model.grow(max_level)
        models[control] = model
        growth_rows.append({"control": control, **source_growth_summary(model)})
        complexes = [
            build_parent_fan_tetra_complex(model, max_level),
            build_dynamic_outward_ngf_complex(model, closure_passes=0),
            build_dynamic_outward_ngf_complex(model, closure_passes=closure_passes),
        ]
        for K in complexes:
            topo = topology(K)
            link = edge_link_metrics(K)
            for source in sources:
                for red in reductions:
                    cm = cochain_metrics(K, model, source, red)
                    row = {"control": control, "geometry": K.name, "source": source, "reduction": red, **topo, **link, **cm}
                    rows.append(row)
    write_csv(outdir / "growth_summary.csv", growth_rows)
    write_csv(outdir / "geometry_operator_summary.csv", rows)
    full_index = {(r["control"], r["geometry"], r["source"]): r for r in rows if r["reduction"] == "full"}
    kill_rows = []
    for r in rows:
        if r["reduction"] == "full":
            continue
        f = full_index.get((r["control"], r["geometry"], r["source"]))
        if f:
            kill_rows.append({
                "control": r["control"], "geometry": r["geometry"], "source": r["source"], "reduction": r["reduction"],
                "K_remaining_fraction": r["K_mean"] / (f["K_mean"] + EPS),
                "harmonic_remaining_fraction": r["harmonic_ratio"] / (f["harmonic_ratio"] + EPS),
            })
    write_csv(outdir / "operator_kill_controls.csv", kill_rows)
    primary = [r for r in rows if r["control"] == "real_growth" and r["source"] == "live" and r["reduction"] == "full"]
    result = {"max_level": max_level, "mode": mode, "closure_passes": closure_passes, "growth": growth_rows, "primary": primary, "kill_controls": kill_rows}
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_docs(pkg: Path, result: dict) -> None:
    primary_lines = []
    for r in result["primary"]:
        primary_lines.append(
            f"- {r['geometry']}: V/E/F/T={r['vertices']}/{r['edges']}/{r['faces']}/{r['tets']}, "
            f"boundary={r['boundary_fraction']:.3f}, saturated={r['saturated_fraction']:.3f}, "
            f"beta=({r['beta0']},{r['beta1']},{r['beta2']},{r['beta3']}), "
            f"edge_link_cycles={r['edge_link_cycle_fraction']:.3f}, K_mean={r['K_mean']:.6g}, "
            f"harmonic={r['harmonic_ratio']:.6g}, exact_res={r['exact_residual_ratio']:.6g}, closed_res={r['closed_residual_ratio']:.6g}"
        )
    growth_lines = []
    for g in result["growth"]:
        growth_lines.append(
            f"- {g['control']}: nodes={g['nodes']}, completed={g['completed_parents']}, "
            f"directed_edges={g['directed_edges']}, mean_abs_neutral_current={g['mean_abs_neutral_current']:.6g}, "
            f"mean_abs_cycle_log_bias={g['mean_abs_cycle_log_bias']:.6g}"
        )
    summary = f"""# SUMMARY

This package is the corrected integration test after the CQNM topology-only audit.

It explicitly combines:

1. Script-1/2-style dynamic provenance growth: parent-line plus older siblings are sensed at birth; the newborn then backreacts to ancestors and older siblings.
2. Root-inside, outward primal geometry: new geometric vertices grow away from the root with an order-dependent transverse offset.
3. Dynamic active-face NGF attachment: the tree is not used as the space; birth events select active boundary faces in a primal 3-complex.
4. CQNM/s=-1-style occupancy gate: faces are allowed occupancy <= 2; the closure attempt is marked as a heuristic comparison, not as a derived CNNA theorem.
5. Real operator commutator diagnostic: K_abc=[A_ab,A_bc] is computed from symmetric Record/Live/Handoff/Aging vertex operators; diagonal and trace-scalar controls are included.

## Model classification

- `parent_fan_tetra`: old local parent+three-children tetrahedron control; it is expected to close local boundaries too fast.
- `dynamic_outward_ngf`: dynamic active-face outward growth; provenance-derived attachment, boundary remains large.
- `dynamic_sminus1_closure_attempt`: same outward growth plus a heuristic s=-1 saturation/closure pass. This is not yet an internally derived CQNM closure law.

## Primary real-growth/live/full results

""" + "\n".join(primary_lines) + "\n\n## Growth controls\n\n" + "\n".join(growth_lines) + "\n"
    (pkg / "SUMMARY.md").write_text(summary, encoding="utf-8")
    results = f"""# RESULTS

## Verdict

The previous topology-only script did not contain all non-obstructed strands. This package is a stricter merger of the still-useful threads.

What is now contained:

- dynamic birth/backreaction and sibling-order asymmetry;
- the transverse geometric offset of ternary sibling fans;
- root-inside/outward primal growth;
- active-face NGF attachment with face occupancy;
- a CQNM/s=-1-style saturation gate as a marked comparison rule;
- the operatorial plaquette commutator K=[A_ab,A_bc], not a synthetic scalar/vector K placeholder;
- record/live/handoff/aging source modes;
- diagonal and trace-scalar kill controls;
- topology, boundary, saturation, edge-link, exactness, closedness, and harmonic-rest diagnostics.

What is still not proven/implemented:

- The s=-1 closure rule is not yet derived from CNNA response dynamics; it is a heuristic closure attempt.
- The vertex operators are DtN-like deterministic surrogates, not imported from the full earlier dynamic DtN package.
- No complex Hilbert space, C*-norm, GNS construction, AQFT net, or global J-lock is claimed.

## Numerical run

```json
{json.dumps(result, indent=2)}
```

## Interpretation

A positive Stage-4 result would require all of the following at once:

```text
low boundary fraction
high face saturation
nontrivial link cycles or nontrivial Betti support
nonzero K from full live/record/handoff operators
large non-exact/harmonic component
collapse under diagonal/trace controls
suppression under symmetrized/no-backreaction controls
```

If `dynamic_outward_ngf` remains boundary-dominated and `parent_fan_tetra` remains locally closing/trivializing, that confirms the corrected obstruction reading: the tree was not the problem; the missing piece is a derived law that turns provenance into the right saturated primal geometry.
"""
    (pkg / "RESULTS.md").write_text(results, encoding="utf-8")
    audit = """# Source-thread audit for scripts 1-40

## Fully represented in this integrated script

- Scripts 1-2: dynamic birth conductance, older-sibling sensing, ancestor/sibling backreaction, neutral phasor, directed circulation, and selected Z3/J-sector diagnostic.
- Scripts 33-35: real plaquette commutator K=[A_ab,A_bc], orientation reversal, diagonal/trace controls, parent-fan local skew/J candidate.
- Script 40: parent+three-children tetrahedron is retained as an obstruction control, not as the final geometry rule.

## Partially represented

- Scripts 8-17 and 36-39: gluing, holonomy, phase density, nonflat response mismatch are represented only by edge-link flux, exactness/closedness/harmonic projection, and link-cycle metrics. The exact old phase-density pipeline is not copied.
- Scripts 18-22: record/live/two-layer and dynamic DtN are represented by deterministic Record/Live/Handoff/Aging vertex operators. This is not the full previous DtN refresh implementation.
- Scripts 24-32: real local operator-system closure and adjoint/port closure are represented only as operator-source/reduction controls. No *-algebra closure is claimed.

## Obstructed controls intentionally kept

- Provenance-only/tree-as-space is not promoted to geometry.
- SG-like/parent-fan tetrahedral closure is retained as a negative control.
- Naive outward NGF is retained as a boundary-dominated control.

## Remaining missing derived step

The main missing theorem/test is a CNNA-derived closure law:

```text
Script-1/2 provenance + dynamic DtN/response data
  => active-face choice and s=-1 saturation rule
  => closed/nontrivial primal geometry
```

Until that is derived, `dynamic_sminus1_closure_attempt` must be read as a comparison gate, not as CNNA ontology.
"""
    (pkg / "SOURCE_AUDIT_1_40.md").write_text(audit, encoding="utf-8")
    readme = """# CNNA dynamic outward CQNM/DtN plaquette frustration package

Run:

```bash
python3 test_cqnm_dynamic_outward_growth_dtn_plaquette_frustration.py --max-level 4 --mode linear --closure-passes 2 --outdir out
```

Outputs:

- `out/growth_summary.csv`
- `out/geometry_operator_summary.csv`
- `out/operator_kill_controls.csv`
- `out/summary.json`
- `SUMMARY.md`
- `RESULTS.md`
- `SOURCE_AUDIT_1_40.md`

The package is a diagnostic Python surrogate, not a Lean proof and not a physical claim.
"""
    (pkg / "README.md").write_text(readme, encoding="utf-8")



# ---------------------------------------------------------------------------
# Response-forced saturation extension
# ---------------------------------------------------------------------------


def face_directed_imbalance(model: DynamicProvenanceGrowth, face: Face) -> float:
    vals: List[float] = []
    vs = list(face)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = vs[i], vs[j]
            wab = model.directed_edges.get((a, b), 0.0)
            wba = model.directed_edges.get((b, a), 0.0)
            vals.append(abs(wab - wba) / (wab + wba + EPS))
    return mean(vals)


def face_live_aging(model: DynamicProvenanceGrowth, face: Face) -> float:
    return mean(max(0.0, model.nodes[v].g - model.nodes[v].birth_g) for v in face)


def face_response_pressure(model: DynamicProvenanceGrowth, face: Face, source: str = "live") -> dict:
    gs = [model.nodes[v].g for v in face]
    births = [model.nodes[v].birth_g for v in face]
    k_norm = fro(face_K(model, face, source, "full"))
    imbalance = face_directed_imbalance(model, face)
    aging = face_live_aging(model, face)
    spread = float(np.std(gs))
    birth_spread = float(np.std(births))
    score = k_norm + 0.45 * imbalance + 0.35 * aging + 0.15 * spread + 0.08 * birth_spread
    return {
        "face_k_norm": k_norm,
        "face_directed_imbalance": imbalance,
        "face_live_aging": aging,
        "face_g_spread": spread,
        "face_birth_spread": birth_spread,
        "face_response_score": score,
    }


def parent_face_coupling(model: DynamicProvenanceGrowth, parent: int, face: Face) -> float:
    vals = []
    for v in face:
        vals.append(model.directed_edges.get((parent, v), 0.0) + model.directed_edges.get((v, parent), 0.0))
    return mean(vals)


def choose_response_forced_boundary_face_for_parent(
    model: DynamicProvenanceGrowth,
    faces_by_vertex: Dict[int, set[Face]],
    occ: Dict[Face, int],
    parent: int,
    child: int,
    decision_rows: Optional[List[dict]] = None,
    phase: str = "birth_attachment",
) -> Optional[Face]:
    local = [f for f in faces_by_vertex.get(parent, set()) if occ.get(f, 0) == 1]
    if not local:
        all_boundary = [f for fs in faces_by_vertex.values() for f in fs if occ.get(f, 0) == 1]
        local = sorted(set(all_boundary))
    if not local:
        return None
    child_node = model.nodes[child]
    best_face: Optional[Face] = None
    best_score = -1e300
    best_meta: dict = {}
    for f in sorted(local):
        rp = face_response_pressure(model, f, "live")
        centroid = sum((model.nodes[v].pos for v in f), np.zeros(3)) / 3.0
        radial_alignment = float(np.dot(unit(centroid), child_node.radial))
        coupling = parent_face_coupling(model, parent, f)
        age_match = -abs(mean(model.nodes[v].birth_time for v in f) - child_node.birth_time) / (child_node.birth_time + 1.0)
        # Response terms dominate; geometry only regularizes the active face choice.
        score = rp["face_response_score"] + 0.55 * coupling + 0.12 * radial_alignment + 0.06 * age_match
        if score > best_score:
            best_score = score
            best_face = f
            best_meta = {**rp, "parent_face_coupling": coupling, "radial_alignment": radial_alignment, "age_match": age_match, "score": score}
    if decision_rows is not None and best_face is not None:
        decision_rows.append({
            "phase": phase,
            "parent": parent,
            "child_or_candidate": child,
            "face": str(tuple(best_face)),
            **best_meta,
        })
    return best_face


def candidate_vertex_response_score(model: DynamicProvenanceGrowth, face: Face, x: int) -> dict:
    Sf = sum((vertex_operator(model, v, "live") for v in face), np.zeros((3, 3))) / 3.0
    Sx = vertex_operator(model, x, "live")
    mismatch = fro(Sx - Sf)
    coupling_vals = []
    for v in face:
        coupling_vals.append(model.directed_edges.get((v, x), 0.0) + model.directed_edges.get((x, v), 0.0))
    coupling = mean(coupling_vals)
    face_centroid = sum((model.nodes[v].pos for v in face), np.zeros(3)) / 3.0
    distance = float(np.linalg.norm(model.nodes[x].pos - face_centroid))
    live_aging = max(0.0, model.nodes[x].g - model.nodes[x].birth_g)
    birth_gap = abs(model.nodes[x].birth_time - mean(model.nodes[v].birth_time for v in face))
    return {
        "candidate_operator_mismatch": mismatch,
        "candidate_face_coupling": coupling,
        "candidate_distance": distance,
        "candidate_live_aging": live_aging,
        "candidate_birth_gap": birth_gap,
    }


def build_response_forced_ngf_complex(
    model: DynamicProvenanceGrowth,
    *,
    saturation_passes: int = 0,
    random_control: bool = False,
    seed: int = 17,
) -> Tuple[SimplicialComplex, List[dict]]:
    name = "response_forced_outward_ngf"
    if saturation_passes > 0:
        name = "response_forced_sminus1_saturation" if not random_control else "random_saturation_control"
    K = SimplicialComplex(name)
    occ: Dict[Face, int] = {}
    faces_by_vertex: Dict[int, set[Face]] = defaultdict(set)
    decision_rows: List[dict] = []
    root_seeded = False
    rng = random.Random(seed)

    for ev in model.birth_events:
        parent = int(ev["parent"])
        child = int(ev["child"])
        if not root_seeded and len(model.nodes[model.root].children) == 3:
            ch = model.child_ids_ordered(model.root)
            tet = tuple(sorted((model.root, ch[0], ch[1], ch[2])))
            if K.add_tet(tet, birth_time=max(model.nodes[c].birth_time for c in ch)):
                update_face_maps(K, faces_by_vertex, occ, tet)
                decision_rows.append({"phase": "root_seed", "parent": model.root, "child_or_candidate": -1, "face": "root_fan", "score": 0.0})
            root_seeded = True
        if child in K.vertices:
            continue
        if random_control:
            candidates = [f for fs in faces_by_vertex.values() for f in fs if occ.get(f, 0) == 1]
            face = rng.choice(candidates) if candidates else None
        else:
            face = choose_response_forced_boundary_face_for_parent(model, faces_by_vertex, occ, parent, child, decision_rows)
        if face is None:
            continue
        tet = tuple(sorted((*face, child)))
        if any(occ.get(f, 0) >= 2 for f in faces_of_tet(tet)):
            continue
        if K.add_tet(tet, birth_time=int(ev["t"])):
            update_face_maps(K, faces_by_vertex, occ, tet)

    if saturation_passes > 0:
        if random_control:
            random_saturation(model, K, saturation_passes, decision_rows, rng)
        else:
            response_forced_saturation(model, K, saturation_passes, decision_rows)
    return K, decision_rows


def response_forced_saturation(model: DynamicProvenanceGrowth, K: SimplicialComplex, passes: int, decision_rows: List[dict]) -> None:
    max_boundary_faces = 240
    max_vertices_per_face = 30
    max_new_tets_per_pass = 48
    for pidx in range(passes):
        changed = False
        occ = K.face_occupancy()
        boundary = [f for f, n in occ.items() if n == 1]
        boundary.sort(key=lambda f: face_response_pressure(model, f, "live")["face_response_score"], reverse=True)
        boundary = boundary[:max_boundary_faces]
        vertices = sorted(K.vertices)
        candidates: List[Tuple[float, Tet, dict]] = []
        for f in boundary:
            rp = face_response_pressure(model, f, "live")
            fset = set(f)
            ranked_x: List[Tuple[float, int, dict]] = []
            for x in vertices:
                if x in fset:
                    continue
                cv = candidate_vertex_response_score(model, f, x)
                local_score = (
                    1.00 * cv["candidate_face_coupling"]
                    + 0.55 * cv["candidate_operator_mismatch"]
                    + 0.20 * cv["candidate_live_aging"]
                    - 0.055 * cv["candidate_distance"]
                    - 0.010 * cv["candidate_birth_gap"]
                )
                ranked_x.append((local_score, x, cv))
            ranked_x.sort(key=lambda z: z[0], reverse=True)
            for sx, x, cv in ranked_x[:max_vertices_per_face]:
                tet = tuple(sorted((*f, x)))
                if tet in K.tets:
                    continue
                fs = faces_of_tet(tet)
                if any(occ.get(ff, 0) >= 2 for ff in fs):
                    continue
                gain = sum(1 for ff in fs if occ.get(ff, 0) == 1)
                # gain=1 only saturates the chosen face but opens too much new boundary; reject.
                if gain < 2:
                    continue
                score = 1.25 * rp["face_response_score"] + sx + 0.30 * gain
                candidates.append((score, tet, {**rp, **cv, "gain": gain, "score": score, "pass": pidx}))
        candidates.sort(key=lambda z: z[0], reverse=True)
        for score, tet, meta in candidates[:max_new_tets_per_pass]:
            occ_now = K.face_occupancy()
            if tet in K.tets:
                continue
            if any(occ_now.get(ff, 0) >= 2 for ff in faces_of_tet(tet)):
                continue
            if K.add_tet(tet, birth_time=max(model.nodes[v].birth_time for v in tet)):
                changed = True
                base_face = None
                for ff in faces_of_tet(tet):
                    if occ_now.get(ff, 0) == 1:
                        base_face = ff
                        break
                decision_rows.append({
                    "phase": "response_forced_saturation",
                    "parent": -1,
                    "child_or_candidate": -1,
                    "tet": str(tuple(tet)),
                    "face": str(tuple(base_face)) if base_face else "",
                    **meta,
                })
        if not changed:
            break


def random_saturation(model: DynamicProvenanceGrowth, K: SimplicialComplex, passes: int, decision_rows: List[dict], rng: random.Random) -> None:
    max_new_tets_per_pass = 32
    for pidx in range(passes):
        changed = False
        occ = K.face_occupancy()
        boundary = [f for f, n in occ.items() if n == 1]
        rng.shuffle(boundary)
        vertices = sorted(K.vertices)
        added = 0
        for f in boundary:
            xs = [x for x in vertices if x not in set(f)]
            rng.shuffle(xs)
            for x in xs[:24]:
                tet = tuple(sorted((*f, x)))
                if tet in K.tets:
                    continue
                occ_now = K.face_occupancy()
                fs = faces_of_tet(tet)
                if any(occ_now.get(ff, 0) >= 2 for ff in fs):
                    continue
                gain = sum(1 for ff in fs if occ_now.get(ff, 0) == 1)
                if gain < 2:
                    continue
                if K.add_tet(tet, birth_time=max(model.nodes[v].birth_time for v in tet)):
                    decision_rows.append({"phase": "random_saturation", "parent": -1, "child_or_candidate": x, "face": str(tuple(f)), "tet": str(tuple(tet)), "gain": gain, "pass": pidx, "score": 0.0})
                    changed = True
                    added += 1
                    break
            if added >= max_new_tets_per_pass:
                break
        if not changed:
            break


def analyze(max_level: int, mode: str, outdir: Path, closure_passes: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    controls = ["real_growth", "symmetrized_birth", "no_backreaction"]
    sources = ["record", "live", "handoff", "aging"]
    reductions = ["full", "diagonal", "trace_scalar"]
    rows: List[dict] = []
    growth_rows: List[dict] = []
    decision_rows: List[dict] = []
    for control in controls:
        model = DynamicProvenanceGrowth(mode=mode, growth_rule=control)
        model.grow(max_level)
        growth_rows.append({"control": control, **source_growth_summary(model)})
        parent_fan = build_parent_fan_tetra_complex(model, max_level)
        response_ngf, dec0 = build_response_forced_ngf_complex(model, saturation_passes=0, random_control=False)
        response_sat, dec1 = build_response_forced_ngf_complex(model, saturation_passes=closure_passes, random_control=False)
        random_sat, dec2 = build_response_forced_ngf_complex(model, saturation_passes=closure_passes, random_control=True, seed=101 + len(control))
        for d in dec0 + dec1 + dec2:
            d["control"] = control
            decision_rows.append(d)
        complexes = [parent_fan, response_ngf, response_sat, random_sat]
        for K in complexes:
            topo = topology(K)
            link = edge_link_metrics(K)
            for source in sources:
                for red in reductions:
                    cm = cochain_metrics(K, model, source, red)
                    rows.append({"control": control, "geometry": K.name, "source": source, "reduction": red, **topo, **link, **cm})
    write_csv(outdir / "growth_summary.csv", growth_rows)
    write_csv(outdir / "geometry_operator_summary.csv", rows)
    write_csv(outdir / "face_saturation_decisions.csv", decision_rows)
    full_index = {(r["control"], r["geometry"], r["source"]): r for r in rows if r["reduction"] == "full"}
    kill_rows = []
    for r in rows:
        if r["reduction"] == "full":
            continue
        f = full_index.get((r["control"], r["geometry"], r["source"]))
        if f:
            kill_rows.append({
                "control": r["control"], "geometry": r["geometry"], "source": r["source"], "reduction": r["reduction"],
                "K_remaining_fraction": r["K_mean"] / (f["K_mean"] + EPS),
                "harmonic_remaining_fraction": r["harmonic_ratio"] / (f["harmonic_ratio"] + EPS),
                "exact_residual_remaining_fraction": r["exact_residual_ratio"] / (f["exact_residual_ratio"] + EPS),
            })
    write_csv(outdir / "operator_kill_controls.csv", kill_rows)
    primary = [r for r in rows if r["control"] == "real_growth" and r["source"] == "live" and r["reduction"] == "full"]
    control_primary = [r for r in rows if r["source"] == "live" and r["reduction"] == "full"]
    gate = evaluate_response_forced_gate(primary, control_primary, kill_rows)
    result = {
        "test_name": "test_cqnm_response_forced_face_saturation",
        "max_level": max_level,
        "mode": mode,
        "response_forced_saturation_passes": closure_passes,
        "growth": growth_rows,
        "primary": primary,
        "gate": gate,
        "decision_row_count": len(decision_rows),
        "kill_controls": kill_rows,
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_response_forced_gate(primary: List[dict], control_primary: List[dict], kill_rows: List[dict]) -> dict:
    by_geom = {r["geometry"]: r for r in primary}
    sat = by_geom.get("response_forced_sminus1_saturation", {})
    ngf = by_geom.get("response_forced_outward_ngf", {})
    parent = by_geom.get("parent_fan_tetra", {})
    random_row = by_geom.get("random_saturation_control", {})
    sat_vs_ngf_boundary_reduced = sat.get("boundary_fraction", 1.0) < ngf.get("boundary_fraction", 1.0)
    sat_creates_link_cycles = sat.get("edge_link_cycle_fraction", 0.0) > max(ngf.get("edge_link_cycle_fraction", 0.0), parent.get("edge_link_cycle_fraction", 0.0))
    sat_has_operator_k = sat.get("K_mean", 0.0) > 1e-9
    sat_has_harmonic = sat.get("harmonic_ratio", 0.0) > 1e-3
    sat_has_nontrivial_betti = any(int(sat.get(k, 0)) > (1 if k == "beta0" else 0) for k in ["beta1", "beta2", "beta3"])
    random_not_cleaner = sat.get("boundary_fraction", 1.0) <= random_row.get("boundary_fraction", 1.0) + 1e-9
    sym = [r for r in control_primary if r["control"] == "symmetrized_birth" and r["geometry"] == "response_forced_sminus1_saturation"]
    no_br = [r for r in control_primary if r["control"] == "no_backreaction" and r["geometry"] == "response_forced_sminus1_saturation"]
    real = sat
    real_exceeds_sym_or_nobr_K = bool(real) and all(real.get("K_mean", 0.0) >= x.get("K_mean", 0.0) for x in sym + no_br)
    return {
        "boundary_reduced_vs_outward_ngf": sat_vs_ngf_boundary_reduced,
        "creates_more_edge_link_cycles": sat_creates_link_cycles,
        "operator_K_nonzero": sat_has_operator_k,
        "harmonic_rest_nonzero_threshold_1e_minus_3": sat_has_harmonic,
        "nontrivial_global_betti": sat_has_nontrivial_betti,
        "cleaner_or_equal_boundary_than_random_control": random_not_cleaner,
        "real_growth_K_ge_sym_no_backreaction": real_exceeds_sym_or_nobr_K,
        "stage4_candidate_strong": bool(sat_vs_ngf_boundary_reduced and sat_creates_link_cycles and sat_has_operator_k and sat_has_harmonic and sat_has_nontrivial_betti and random_not_cleaner and real_exceeds_sym_or_nobr_K),
        "interpretation": "Positive local/saturation gate only if boundary/link/operator conditions hold. Strong Stage-4 requires nontrivial Betti plus harmonic rest; otherwise the closure law remains insufficient."
    }


def make_docs(pkg: Path, result: dict) -> None:
    primary_lines = []
    for r in result["primary"]:
        primary_lines.append(
            f"- {r['geometry']}: V/E/F/T={r['vertices']}/{r['edges']}/{r['faces']}/{r['tets']}, "
            f"boundary={r['boundary_fraction']:.3f}, saturated={r['saturated_fraction']:.3f}, "
            f"beta=({r['beta0']},{r['beta1']},{r['beta2']},{r['beta3']}), "
            f"edge_link_cycles={r['edge_link_cycle_fraction']:.3f}, K_mean={r['K_mean']:.6g}, "
            f"harmonic={r['harmonic_ratio']:.6g}, exact_res={r['exact_residual_ratio']:.6g}, closed_res={r['closed_residual_ratio']:.6g}"
        )
    gate_lines = [f"- {k}: {v}" for k, v in result["gate"].items()]
    growth_lines = []
    for g in result["growth"]:
        growth_lines.append(
            f"- {g['control']}: nodes={g['nodes']}, completed={g['completed_parents']}, "
            f"directed_edges={g['directed_edges']}, neutral_current={g['mean_abs_neutral_current']:.6g}, "
            f"cycle_log_bias={g['mean_abs_cycle_log_bias']:.6g}"
        )
    summary = f"""# SUMMARY

Package: `test_cqnm_response_forced_face_saturation.py`

Purpose: replace the previous heuristic closure pass by a response-forced active-face/saturation rule. The tree remains provenance, not space. The root is inside; vertices grow outward with sibling-order transverse offset. Boundary faces are selected by local response quantities: live DtN-like K norm, directed imbalance, aging/backreaction, conductance spread, parent-face coupling, and candidate operator mismatch.

## Model status

This is still a Python diagnostic surrogate, not a Lean proof and not a derived CNNA theorem. The crucial improvement is that the closure choice is no longer purely geometric/random; it is forced by locally available response/provenance data.

## Primary real-growth/live/full results

""" + "\n".join(primary_lines) + "\n\n## Gate evaluation\n\n" + "\n".join(gate_lines) + "\n\n## Growth controls\n\n" + "\n".join(growth_lines) + "\n"
    (pkg / "SUMMARY.md").write_text(summary, encoding="utf-8")
    results = f"""# RESULTS

## Verdict

The test advances the previous package by making the boundary-face saturation decision depend on local Response/DtN/backreaction data rather than on a generic closure heuristic.

The test is positive only in a limited sense if response-forced saturation reduces boundary and creates more edge-link cycles while preserving nonzero operator K. It is a strong Stage-4 candidate only if it also creates nontrivial global Betti support and a non-negligible harmonic/non-exact K component.

## Numerical result

```json
{json.dumps(result, indent=2)}
```

## Interpretation discipline

Read `stage4_candidate_strong=false` as a real obstruction, not as failure of the test. It means the response-forced local saturation law is not yet enough to produce a non-exact global carrier. That would localize the next missing ingredient to the closure dynamics/topological growth rule, not to the tree-provenance mechanism.

Read `stage4_candidate_strong=true` only as a Python-level candidate gate, not as a CNNA theorem. It would still need replacement of the surrogate DtN operators by the full dynamic DtN pipeline and later Lean formalization of the discrete growth law.
"""
    (pkg / "RESULTS.md").write_text(results, encoding="utf-8")
    audit = """# SOURCE_AUDIT_1_40

This package explicitly carries forward the non-obstructed strands:

- Script 1/2: dynamic birth, older-sibling sensing, ancestor/sibling backreaction, transverse sibling-order offset, directed circulation controls.
- Script 35: genuine operatorial plaquette commutator K=[A_ab,A_bc], with diagonal/trace controls.
- Script 40: parent-fan tetrahedron is retained only as an obstruction control.

The new contribution relative to the previous package is response-forced face saturation:

```text
boundary face score = K_norm + directed imbalance + aging/backreaction + conductance spread + coupling
candidate score     = face coupling + operator mismatch + live aging - distance/birth-gap regularizers
```

So the closure decision is no longer arbitrary geometry-only closure. It is still a surrogate rule and must be audited further.

Missing:

- full previous dynamic DtN refresh implementation;
- formal proof that the score is the unique/canonical CNNA-derived saturation rule;
- C*- / GNS / AQFT interpretation;
- Lean formalization.
"""
    (pkg / "SOURCE_AUDIT_1_40.md").write_text(audit, encoding="utf-8")
    readme = """# Response-forced CQNM face saturation test

Run:

```bash
python3 test_cqnm_response_forced_face_saturation.py --max-level 4 --mode linear --closure-passes 2 --outdir out --package pkg
```

Outputs:

- `growth_summary.csv`
- `geometry_operator_summary.csv`
- `face_saturation_decisions.csv`
- `operator_kill_controls.csv`
- `summary.json`
- `SUMMARY.md`
- `RESULTS.md`
- `SOURCE_AUDIT_1_40.md`

The test is a diagnostic surrogate. It does not claim a derived CNNA theorem.
"""
    (pkg / "README.md").write_text(readme, encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--mode", choices=["linear", "log", "saturating"], default="linear")
    ap.add_argument("--closure-passes", type=int, default=2)
    ap.add_argument("--outdir", type=Path, default=Path("response_forced_face_saturation_out"))
    ap.add_argument("--package", type=Path, default=None)
    args = ap.parse_args()
    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    result = analyze(args.max_level, args.mode, args.outdir, args.closure_passes)
    print(json.dumps(result, indent=2))
    if args.package is not None:
        pkg = args.package
        if pkg.exists():
            shutil.rmtree(pkg)
        pkg.mkdir(parents=True)
        shutil.copy2(Path(__file__), pkg / Path(__file__).name)
        shutil.copytree(args.outdir, pkg / args.outdir.name)
        make_docs(pkg, result)
        zip_path = pkg.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(pkg.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(pkg.parent))


if __name__ == "__main__":
    main()
