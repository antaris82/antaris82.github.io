#!/usr/bin/env python3
"""
Dynamic birth-conductance test for the CNNA/NGF provenance-growth picture.

Purpose:
- This is a diagnostic Python surrogate, not a Lean theorem and not a physics claim.
- Geometry is NOT changed by conductance updates. Conductance is response/coupling data.
- A newborn child has no own UV-tail, but immediately acts as UV-tail/backreaction
  for its parent line up to the root.
- Siblings born sequentially are not equivalent: child k senses parent line plus
  older siblings; each birth updates already-grown ancestors and older siblings.

The model is intentionally small but event-resolved:
- every child birth is logged
- every completed sibling triple is tested for neutral phasor and directed cycle bias
- after every level a global summary is written
"""

from __future__ import annotations
import argparse
import csv
import math
import cmath
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np


@dataclass
class Node:
    id: int
    parent: int | None
    level: int
    birth_order: int
    birth_time: int
    birth_g: float
    g: float
    children: list[int] = field(default_factory=list)


class DynamicBirthConductanceModel:
    def __init__(
        self,
        mode: str = "linear",
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
        eps: float = 1e-12,
    ):
        if branching != 3:
            raise ValueError("This diagnostic currently assumes ternary sibling triples.")
        self.mode = mode
        self.branching = branching
        self.base = base
        self.alpha_env = alpha_env
        self.ancestor_decay = ancestor_decay
        self.br_ancestor = br_ancestor
        self.br_sibling = br_sibling
        self.eps = eps

        self.nodes: dict[int, Node] = {}
        self.t = 0
        self.next_id = 0
        self.directed_edges: dict[tuple[int, int], float] = defaultdict(float)
        self.event_rows: list[dict] = []
        self.level_rows: list[dict] = []

        root = self._new_node(parent=None, level=0, birth_order=0, birth_g=1.0)
        self.root = root.id

    def _new_node(self, parent: int | None, level: int, birth_order: int, birth_g: float) -> Node:
        n = Node(
            id=self.next_id,
            parent=parent,
            level=level,
            birth_order=birth_order,
            birth_time=self.t,
            birth_g=birth_g,
            g=birth_g,
        )
        self.nodes[n.id] = n
        self.next_id += 1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n

    def parent_line(self, parent: int) -> list[int]:
        line = []
        cur: int | None = parent
        while cur is not None:
            line.append(cur)
            cur = self.nodes[cur].parent
        return line

    def birth_environment_load(self, parent: int, older_siblings: list[int]) -> float:
        env = 0.0
        for d, a in enumerate(self.parent_line(parent), start=1):
            env += self.nodes[a].g * (self.ancestor_decay ** (d - 1))
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
        raise ValueError(f"unknown mode: {self.mode}")

    def neutral_phasor_values(self, vals: list[float]) -> complex:
        omega = cmath.exp(2j * math.pi / 3)
        return vals[0] + vals[1] * omega + vals[2] * (omega ** 2)

    def neutral_for_parent(self, parent: int, current: bool = True) -> complex | None:
        ch = self.nodes[parent].children
        if len(ch) != 3:
            return None
        vals = [self.nodes[c].g if current else self.nodes[c].birth_g for c in ch]
        return self.neutral_phasor_values(vals)

    def local_cycle_bias(self, parent: int) -> tuple[float | None, float | None, float | None]:
        ch = self.nodes[parent].children
        if len(ch) != 3:
            return None, None, None
        c1, c2, c3 = ch

        def w(u: int, v: int) -> float:
            return self.directed_edges.get((u, v), 0.0) + self.eps

        # Forward birth-cycle: 1->2, 2->3, 3->1.
        # Reverse: 1->3, 3->2, 2->1.
        fprod = w(c1, c2) * w(c2, c3) * w(c3, c1)
        rprod = w(c1, c3) * w(c3, c2) * w(c2, c1)
        return math.log(fprod / rprod), fprod, rprod

    def global_undirected_h1_rank(self) -> tuple[int, int, int]:
        und = set()
        for (u, v), w in self.directed_edges.items():
            if u != v and w > 0:
                und.add(tuple(sorted((u, v))))
        V = len(self.nodes)
        E = len(und)
        adj = defaultdict(list)
        for u, v in und:
            adj[u].append(v)
            adj[v].append(u)
        seen = set()
        comps = 0
        for n in self.nodes:
            if n in seen:
                continue
            comps += 1
            stack = [n]
            seen.add(n)
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
        return E - V + comps, E, comps

    def completed_parents(self) -> list[int]:
        return [n.id for n in self.nodes.values() if len(n.children) == 3]

    def add_child(self, parent: int, order: int, record_global_each_event: bool = False) -> int:
        self.t += 1
        older = list(self.nodes[parent].children)
        env_load = self.birth_environment_load(parent, older)
        birth_g = self.child_conductance_from_env(env_load)
        child = self._new_node(parent, self.nodes[parent].level + 1, order, birth_g)
        c = child.id

        # Environment influence: existing parent-line and older siblings -> newborn.
        total_env = env_load + self.eps
        for d, a in enumerate(self.parent_line(parent), start=1):
            contrib = self.nodes[a].g * (self.ancestor_decay ** (d - 1))
            self.directed_edges[(a, c)] += self.alpha_env * contrib / total_env * birth_g
        for s in older:
            contrib = self.nodes[s].g
            self.directed_edges[(s, c)] += self.alpha_env * contrib / total_env * birth_g

        # Backreaction: newborn acts as UV-tail for parent line.
        for d, a in enumerate(self.parent_line(parent), start=1):
            delta = self.br_ancestor * birth_g / (d * d)
            self.nodes[a].g += delta
            self.directed_edges[(c, a)] += delta

        # Backreaction to older siblings. This is the minimal "already-grown
        # cells are updated again" mechanism.
        for s in older:
            delta = self.br_sibling * birth_g
            self.nodes[s].g += delta
            self.directed_edges[(c, s)] += delta

        children = self.nodes[parent].children
        partial = [self.nodes[x].g for x in children]
        padded = partial + [0.0] * (3 - len(partial))
        z_partial = self.neutral_phasor_values(padded)

        completed = len(children) == 3
        z_abs = z_phase = z_norm = None
        z_birth_abs = z_birth_phase = z_birth_norm = None
        cycle_bias = fprod = rprod = None
        if completed:
            z = self.neutral_for_parent(parent, current=True)
            zb = self.neutral_for_parent(parent, current=False)
            assert z is not None and zb is not None
            mean_g = sum(self.nodes[x].g for x in children) / 3
            mean_birth_g = sum(self.nodes[x].birth_g for x in children) / 3
            z_abs = abs(z)
            z_phase = math.degrees(cmath.phase(z))
            z_norm = z_abs / mean_g
            z_birth_abs = abs(zb)
            z_birth_phase = math.degrees(cmath.phase(zb))
            z_birth_norm = z_birth_abs / mean_birth_g
            cycle_bias, fprod, rprod = self.local_cycle_bias(parent)

        if record_global_each_event:
            h1, und_e, comps = self.global_undirected_h1_rank()
        else:
            h1 = und_e = comps = None

        self.event_rows.append(
            {
                "mode": self.mode,
                "t": self.t,
                "parent": parent,
                "child": c,
                "child_level": child.level,
                "child_order": order,
                "older_siblings": " ".join(map(str, older)),
                "env_load": env_load,
                "child_birth_g": birth_g,
                "parent_g_after": self.nodes[parent].g,
                "partial_child_g_current": " ".join(f"{x:.12g}" for x in partial),
                "partial_neutral_abs": abs(z_partial),
                "triple_completed": int(completed),
                "triple_neutral_abs_current": z_abs,
                "triple_neutral_phase_current_deg": z_phase,
                "triple_neutral_norm_current": z_norm,
                "triple_neutral_abs_birth": z_birth_abs,
                "triple_neutral_phase_birth_deg": z_birth_phase,
                "triple_neutral_norm_birth": z_birth_norm,
                "cycle_log_bias_forward_vs_reverse": cycle_bias,
                "cycle_forward_product": fprod,
                "cycle_reverse_product": rprod,
                "global_undirected_H1_rank": h1,
                "global_undirected_edges": und_e,
                "global_components": comps,
            }
        )

        return c

    def grow_level(self, frontier: list[int]) -> list[int]:
        next_frontier = []
        for p in frontier:
            for k in range(1, 4):
                next_frontier.append(self.add_child(p, k))
        return next_frontier

    def level_summary(self, level: int) -> dict:
        completed = self.completed_parents()
        z_norms = []
        z_birth_norms = []
        phases = []
        biases = []
        for p in completed:
            z = self.neutral_for_parent(p, current=True)
            zb = self.neutral_for_parent(p, current=False)
            if z is None or zb is None:
                continue
            ch = self.nodes[p].children
            mean_g = sum(self.nodes[x].g for x in ch) / 3
            mean_bg = sum(self.nodes[x].birth_g for x in ch) / 3
            z_norms.append(abs(z) / mean_g)
            z_birth_norms.append(abs(zb) / mean_bg)
            phases.append(math.degrees(cmath.phase(z)))
            b, _, _ = self.local_cycle_bias(p)
            if b is not None:
                biases.append(b)

        h1, und_e, comps = self.global_undirected_h1_rank()
        gs = [n.g for n in self.nodes.values()]
        row = {
            "mode": self.mode,
            "level": level,
            "time": self.t,
            "nodes": len(self.nodes),
            "directed_edges": len(self.directed_edges),
            "undirected_edges": und_e,
            "undirected_H1_rank": h1,
            "completed_triples": len(completed),
            "mean_neutral_norm_current": float(np.mean(z_norms)) if z_norms else 0.0,
            "max_neutral_norm_current": float(np.max(z_norms)) if z_norms else 0.0,
            "mean_neutral_norm_birth": float(np.mean(z_birth_norms)) if z_birth_norms else 0.0,
            "mean_neutral_phase_current_deg": float(np.mean(phases)) if phases else 0.0,
            "std_neutral_phase_current_deg": float(np.std(phases)) if phases else 0.0,
            "mean_cycle_log_bias": float(np.mean(biases)) if biases else 0.0,
            "mean_abs_cycle_log_bias": float(np.mean(np.abs(biases))) if biases else 0.0,
            "min_g": min(gs),
            "max_g": max(gs),
            "mean_g": float(np.mean(gs)),
        }
        self.level_rows.append(row)
        return row

    def run(self, max_level: int) -> None:
        frontier = [self.root]
        self.level_summary(0)
        for level in range(1, max_level + 1):
            frontier = self.grow_level(frontier)
            self.level_summary(level)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=Path("dynamic_birth_conductance_out"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("linear", 0.22),
        ("log", 0.22),
        ("saturating", 0.90),
    ]

    all_level_rows = []
    summary_lines = []
    for mode, alpha in configs:
        model = DynamicBirthConductanceModel(mode=mode, alpha_env=alpha)
        model.run(args.max_level)
        write_csv(args.outdir / f"events_{mode}.csv", model.event_rows)
        write_csv(args.outdir / f"levels_{mode}.csv", model.level_rows)
        all_level_rows.extend(model.level_rows)

        root_children = model.nodes[model.root].children
        root_birth = [model.nodes[x].birth_g for x in root_children]
        root_current = [model.nodes[x].g for x in root_children]
        root_z = model.neutral_for_parent(model.root, current=True)
        root_bias, _, _ = model.local_cycle_bias(model.root)
        final = model.level_rows[-1]

        summary_lines.append(f"MODE {mode}")
        summary_lines.append(f"  root birth conductances: {[round(x,6) for x in root_birth]}")
        summary_lines.append(f"  root current conductances after L={args.max_level}: {[round(x,6) for x in root_current]}")
        summary_lines.append(f"  root current |Z|: {abs(root_z):.6f}, phase deg: {math.degrees(cmath.phase(root_z)):.3f}")
        summary_lines.append(f"  root local cycle log-bias forward/reverse: {root_bias:.6f}")
        summary_lines.append(
            "  final level: nodes={nodes}, H1_support={undirected_H1_rank}, "
            "mean neutral norm={mean_neutral_norm_current:.6f}, "
            "mean cycle log-bias={mean_cycle_log_bias:.6f}, "
            "g range=[{min_g:.6f}, {max_g:.6f}]".format(**final)
        )
        summary_lines.append("")

    write_csv(args.outdir / "all_level_summaries.csv", all_level_rows)
    (args.outdir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
