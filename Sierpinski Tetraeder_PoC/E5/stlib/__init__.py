
"""
stgraph — Sierpinski‑Tetrahedron Graph + Quantum Toolkit
========================================================

Features
--------
- Sierpinski‑Tetrahedron (ST) graph generator with *addresses* for nodes.
- Address tree utilities and fine↔coarse mappings.
- Simple graph renormalization (block by address prefix, coarse adjacency).
- Quantum utilities: Hilbert‑space layout, operators, partial trace.
- Liouville–von Neumann (unitary or Lindblad‑type dephasing) integrator.
- State coarse‑graining channels (block‑average of single‑site states).

Notes
-----
Quantum parts scale exponentially in the number of nodes N (d^N). Use for
small N (≲ 10 qubits) or restrict to subgraphs/blocks.

Public Modules
--------------
- generator: build ST graphs with node positions and address sets
- address:   address tree & mappings between levels
- renorm:    coarse‑graining of graphs and states
- hilbert:   qubit layouts, operators, Hamiltonians
- dynamics:  Liouville-von Neumann evolution (RK4), Lindblad dephasing
- trace:     general partial trace and node‑wise reductions
- io:        save/load helpers

License
-------
MIT (code); © 2025 antaris. See README.md for details.
"""
from .generator import build_graph, build_graph_with_addresses
from .address import build_address_tree, node_primary_address, blocks_from_prefix
from .renorm import coarse_grain_graph, coarse_grain_state_average
from .hilbert import (
    qubit_layout, kron_all, op_on_site, pauli, build_ising_hamiltonian,
    build_xx_hamiltonian, basis_state, random_density_matrix
)
from .trace import partial_trace, reduce_to_nodes, trace_out_nodes
from .dynamics import lvne_rk4, time_evolve_unitary, add_dephasing_lindblad
from . import io as stio

__all__ = [
    "build_graph","build_graph_with_addresses",
    "build_address_tree","node_primary_address","blocks_from_prefix",
    "coarse_grain_graph","coarse_grain_state_average",
    "qubit_layout","kron_all","op_on_site","pauli",
    "build_ising_hamiltonian","build_xx_hamiltonian","basis_state",
    "random_density_matrix",
    "partial_trace","reduce_to_nodes","trace_out_nodes",
    "lvne_rk4","time_evolve_unitary","add_dephasing_lindblad","stio"
]
__version__ = "0.1.0"

from .renorm_effective import (
    build_block_isometry, renormalize_hamiltonian_by_blocks, renormalize_operator_on_block
)
from .dynamics_td import lvne_rk4_t, lvne_rhs_t, lindblad_from_rates_z
from .measurement import (
    povm_measure_effects, povm_measure_kraus, projective_measure_z_on_sites, EventLogger
)

