
# stgraph - Sierpinski-Tetrahedron Graph + Quantum Toolkit

**Short version:** A small, standalone Python package for the **ST-Graph§** (IFS-Generator,
Address tree, simple renormalization) plus **quantum tools** (Hilbert space, partial trace,
Liouville-von-Neumann, Lindblad damping).

> Attention: Quantum operations scale exponentially with the number of nodes N (2^N).
> For real dynamics, please use only very small subgraphs/blocks.

## Installation (local)

```bash
pip install -U networkx numpy
§§X2§§Paket ist "flat"; einfach importieren, wenn der Ordner im PYTHONPATH liegt:
§§X3§§import sys; sys.path.append("§§X22§§/stgraph")
```

## Quick start

```python
import networkx as nx
import numpy as np
from stgraph import (
    build_graph_with_addresses, build_address_tree, blocks_from_prefix,
    coarse_grain_graph, qubit_layout, build_xx_hamiltonian, basis_state,
    lvne_rk4, reduce_to_nodes, partial_trace
)

§§X5§§1) ST-Graph mit Adressen
G = build_graph_with_addresses(level=2)
addr_tree = build_address_tree(G)

§§X6§§2) Blöcke aus Präfix (Renormierung)
blocks = blocks_from_prefix(G, prefix_len=2)
H_coarse = coarse_grain_graph(G, prefix_len=2)

§§X7§§3) Quanten‑Demo auf kleinem Subgraphen (z.B. die ersten 3 Knoten)
nodes = sorted(list(G.nodes()))[:3]
edges = [(u,v) for (u,v) in G.edges() if u in nodes and v in nodes]
H = build_xx_hamiltonian(nodes, edges, J=1.0)

§§X8§§|000>-Zustand
psi0 = basis_state([0,0,0])
rho0 = psi0 @ psi0.conj().T

§§X9§§4) LVN-Evolution (unitär; falls Lindblad, siehe dynamics.add_dephasing_lindblad)
t = np.linspace(0, 5, 101)
rhos = lvne_rk4(rho0, H, t)

§§X10§§5) Ein‑Site Reduktion (partielle Spur): ρ_{site 0}
rho_site0 = reduce_to_nodes(rhos[-1], [0], [2,2,2])
print("Bloch‑Z von Site 0 =", np.trace(rho_site0 @ np.array([[1,0],[0,-1]])))
```

## Important modules

- `generator.build_graph(level)`: only positions, no address tree.
- `generator.build_graph_with_addresses(level)`: adds `addresses: set[str]` for each node.
- `address.build_address_tree(G)`: Map address → node list.
- `address.blocks_from_prefix(G, Lp)`: arranges nodes by prefix (block assignment).
- `renorm.coarse_grain_graph(G, Lp)`: Block graph (adjacency between blocks).
- `renorm.coarse_grain_state_average(rho, nodes, blocks)`: CPTP channel, which determines per block
  the one-site density matrices → coarse state as tensor product.
- `hilbert`: Pauli ops, embedding on sites, Ising/XX-Hamiltonian, random-ρ.
- `trace`: general partial trace; reduction to nodes/positions.
- `dynamics`: LVN right-hand side, RK4 integrator, unitary step evolution, dephasing.

## tracing of geometry / degrees of freedom

- **Geometry as block structure:** Use `blocks_from_prefix` to define *Geometry cells* (prefix length = scale)
  (prefix length = scale). With `coarse_grain_state_average` you then obtain a
  reduced density matrix **per cell** (geometry "untracked").
- **Explicit degrees of freedom:** With `trace.reduce_to_nodes` or `trace_out_nodes` you can
  any sites (nodes).

## Limits / TODO

- Effective Hamiltonians under renormalization (block spin, isometries) are **placeholders**.
  Currently only the **§graph** is coarsely scaled and **states§** are reduced via CPTP averaging.
- The fully quantized treatment is not practicable for large N. Use subgraphs.

## License

© 2025 antaris - Code: MIT; Data/Images/Texts: CC BY 4.0


## NEW: (1) Effective Hamiltonian renormalization (block isometries)

```python
from stgraph import build_graph_with_addresses, blocks_from_prefix
from stgraph import build_xx_hamiltonian
from stgraph import renormalize_hamiltonian_by_blocks

§§X16§§Fine graph and small quantum model
G = build_graph_with_addresses(level=2)
nodes = sorted(G.nodes())[:6]  # 6 Qubits (Demo)
edges = [(u,v) for (u,v) in G.edges() if u in nodes and v in nodes]
H = build_xx_hamiltonian(nodes, edges, J=1.0)

§§X17§§Blocks via address prefix (e.g., 2 blocks from 6 nodes)
blocks = blocks_from_prefix(G, prefix_len=1)  # coarse cells
Heff, block_labels, block_members = renormalize_hamiltonian_by_blocks(H, nodes, blocks, method="parity")
print(Heff.shape)  # (2^M, 2^M), e.g. (4,4) for M=2 blocks
```

Methods for block isometry: `"parity"`, `"majority"`, `"magnetization"`.

## NEW: (2) Master equations with generic Lindblad channels (time-dependent)

```python
import numpy as np
from stgraph import lvne_rk4_t, lindblad_from_rates_z, build_xx_hamiltonian

§§X19§§H(t): hier zeitunabhängig
H0 = build_xx_hamiltonian(nodes, edges, J=1.0)
H_of_t = lambda t: H0

§§X20§§L_k(t): einfache Dephasings auf Sites [0,1]
Ls_of_t = lambda t: lindblad_from_rates_z(gamma_z=0.1, N=len(nodes), sites=[0,1])

rho_t = lvne_rk4_t(rho0, np.linspace(0,5,101), H_of_t, Ls_of_t)
```

## NEW: (3) Measurements (POVMs/projections) + logging

```python
from stgraph import projective_measure_z_on_sites, EventLogger

logger = EventLogger("st_events.jsonl")
probs, posts = projective_measure_z_on_sites(rho_t[-1], sites=[0,1], N=len(nodes))
logger.log("measurement", {"sites":[0,1], "probs": probs.tolist()})
```

File format: **§JSONL§**, one line per event.
```json
{"ts":"2025-08-28T19:17:00Z","type":"measurement","payload":{"sites":[0,1],"probs":[0.52,0.48,...]}}
```
