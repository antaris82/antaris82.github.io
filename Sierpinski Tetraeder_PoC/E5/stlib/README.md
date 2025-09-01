
# stgraph — Sierpinski‑Tetrahedron Graph + Quantum Toolkit

**Kurzfassung:** Ein kleines, eigenständiges Python‑Paket für den **ST‑Graphen** (IFS‑Generator,
Adresstree, einfache Renormierung) plus **Quanten‑Werkzeuge** (Hilbertraum, partielle Spur,
Liouville‑von‑Neumann, Lindblad‑Dämpfung).

> Achtung: Quanten‑Operationen skalieren exponentiell mit der Knotenzahl N (2^N).
> Für echte Dynamik bitte nur sehr kleine Subgraphen/Blöcke verwenden.

## Installation (lokal)

```bash
pip install -U networkx numpy
# Paket ist "flat"; einfach importieren, wenn der Ordner im PYTHONPATH liegt:
#   import sys; sys.path.append("<pfad-zu>/stgraph")
```

## Schnellstart

```python
import networkx as nx
import numpy as np
from stgraph import (
    build_graph_with_addresses, build_address_tree, blocks_from_prefix,
    coarse_grain_graph, qubit_layout, build_xx_hamiltonian, basis_state,
    lvne_rk4, reduce_to_nodes, partial_trace
)

# 1) ST-Graph mit Adressen
G = build_graph_with_addresses(level=2)
addr_tree = build_address_tree(G)

# 2) Blöcke aus Präfix (Renormierung)
blocks = blocks_from_prefix(G, prefix_len=2)
H_coarse = coarse_grain_graph(G, prefix_len=2)

# 3) Quanten‑Demo auf kleinem Subgraphen (z.B. die ersten 3 Knoten)
nodes = sorted(list(G.nodes()))[:3]
edges = [(u,v) for (u,v) in G.edges() if u in nodes and v in nodes]
H = build_xx_hamiltonian(nodes, edges, J=1.0)

# |000>-Zustand
psi0 = basis_state([0,0,0])
rho0 = psi0 @ psi0.conj().T

# 4) LVN-Evolution (unitär; falls Lindblad, siehe dynamics.add_dephasing_lindblad)
t = np.linspace(0, 5, 101)
rhos = lvne_rk4(rho0, H, t)

# 5) Ein‑Site Reduktion (partielle Spur): ρ_{site 0}
rho_site0 = reduce_to_nodes(rhos[-1], [0], [2,2,2])
print("Bloch‑Z von Site 0 =", np.trace(rho_site0 @ np.array([[1,0],[0,-1]])))
```

## Wichtige Bausteine

- `generator.build_graph(level)`: nur Positionen, kein Adressbaum.
- `generator.build_graph_with_addresses(level)`: fügt `addresses: set[str]` je Knoten an.
- `address.build_address_tree(G)`: Map Adresse → Knotenliste.
- `address.blocks_from_prefix(G, Lp)`: ordnet Knoten nach Präfix (Block‑Zuordnung).
- `renorm.coarse_grain_graph(G, Lp)`: Block‑Graph (Adjazenz zwischen Blöcken).
- `renorm.coarse_grain_state_average(rho, nodes, blocks)`: CPTP‑Kanal, der je Block
  die Ein‑Site‑Dichtematrizen mittelt → Grobzustand als Tensorprodukt.
- `hilbert`: Pauli‑Ops, Einbettung auf Sites, Ising/XX‑Hamiltonian, Zufalls‑ρ.
- `trace`: allgemeine partielle Spur; Reduktion auf Knoten/Positionen.
- `dynamics`: LVN‑Rechte‑Seite, RK4‑Integrator, unitäre Schritt‑Evolution, Dephasing.

## Ausspuren von Geometrie / Freiheitsgraden

- **Geometrie als Blockstruktur:** Verwende `blocks_from_prefix` um *Geometrie‑Zellen* zu
  definieren (Präfixlänge = Skala). Mit `coarse_grain_state_average` erhältst du dann eine
  reduzierte Dichtematrix **pro Zelle** (Geometrie „ausgespurt“).
- **Explizite Freiheitsgrade:** Mit `trace.reduce_to_nodes` bzw. `trace_out_nodes` kannst du
  beliebige Sites (Knoten) ausspuren.

## Grenzen / TODO

- Effektive Hamiltonians unter Renormierung (Block‑Spin, Isometrien) sind **Platzhalter**.
  Derzeit wird nur der **Graph** grobskaliert und **Zustände** via CPTP‑Mittelung reduziert.
- Für große N ist die vollquantige Behandlung nicht praktikabel. Nutze Subgraphen.

## Lizenz

© 2025 antaris — Code: MIT; Daten/Abbildungen/Texte: CC BY 4.0


## NEU: (1) Effektive Hamilton-Renormierung (Block-Isometrien)

```python
from stgraph import build_graph_with_addresses, blocks_from_prefix
from stgraph import build_xx_hamiltonian
from stgraph import renormalize_hamiltonian_by_blocks

# Fine graph and small quantum model
G = build_graph_with_addresses(level=2)
nodes = sorted(G.nodes())[:6]  # 6 Qubits (Demo)
edges = [(u,v) for (u,v) in G.edges() if u in nodes and v in nodes]
H = build_xx_hamiltonian(nodes, edges, J=1.0)

# Blocks via address prefix (e.g., 2 blocks from 6 nodes)
blocks = blocks_from_prefix(G, prefix_len=1)  # coarse cells
Heff, block_labels, block_members = renormalize_hamiltonian_by_blocks(H, nodes, blocks, method="parity")
print(Heff.shape)  # (2^M, 2^M), e.g. (4,4) for M=2 blocks
```

Methoden für die Block-Isometrie: `"parity"`, `"majority"`, `"magnetization"`.

## NEU: (2) Master-Equations mit generischen Lindblad-Kanälen (zeitabhängig)

```python
import numpy as np
from stgraph import lvne_rk4_t, lindblad_from_rates_z, build_xx_hamiltonian

# H(t): hier zeitunabhängig
H0 = build_xx_hamiltonian(nodes, edges, J=1.0)
H_of_t = lambda t: H0

# L_k(t): einfache Dephasings auf Sites [0,1]
Ls_of_t = lambda t: lindblad_from_rates_z(gamma_z=0.1, N=len(nodes), sites=[0,1])

rho_t = lvne_rk4_t(rho0, np.linspace(0,5,101), H_of_t, Ls_of_t)
```

## NEU: (3) Messungen (POVMs/Projektionen) + Logging

```python
from stgraph import projective_measure_z_on_sites, EventLogger

logger = EventLogger("st_events.jsonl")
probs, posts = projective_measure_z_on_sites(rho_t[-1], sites=[0,1], N=len(nodes))
logger.log("measurement", {"sites":[0,1], "probs": probs.tolist()})
```

Dateiformat: **JSONL**, eine Zeile pro Event.
```json
{"ts":"2025-08-28T19:17:00Z","type":"measurement","payload":{"sites":[0,1],"probs":[0.52,0.48,...]}}
```
