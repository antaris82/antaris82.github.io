
# Minimal demo for stgraph
import numpy as np
from stgraph import build_graph_with_addresses, build_xx_hamiltonian, basis_state, lvne_rk4, reduce_to_nodes

G = build_graph_with_addresses(level=2)
nodes = sorted(list(G.nodes()))[:3]
edges = [(u,v) for (u,v) in G.edges() if u in nodes and v in nodes]
H = build_xx_hamiltonian(nodes, edges, J=1.0)
psi0 = basis_state([0,0,0])
rho0 = psi0 @ psi0.conj().T
t = np.linspace(0, 2, 51)
rhos = lvne_rk4(rho0, H, t)
rho_site0 = reduce_to_nodes(rhos[-1], [0], [2,2,2])
print("Tr(rho) =", np.trace(rhos[-1]))
print("rho_site0 =", rho_site0)
