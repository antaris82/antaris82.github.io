# demo_ST_vs_Zd.py
import numpy as np, math, json, os
import matplotlib.pyplot as plt
from stlib.graph_st import build_st_graph, outer_corners
from stlib.graphs_regular import grid_graph_2d, grid_graph_3d
from stlib.linops import laplacian_from_edges, normalized_laplacian, bfs_distances
from stlib.heat import hutch_trace_expm, spectral_dimension
from stlib.quantum import unitary_packet, lr_front_radius
from stlib.gkls import gkls_dephasing_stepper
from stlib.dtn import dirichlet_to_neumann, perturb_edges_in_cell

OUT = "out"
os.makedirs(OUT, exist_ok=True)

# ---- Parameter (gern anpassen) ----
st_level          = 3        # 3..5
use_normalized    = False    # True -> L_norm
t_min, t_max      = 1e-3, 3e+0
t_grid            = 24
n_probes          = 24
seed              = 7

grid2d_shape      = (24,24)
grid3d_shape      = (10,10,10)
grid_periodic     = True

quantum_times     = np.linspace(0.0, 6.0, 25)
quantile_front    = 0.9

gamma_dephase     = 0.5      # GKLS Dephasierung
dt_dephase        = 0.02
steps_dephase     = 50
N_max_gkls        = 220      # Sicherheitslimit

# ---- ST Graph ----
nodes_st, edges_st, corners_st, coords_st = build_st_graph(st_level)
L_st, idx_st = laplacian_from_edges(nodes_st, edges_st)
if use_normalized:
    L_st = normalized_laplacian(L_st)

# ---- Vergleichsgitter (2D/3D) ----
nodes_g2, edges_g2 = grid_graph_2d(*grid2d_shape, periodic=grid_periodic)
L_g2, idx_g2 = laplacian_from_edges(nodes_g2, edges_g2)
nodes_g3, edges_g3 = grid_graph_3d(*grid3d_shape, periodic=grid_periodic)
L_g3, idx_g3 = laplacian_from_edges(nodes_g3, edges_g3)

# ---- Heat-Trace & spektrale Dimension ----
rng = np.random.default_rng(seed)
ts  = np.geomspace(t_min, t_max, t_grid)
def heat_stats(L, label):
    pvals=[]; err=[]
    for t in ts:
        tr_est, se = hutch_trace_expm(L, t, n_probes=n_probes, rng=rng)
        pvals.append(tr_est/L.shape[0]); err.append(se/L.shape[0])
    pvals = np.array(pvals); err = np.array(err)
    ds = spectral_dimension(ts, pvals)
    np.savetxt(f"{OUT}/heat_{label}.csv",
               np.c_[ts, pvals, err, ds],
               delimiter=",", header="t,pbar,std_err,ds", comments="")
    return pvals, err, ds

p_st, e_st, ds_st = heat_stats(L_st, "ST")
p_g2, e_g2, ds_g2 = heat_stats(L_g2, "Z2")
p_g3, e_g3, ds_g3 = heat_stats(L_g3, "Z3")

plt.figure()
plt.loglog(ts, p_st, label="ST")
plt.loglog(ts, p_g2, label="Z^2")
plt.loglog(ts, p_g3, label="Z^3")
plt.xlabel("t"); plt.ylabel("p̄_t")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/heat_trace_compare.png", dpi=160)

plt.figure()
plt.plot(ts, ds_st, label="ST")
plt.plot(ts, ds_g2, label="Z^2")
plt.plot(ts, ds_g3, label="Z^3")
plt.xscale("log"); plt.xlabel("t"); plt.ylabel("d_s(t)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/spectral_dimension_compare.png", dpi=160)

# ---- Lieb–Robinson-artige Front (Unitär, Tight-Binding H = -A = L - D) ----
# Wir nehmen H = -A = D - L (da L = D - A)
import scipy.sparse as sp
D = sp.diags(L_st.diagonal())
H_st = (D - L_st) * (-1.0)

# Quelle: zentralster Knoten ~ kürzeste Adresse
seed_node = min(nodes_st, key=len)
seed_idx  = idx_st[seed_node]
dists     = bfs_distances(nodes_st, edges_st, seed_idx)

prob_t = unitary_packet(H_st.tocsr(), psi0_idx=seed_idx, times=quantum_times, normalize=True)
r90    = lr_front_radius(prob_t, dists, quantile=quantile_front)
np.savetxt(f"{OUT}/lr_front_ST.csv", np.c_[quantum_times, r90], delimiter=",", header="t,r90", comments="")
plt.figure()
plt.plot(quantum_times, r90, marker="o")
plt.xlabel("t"); plt.ylabel(f"LR-Front-Radius (q={quantile_front})")
plt.tight_layout(); plt.savefig(f"{OUT}/lr_front_ST.png", dpi=160)

# ---- GKLS Dephasierung (klein halten) ----
N = L_st.shape[0]
if N <= N_max_gkls:
    step = gkls_dephasing_stepper(H_st.tocsr(), gamma=gamma_dephase, dt=dt_dephase)
    R = np.zeros((N,N), dtype=complex); R[seed_idx,seed_idx]=1.0
    radii=[]
    for s in range(steps_dephase):
        R = step(R)
        # Diagonale als Wahrscheinlichkeiten
        p = np.real(np.diag(R))
        # gleiche LR-Metrik
        # (dists bereits berechnet)
        dmax = dists.max()
        mass = np.zeros(dmax+1)
        for i,di in enumerate(dists):
            if di>=0: mass[di]+=p[i]
        c = np.cumsum(mass)
        r = np.searchsorted(c, 0.9)
        radii.append(r)
    np.savetxt(f"{OUT}/lr_front_ST_gkls.csv",
               np.c_[np.arange(steps_dephase)*dt_dephase, np.array(radii)],
               delimiter=",", header="t,r90_gkls", comments="")
    plt.figure(); plt.plot(np.arange(steps_dephase)*dt_dephase, radii, marker=".")
    plt.xlabel("t"); plt.ylabel("Front-Radius (GKLS 0.9)")
    plt.tight_layout(); plt.savefig(f"{OUT}/lr_front_ST_gkls.png", dpi=160)

# ---- DtN + Tiefen-Perturbation ----
# Wähle Rand = 4 äußere Ecken
b_nodes = [ idx_st[name] for name in corners_st ]
Lambda0 = dirichlet_to_neumann(L_st, b_nodes)
np.savetxt(f"{OUT}/dtn_ST_baseline.csv", Lambda0, delimiter=",")

# Perturbiere ein tiefes Sub‑Zell‑Prefix (falls vorhanden)
cell_prefix = "01" if any(str(u).startswith("01") for u in nodes_st) else min(nodes_st, key=len)
weights = perturb_edges_in_cell(nodes_st, edges_st, cell_prefix=cell_prefix, factor=2.0)
L_pert, _ = laplacian_from_edges(nodes_st, edges_st, weights=weights)
if use_normalized: 
    from stlib.linops import normalized_laplacian
    L_pert = normalized_laplacian(L_pert)
Lambda1 = dirichlet_to_neumann(L_pert, b_nodes)
np.savetxt(f"{OUT}/dtn_ST_perturbed.csv", Lambda1, delimiter=",")

# Differenzen‑Norm
dLambda = Lambda1 - Lambda0
sv = np.linalg.svd(dLambda, compute_uv=False)
delta_op = float(sv[0]) if len(sv)>0 else 0.0
with open(f"{OUT}/dtn_delta.json","w") as f:
    json.dump({"cell_prefix":cell_prefix,"factor":2.0,"||delta_Lambda||_2":delta_op}, f, indent=2)

print("Fertig. Ergebnisse in 'out/'.")
