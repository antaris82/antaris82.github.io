# ================================================================
# Sierpiński-Tetraeder (ST) – Level-4 Urgraph, Approximant-Verkleinerung,
# realistische Dichtematrizen via Teilspur, Observablen, 4 GIFs
# ================================================================
# Anforderungen: numpy, matplotlib, Pillow (für GIF)
# Keine externen Styles/Farben; nur Matplotlib-Default.
# ================================================================

import numpy as np
import itertools, math, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# ----------------------
# 0) Reproduzierbarkeit
# ----------------------
np.random.seed(7)

# ----------------------
# 1) Geometrie & Graph
# ----------------------
# Eckpunkte eines regulären Tetraeders (nur für IFS/Koordinaten)
V0 = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, math.sqrt(3)/2, 0.0],
    [0.5, math.sqrt(3)/6, math.sqrt(6)/3],
], dtype=float)

def build_graph_by_addresses(level: int):
    """
    Robuste Konstruktion des Level-m ST-Graphs:
    - Knoten = Vereinigung der Ecken aller Level-m-Tetraeder (per Adressliste)
    - Kanten = 1-Skelette jeder Zelle (6 Kanten)
    """
    pts = []
    addresses = list(itertools.product(range(4), repeat=level))
    for addr in addresses:
        for j in range(4):
            x = V0[j].copy()
            for i in addr:
                x = 0.5*(x + V0[i])
            pts.append(tuple(np.round(x, 12)))
    V = np.array(sorted(set(pts)), dtype=float)                 # Knoten
    idx = {tuple(p): k for k,p in enumerate(V)}                # Punkt->Index
    edges = set()
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]              # Kanten in einer Zelle
    for addr in addresses:
        verts = []
        for j in range(4):
            x = V0[j].copy()
            for i in addr:
                x = 0.5*(x + V0[i])
            verts.append(tuple(np.round(x,12)))
        ids = [idx[v] for v in verts]
        for a,b in pairs:
            i,j = ids[a], ids[b]
            if i>j: i,j=j,i
            edges.add((i,j))
    n = len(V)
    A = np.zeros((n,n), dtype=float)
    for i,j in edges:
        A[i,j] = 1.0; A[j,i] = 1.0
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A                                       # kombinatorischer Laplace-Operator
    return V, A, L

# Level 4 Urgraph
LEVEL_FINE = 4
V4, A4, L4 = build_graph_by_addresses(LEVEL_FINE)
n = V4.shape[0]
V2 = V4[:, :2].copy()  # 2D-Projektion für Scatterplots

# ----------------------
# 2) Coarse-Graining (Lift von Level 0)
# ----------------------
V0_points, A0, L0 = build_graph_by_addresses(0)  # 4 Ecken (vollständig verbunden)
# Cluster-Zuordnung: feiner Knoten -> nächster Eckpunkt
assign = np.argmin(((V4[:,None,:]-V0_points[None,:,:])**2).sum(axis=2), axis=1)
c = V0_points.shape[0]  # =4

# Aggregationsmatrix C: coarse (4)  <- fine (n)   ;   Rekonstruktion R = C^T
C = np.zeros((c, n), dtype=float)
for k in range(c):
    idxs = np.where(assign==k)[0]
    C[k, idxs] = 1.0/len(idxs)
R = C.T

# Lift des Level-0-Operators in feinen Raum
L_lift = R @ L0 @ C
L_lift = 0.5*(L_lift + L_lift.T)  # numerisch symmetrieren

def L_A_alpha(alpha: float):
    """Stetige Verkleinerung: Approximant-Operator zwischen fein (L4) und gehoben grob (L_lift)."""
    return (1.0-alpha)*L4 + alpha*L_lift

# ---------------------------------------------------------
# 3) Dichtematrizen via *expliziter* Teilspur (Kronecker-Summe)
# ---------------------------------------------------------
# H_total = L ⊕ H_env = L ⊗ I + I ⊗ H_env
# ⇒ ρ_tot ∝ exp(-β H_total); ρ_red = Tr_env ρ_tot  ∝  exp(-β L)
# Implementiert mit *expliziter Summation* über die Eigenwerte des Environments
# (spart explizite Konstruktion von (n*d_env)×(n*d_env)-Matrizen, ist aber mathematisch äquivalent).

BETA = 3.0

def reduced_density_via_partial_trace(L, beta=BETA, env_evals=None):
    """
    Liefert ρ_red, Eigenwerte/-vektoren von L und die Populations p (thermisch).
    env_evals: Eigenwerte von H_env (1D-Array), z.B. [0, Δ] oder [0,1,2] etc.
    """
    if env_evals is None:
        env_evals = np.array([0.0, 1.0])  # minimalistisches 2-Level-Environment
    evals, evecs = np.linalg.eigh(L)                     # L = Q Λ Q^T
    w_env = np.exp(-beta*env_evals)
    Z_env = float(np.sum(w_env))                         # Tr e^{-β H_env}
    w = np.exp(-beta*evals) * Z_env                      # Summe über Env-Zustände
    Z = float(np.sum(w))                                 # Gesamtpartition (bis Norm)
    p = w / Z                                            # Populations in L-Eigenbasis
    rho = (evecs * p[None,:]) @ evecs.T                  # ρ = Q diag(p) Q^T
    return rho, evals, p, evecs

# Urgraph – reduzierte Dichte über Teilspur
rho_U, evals_U, p_U, evecs_U = reduced_density_via_partial_trace(
    L4, beta=BETA, env_evals=np.array([0.0, 1.5, 3.0])
)

# Observablen (in gemeinsamer Diagonalbasis)
def energy_from_spectrum(evals, p):  # E = Tr(ρ L) = Σ p_i λ_i
    return float(np.sum(p * evals))
def entropy_from_p(p):               # S = -Tr(ρ log ρ)
    eps = 1e-16
    px = np.clip(p, eps, 1.0)
    return float(-np.sum(px*np.log(px)))
def purity_from_p(p):                # Tr(ρ^2)
    return float(np.sum(p*p))

E_U = energy_from_spectrum(evals_U, p_U)
S_U = entropy_from_p(p_U)
P_U = purity_from_p(p_U)

# ---------------------------------------------------------
# 4) Subgraph für performante Animation (Approximant)
# ---------------------------------------------------------
# Für "live" GIFs verwenden wir einen repräsentativen Subgraph (z.B. 160 Knoten).
# Der Urgraph (voll) wird als statische Heatmap ausgegeben.

N_SUB = 160
rng = np.random.default_rng(42)
idx_sub = np.sort(rng.choice(n, size=min(N_SUB, n), replace=False))

L4_sub    = L4[np.ix_(idx_sub, idx_sub)].copy()
Llift_sub = L_lift[np.ix_(idx_sub, idx_sub)].copy()

def L_A_sub_alpha(alpha: float):
    return (1.0-alpha)*L4_sub + alpha*Llift_sub

# Frames über α
frames = 8
alphas = np.linspace(0.0, 1.0, frames)

# Approximant: reduzierte Dichten via Teilspur (wie oben), plus Observablen
rhoA_sub_list = []
E_list, S_list, P_list = [], [], []
for a in alphas:
    Lsub = L_A_sub_alpha(a)
    rhoA, evalsA, pA, _ = reduced_density_via_partial_trace(
        Lsub, beta=BETA, env_evals=np.array([0.0, 1.5, 3.0])
    )
    rhoA_sub_list.append(rhoA)
    E_list.append(energy_from_spectrum(evalsA, pA))
    S_list.append(entropy_from_p(pA))
    P_list.append(purity_from_p(pA))

E = np.array(E_list); S = np.array(S_list); P = np.array(P_list)

# ---------------------------------------------------------
# 5) GIFs erzeugen (lokal im Ordner speichern)
# ---------------------------------------------------------
# Lokaler Ordner:
# - .py: Ordner der Datei
# - Jupyter: aktuelles Arbeitsverzeichnis
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()

out_dir = base_dir  # exakt „im aktuellen Ordner“ speichern
out_dir.mkdir(parents=True, exist_ok=True)

# 5.1) Urgraph-Dichte (voll) – Heatmap mit Achsen/Colorbar & Werten
figU = plt.figure(figsize=(8, 6), dpi=120)
axU = plt.gca()
imU = axU.imshow(rho_U, aspect='auto')
axU.set_title("Urgraph (Level 4): reduzierte Dichtematrix ρ_U ∝ e^{-β L_U}")
axU.set_xlabel("Knotenindex j")
axU.set_ylabel("Knotenindex i")
cbarU = plt.colorbar(imU, ax=axU)
cbarU.set_label("ρ_U[i, j]")
txtU = axU.text(
    0.02, 0.98,
    f"E = {E_U:.6f}\nS = {S_U:.6f}\nPurity = {P_U:.6f}",
    transform=axU.transAxes, va='top', ha='left',
    bbox=dict(boxstyle="round")
)
plt.tight_layout()
gif1_path = out_dir / "st_density_urgraph.gif"
animU = FuncAnimation(figU, lambda i: (imU, txtU), frames=2, interval=400, blit=False)
animU.save(str(gif1_path), writer=PillowWriter(fps=1))
plt.close(figU)

# 5.2) Approximant-Dichte (Animation, Subgraph) – Heatmap mit Labels & Werten
figA = plt.figure(figsize=(8, 6), dpi=120)
axA = plt.gca()
imA = axA.imshow(rhoA_sub_list[0], aspect='auto')
ttlA = axA.set_title("Approximant (Subgraph): ρ_A(α)  |  α=0.00")
axA.set_xlabel("Knotenindex j")
axA.set_ylabel("Knotenindex i")
cbarA = plt.colorbar(imA, ax=axA)
cbarA.set_label("ρ_A[i, j]")
boxA = axA.text(
    0.02, 0.98,
    f"α = {alphas[0]:.2f}\nE = {E[0]:.6f}\nS = {S[0]:.6f}\nPurity = {P[0]:.6f}",
    transform=axA.transAxes, va='top', ha='left',
    bbox=dict(boxstyle="round")
)
plt.tight_layout()
gif2_path = out_dir / "st_density_approximant.gif"
def updA(i):
    imA.set_data(rhoA_sub_list[i])
    ttlA.set_text(f"Approximant (Subgraph): ρ_A(α)  |  α={alphas[i]:.2f}")
    boxA.set_text(f"α = {alphas[i]:.2f}\nE = {E[i]:.6f}\nS = {S[i]:.6f}\nPurity = {P[i]:.6f}")
    return (imA, ttlA, boxA)
animA = FuncAnimation(figA, updA, frames=len(rhoA_sub_list), interval=500, blit=False)
animA.save(str(gif2_path), writer=PillowWriter(fps=2))
plt.close(figA)

# 5.3) Observablen über α (Animation, Subgraph) – Kurven + Info-Box
figO = plt.figure(figsize=(9, 6), dpi=120)
axO = plt.gca()
lE, = axO.plot(alphas, E, label="Energie  Tr(ρL)")
lS, = axO.plot(alphas, S, label="Entropie  -Tr(ρ log ρ)")
lP, = axO.plot(alphas, P, label="Purity  Tr(ρ^2)")
pE, = axO.plot([alphas[0]],[E[0]], marker='o')
pS, = axO.plot([alphas[0]],[S[0]], marker='o')
pP, = axO.plot([alphas[0]],[P[0]], marker='o')
axO.set_xlabel("α  (0 = Level-4; 1 = gehobener Level-0-Effektivoperator)")
axO.set_ylabel("Wert")
axO.set_title("Approximant-Observablen (Subgraph)")
axO.legend()
boxO = axO.text(
    0.98, 0.98,
    f"α = {alphas[0]:.2f}\nE = {E[0]:.6f}\nS = {S[0]:.6f}\nPurity = {P[0]:.6f}",
    transform=axO.transAxes, va='top', ha='right',
    bbox=dict(boxstyle="round")
)
plt.tight_layout()
gif3_path = out_dir / "st_observables.gif"
def updO(i):
    pE.set_data([alphas[i]],[E[i]])
    pS.set_data([alphas[i]],[S[i]])
    pP.set_data([alphas[i]],[P[i]])
    boxO.set_text(f"α = {alphas[i]:.2f}\nE = {E[i]:.6f}\nS = {S[i]:.6f}\nPurity = {P[i]:.6f}")
    return (pE,pS,pP,boxO)
animO = FuncAnimation(figO, updO, frames=len(alphas), interval=500, blit=False)
animO.save(str(gif3_path), writer=PillowWriter(fps=2))
plt.close(figO)

# 5.4) Graph & Coarsening (Animation) – Punktwolke + Clustergrößen + α-Box
cent2 = V0_points[:,:2]
sizes0 = np.array([(assign==k).sum() for k in range(c)], dtype=float)
sizes0 = 30 + 120 * sizes0/sizes0.max()
figG = plt.figure(figsize=(8, 8), dpi=120)
axG = plt.gca()
axG.scatter(V2[:,0], V2[:,1], s=2)
scent = axG.scatter(cent2[:,0], cent2[:,1], s=sizes0)
axG.set_aspect('equal'); axG.set_xticks([]); axG.set_yticks([])
axG.set_title("ST-Urgraph & Cluster (α skaliert Markergröße)")
boxG = axG.text(
    0.02, 0.98, f"α = {alphas[0]:.2f}",
    transform=axG.transAxes, va='top', ha='left',
    bbox=dict(boxstyle="round")
)
plt.tight_layout()
gif4_path = out_dir / "st_graph_and_coarsening.gif"
def updG(i):
    a = alphas[i]
    scent.set_sizes((1.0+2.0*a)*sizes0)
    boxG.set_text(f"α = {a:.2f}")
    return (scent, boxG)
animG = FuncAnimation(figG, updG, frames=len(alphas), interval=500, blit=False)
animG.save(str(gif4_path), writer=PillowWriter(fps=2))
plt.close(figG)

print("Fertig! Dateien gespeichert:")
print((out_dir / "st_density_urgraph.gif").resolve())
print((out_dir / "st_density_approximant.gif").resolve())
print((out_dir / "st_observables.gif").resolve())
print((out_dir / "st_graph_and_coarsening.gif").resolve())
