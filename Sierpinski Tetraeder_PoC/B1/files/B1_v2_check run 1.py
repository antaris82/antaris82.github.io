# B1_checks.py — numerical consistency checks for the B1 lemma/theorem
# (c) PoC helper — builds a connected graph, constructs L, L_lift, L_A(alpha),
# verifies PSD+kernel, tests partial-trace reduction, and thermo identities.

import numpy as np
from pathlib import Path

OUT_DIR = Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def triangular_lattice(n_side=10):
    """Build a connected triangular-lattice graph and its Laplacian L = D - A.
    Returns (coords, L).
    """
    pts = []
    index = {}
    for i in range(n_side):
        for j in range(n_side):
            x = i + 0.5*(j%2)
            y = (np.sqrt(3)/2.0)*j
            idx = len(pts)
            index[(i,j)] = idx
            pts.append((x,y))
    pts = np.array(pts, dtype=float)
    n = len(pts)
    A = np.zeros((n,n), dtype=float)
    # neighbor offsets (triangular lattice)
    nbrs = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    for i in range(n_side):
        for j in range(n_side):
            u = index[(i,j)]
            for di,dj in nbrs:
                ii, jj = i+di, j+dj
                if 0 <= ii < n_side and 0 <= jj < n_side:
                    v = index[(ii,jj)]
                    A[u,v] = 1.0
    A = np.maximum(A, A.T)  # undirected
    D = np.diag(A.sum(axis=1))
    L = D - A
    return pts, L

def make_clusters(pts):
    """Assign nodes to 4 clusters by nearest corner (rough 'coarse geometry')."""
    x_min,y_min = pts.min(axis=0)
    x_max,y_max = pts.max(axis=0)
    corners = np.array([[x_min,y_min],[x_min,y_max],[x_max,y_min],[x_max,y_max]])
    d2 = ((pts[:,None,:]-corners[None,:,:])**2).sum(axis=2)
    lab = d2.argmin(axis=1)  # 0..3
    return lab

def build_C_R_L0(labels, n_nodes):
    """Aggregation matrix C (4 x n), R=C^T, and coarse Laplacian L0 (K4)."""
    c = 4
    C = np.zeros((c, n_nodes), dtype=float)
    for k in range(c):
        idx = np.where(labels==k)[0]
        if len(idx)==0:
            continue
        C[k, idx] = 1.0/len(idx)
    R = C.T
    # fully connected coarse graph on 4 nodes ⇒ Laplacian of K4
    A0 = np.ones((c,c)) - np.eye(c)
    D0 = np.diag(A0.sum(axis=1))
    L0 = D0 - A0
    return C, R, L0

def observables_from_L(L, beta):
    """Return (rho, evals, p, E,S,P, Z) for Gibbs(β) of L using spectral calculus."""
    evals, evecs = np.linalg.eigh(L)
    w = np.exp(-beta*evals)
    Z = float(np.sum(w))
    p = w / Z
    rho = (evecs * p[None,:]) @ evecs.T
    E = float((p*evals).sum())
    # entropy (safe log)
    S = float(-(p*np.log(p)).sum())
    P = float((p*p).sum())
    return rho, evals, p, E, S, P, Z

def partial_trace_over_env(rho_tot, n_sys, n_env):
    """Tr_E of rho_tot; assumes Hilbert space ordering S⊗E, with dims (n_sys, n_env)."""
    rho4 = rho_tot.reshape(n_sys, n_env, n_sys, n_env)
    # trace over env indices (1 and 3)
    return np.einsum('i j k j -> i k', rho4)

def check_all(n_side=10, beta=3.0, env_evals=np.array([0.0, 1.0]), alphas=(0.0,0.25,0.5,0.75,1.0)):
    pts, L = triangular_lattice(n_side=n_side)
    n = L.shape[0]
    labels = make_clusters(pts)
    C,R,L0 = build_C_R_L0(labels, n)
    # lifted operator and approximants
    L_lift = R @ L0 @ C
    L_lift = 0.5*(L_lift + L_lift.T)  # numerical symmetrization (analytically symmetric)

    report = []
    report.append(f"n={n}, beta={beta}, env_evals={env_evals.tolist()}")
    one = np.ones((n,1))

    # Check 1: PSD & kernel for each alpha
    for a in alphas:
        L_A = (1.0-a)*L + a*L_lift
        # symmetry error
        sym_err = float(np.linalg.norm(L_A - L_A.T, ord='fro'))
        # min eigenvalue
        evals = np.linalg.eigvalsh(L_A)
        lam_min = float(evals.min())
        ker_norm = float(np.linalg.norm(L_A @ one))
        report.append(f"[Check1 α={a:.2f}] symmetry={sym_err:.2e}, λ_min={lam_min:.3e}, ||L_A 1||={ker_norm:.3e}")

    # Check 2: reduction via explicit partial trace on ρ_tot ∝ exp(-β(L⊕H_E))
    # Build exp(-βL) and exp(-βH_E) via spectra
    evals_L, QL = np.linalg.eigh(L)
    wL = np.exp(-beta*evals_L)
    eL = (QL * wL[None,:]) @ QL.T
    # env
    evals_E = np.array(env_evals, dtype=float)
    wE = np.exp(-beta*evals_E)
    QE = np.eye(len(evals_E))
    eE = QE * wE  # diagonal
    nE = len(evals_E)

    # ρ_tot ∝ e^{-βL}⊗e^{-βH_E}
    e_tot = np.kron(eL, eE)
    Z_tot = float(e_tot.trace())
    rho_tot = e_tot / Z_tot
    # partial trace over env
    rho_red = partial_trace_over_env(rho_tot, n, nE)
    # compare with Gibbs(L)
    rhoL, evals, p, E, S, P, Z = observables_from_L(L, beta)
    err_red = float(np.linalg.norm(rho_red - rhoL, ord='fro'))
    report.append(f"[Check2] ||Tr_E ρ_tot - ρ(L)||_F = {err_red:.3e}  (should be ~ 0)")

    # Check 3: thermo identities
    # d/dβ log Z = -E, d^2/dβ^2 log Z = Var(L) >= 0
    # finite differences
    d = 1e-4
    _,_,_,E0,_,_,Z0 = observables_from_L(L, beta)
    _,_,_,E_plus,_,_,Z_plus = observables_from_L(L, beta+d)
    _,_,_,E_minus,_,_,Z_minus = observables_from_L(L, beta-d)
    dlogZ_num = (np.log(Z_plus)-np.log(Z_minus))/(2*d)
    E_from_dlogZ = -dlogZ_num
    d2logZ_num = (np.log(Z_plus) - 2*np.log(Z0) + np.log(Z_minus))/(d*d)
    # variance directly in Gibbs state
    # Var(L) = <L^2> - <L>^2 = sum p_i λ_i^2 - E^2
    lam2_mean = float(((evals**2) * (np.exp(-beta*evals)/Z)).sum())
    Var_dir = lam2_mean - E0*E0
    report.append(f"[Check3] |E + d/dβ log Z| = {abs(E0 - E_from_dlogZ):.3e}")
    report.append(f"[Check3] Var(L) (direct) = {Var_dir:.6e},  d2/dβ^2 log Z = {d2logZ_num:.6e}")

    # Also export E,S,P vs alpha (for quick visual check)
    lines = ["alpha,E,S,P"]
    for a in alphas:
        L_A = (1.0-a)*L + a*L_lift
        _,_,_,E,S,P,_ = observables_from_L(L_A, beta)
        lines.append(f"{a},{E},{S},{P}")
    (OUT_DIR/"B1_alpha_observables.csv").write_text("\n".join(lines), encoding="utf-8")

    rep = "\n".join(report) + "\n"
    (OUT_DIR/"B1_checks_report.txt").write_text(rep, encoding="utf-8")
    return rep

if __name__ == "__main__":
    print(check_all())
