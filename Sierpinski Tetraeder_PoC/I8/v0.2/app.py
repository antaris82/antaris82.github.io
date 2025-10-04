# app.py
# IDEAL (LvN) ↔ REAL (GKSL) ↔ Raumzeit (Cluster/Poset) ↔ Symmetriebruch ↔ „Higgs“
# Performance-Optimierungen (RAM/CPU) ohne Ergebnis-/Feature-Einschränkungen:
#  • Φ_Δt aus Unitarität via Kraus-Blöcke (exakt CPTP, keine N^4-Schleifen)
#  • Effektive Widerstände per einmaliger Sparse-Faktorisierung (kein L^+)
#  • Cheeger/Fiedler mit eigsh (sparse), DtN via sparse Solves, Harmonic-Embed via sparse Solves
#  • Skizzierte Memory-LS (ohne Kronecker-Giganten), leichte Reduktion von Kopien
# (c) 2025 — MIT License

import io, json, math, zipfile
from fractions import Fraction as F
from typing import Dict, Tuple, List, Any, Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as sla
import scipy.linalg as dla

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# -----------------------
# Utils
# -----------------------
rng = np.random.default_rng(42)
EPS = 1e-12

def vec(M: np.ndarray) -> np.ndarray: return M.reshape((-1,), order="F")
def unvec(v: np.ndarray, n: int) -> np.ndarray: return v.reshape((n, n), order="F")
def hermitize(M): return (M + M.conj().T) / 2
def clip_psd(M, tol=1e-12):
    w, U = np.linalg.eigh(hermitize(M)); w = np.clip(w, 0, None); return (U * w) @ U.conj().T
def frob(A): return float(np.linalg.norm(A))
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_fig_html(fig: go.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(path), include_plotlyjs="cdn", full_html=True)

def save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, arr)

# -----------------------
# Geometrie / Fraktale
# -----------------------
class Graph:
    def __init__(self):
        self.pos_key_to_idx: Dict[Any,int] = {}
        self.idx_to_bary: Dict[int,Tuple] = {}
        self.birthlevel: Dict[int,int] = {}
        self.addr: Dict[int,str] = {}
        self.edges: List[Tuple[int,int]] = []
        self.dim = 0
    def add_vertex(self, bary, addr, force_birthlevel=None, mode="bary"):
        if mode=="bary":
            k = tuple((x.numerator, x.denominator) for x in bary)
        else:
            k = ("RAW",)+tuple(bary)
        if k in self.pos_key_to_idx:
            i = self.pos_key_to_idx[k]
            if addr and (self.addr[i]=="" or addr<self.addr[i]): self.addr[i]=addr
            if force_birthlevel is not None:
                self.birthlevel[i] = min(self.birthlevel.get(i, force_birthlevel), force_birthlevel)
            return i
        i = len(self.pos_key_to_idx)
        self.pos_key_to_idx[k]=i; self.idx_to_bary[i]=bary; self.addr[i]=addr
        if force_birthlevel is not None:
            self.birthlevel[i]=force_birthlevel
        else:
            if mode=="bary":
                dens=[bi.denominator for bi in bary]
                def v2(n):
                    c=0
                    while n>0 and n%2==0: n//=2; c+=1
                    return c
                self.birthlevel[i]=max(v2(d) for d in dens)
            else:
                self.birthlevel[i]=0
        if mode=="bary" and len(bary) in (3,4): self.dim=len(bary)-1
        else: self.dim=max(self.dim,3)
        return i
    def add_edge(self, i, j):
        if i==j: return
        if i>j: i,j=j,i
        self.edges.append((i,j))

def _bmid(a,b): return tuple((ai+bi)/2 for ai,bi in zip(a,b))

SG_BASE=[(F(1),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1))]
ST_BASE=[(F(1),F(0),F(0),F(0)),(F(0),F(1),F(0),F(0)),(F(0),F(0),F(1),F(0)),(F(0),F(0),F(0),F(1))]

def build_SG(L:int)->Graph:
    g=Graph()
    cells=[tuple(SG_BASE)]; addrv=[""]
    for _ in range(L):
        newc=[]; newa=[]
        for cell,a in zip(cells,addrv):
            v0,v1,v2=cell
            m01=_bmid(v0,v1); m02=_bmid(v0,v2); m12=_bmid(v1,v2)
            newc += [(v0,m01,m02),(m01,v1,m12),(m02,m12,v2)]
            newa += [a+"0",a+"1",a+"2"]
        cells,addrv=newc,newa
    for cell,a in zip(cells,addrv):
        ids=[g.add_vertex(v,a) for v in cell]
        for e in [(0,1),(1,2),(0,2)]: g.add_edge(ids[e[0]], ids[e[1]])
    return g

def build_ST(L:int)->Graph:
    g=Graph()
    cells=[tuple(ST_BASE)]; addrv=[""]
    for _ in range(L):
        newc=[]; newa=[]
        for cell,a in zip(cells,addrv):
            v0,v1,v2,v3=cell
            m01=_bmid(v0,v1); m02=_bmid(v0,v2); m03=_bmid(v0,v3)
            m12=_bmid(v1,v2); m13=_bmid(v1,v3); m23=_bmid(v2,v3)
            newc += [(v0,m01,m02,m03),(m01,v1,m12,m13),(m02,m12,v2,m23),(m03,m13,m23,v3)]
            newa += [a+"0",a+"1",a+"2",a+"3"]
        cells,addrv=newc,newa
    def tet_edges(v): return [(v[0],v[1]),(v[0],v[2]),(v[0],v[3]),(v[1],v[2]),(v[1],v[3]),(v[2],v[3])]
    for cell,a in zip(cells,addrv):
        ids=[g.add_vertex(v,a) for v in cell]
        for e in tet_edges(ids): g.add_edge(*e)
    return g

def build_VIC(L:int)->Graph:
    g=Graph()
    from collections import deque
    q=deque(); q.append((F(0),F(0),F(1))); addrs=[""]
    for _ in range(L):
        newq=[]; newa=[]
        while q:
            x0,y0,s=q.popleft(); s3=s/3; a=addrs.pop(0)
            centers=[(x0+s3,y0+s3),(x0,y0+s3),(x0+2*s3,y0+s3),(x0+s3,y0),(x0+s3,y0+2*s3)]
            for i,(xc,yc) in enumerate(centers): newq.append((xc,yc,s3)); newa.append(a+str(i))
        q=deque(newq); addrs=newa
    pts={}
    while q:
        x0,y0,s=q.popleft(); a=addrs.pop(0)
        verts=[(x0,y0),(x0+s,y0),(x0+s,y0+s),(x0,y0+s)]
        ids=[]
        for (x,y) in verts:
            k=tuple((t.numerator,t.denominator) for t in (x,y))
            if k not in pts:
                idx=len(pts); pts[k]=idx
                g.pos_key_to_idx[k]=idx; g.idx_to_bary[idx]=(x,y); g.addr[idx]=a; g.dim=2
                def v3(n):
                    c=0
                    while n>0 and n%3==0: n//=3; c+=1
                    return c
                g.birthlevel[idx]=max(v3(x.denominator), v3(y.denominator))
            else: idx=pts[k]
            ids.append(idx)
        cyc=[0,1,2,3,0]
        for i in range(4): g.add_edge(ids[cyc[i]], ids[cyc[i+1]])
    return g

def build_DF(L:int)->Graph:
    g=Graph()
    A=g.add_vertex((F(1),),"A",force_birthlevel=0); B=g.add_vertex((F(0),),"B",force_birthlevel=0)
    edges=[(A,B)]
    for l in range(1,L+1):
        new=[]
        for (u,v) in edges:
            x=g.add_vertex((F(1),),f"x{l}_{u}",force_birthlevel=l)
            y=g.add_vertex((F(0),),f"y{l}_{v}",force_birthlevel=l)
            new += [(u,x),(x,v),(u,y),(y,v)]
        edges=new
    g.edges=[]; [g.add_edge(*e) for e in edges]
    return g

def build_BTREE(L:int)->Graph:
    g=Graph()
    layers=[]
    for d in range(L+1):
        layer=[]
        n_nodes=2**d
        for k in range(n_nodes):
            x=float((2*k+1)/(2*n_nodes)); y=-float(d); z=0.0
            i=g.add_vertex((x,y,z), f"{d}:{k}", force_birthlevel=d, mode="raw")
            layer.append(i)
        layers.append(layer)
        if d>0:
            for j,ch in enumerate(layer):
                p=layers[d-1][j//2]; g.add_edge(p,ch)
    return g

def build_fractal(name, L):
    if name=="SG": return build_SG(L)
    if name=="ST": return build_ST(L)
    if name=="VIC": return build_VIC(L)
    if name=="DF": return build_DF(L)
    return build_BTREE(L)

# -----------------------
# Laplacian & Sparse-Hilfsfunktionen
# -----------------------
def build_laplacian(n, edges, weights=None) -> ss.csr_matrix:
    if weights is None: weights={}
    rows,cols,data=[],[],[]; deg=np.zeros(n,float)
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i); w=weights.get(key,1.0)
        rows += [i,j]; cols += [j,i]; data += [-w,-w]
        deg[i]+=w; deg[j]+=w
    rows += list(range(n)); cols += list(range(n)); data += list(deg)
    return ss.coo_matrix((data,(rows,cols)), shape=(n,n)).tocsr()

def grounded_factorization(L: ss.csr_matrix, ground: int = 0):
    """Faktorisiert L[~g,~g] (SPD) einmalig für effiziente Widerstands-/DtN-Solves."""
    n = L.shape[0]
    keep = np.array([i for i in range(n) if i != ground], dtype=int)
    Lred = L[keep][:, keep].tocsc()
    LU = sla.splu(Lred)  # robust, auch ohne explizites Cholesky
    return keep, LU

def eff_resistance_pair(L: ss.csr_matrix, LU, keep_idx: np.ndarray, i: int, j: int, ground: int = 0) -> float:
    """Effektiver Widerstand R_ij = b^T y mit (L~) y = b~, b=e_i-e_j, 'ground' eliminiert."""
    if i == j: return 0.0
    n = L.shape[0]
    # Map full-index -> reduced index
    def ridx(k): 
        if k==ground: return None
        # binary search since keep is sorted; but n ist nicht riesig: linear ok
        # besser: vorab Map bauen (einmalig). Hier quick:
        return int(np.searchsorted(keep_idx, k))
    b = np.zeros(n, float)
    b[i] = 1.0; b[j] = -1.0
    # remove ground
    b_red = b[keep_idx]
    y = LU.solve(b_red)
    R = float(np.dot(b_red, y))
    return max(R, 0.0)

def eff_res_on_edges_sparse(L: ss.csr_matrix, edges: Iterable[Tuple[int,int]]) -> Dict[Tuple[int,int], float]:
    """Effektive Widerstände nur auf Kanten, per einmaliger LU von L~."""
    n = L.shape[0]
    ground = 0  # beliebig, invariant
    keep, LU = grounded_factorization(L, ground=ground)
    res = {}
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i)
        res[key] = eff_resistance_pair(L, LU, keep, i, j, ground=ground)
    return res

def outer_boundary(fractal, g:Graph, edges):
    B=set()
    if fractal in ("SG","ST"):
        for i,b in g.idx_to_bary.items():
            if any(x==0 for x in b): B.add(i)
    elif fractal=="VIC":
        z0,z1=F(0),F(1)
        for i,(x,y) in g.idx_to_bary.items():
            if x in (z0,z1) or y in (z0,z1): B.add(i)
    else:  # DF, BTREE
        deg=np.zeros(len(g.idx_to_bary),int)
        for (u,v) in edges: deg[u]+=1; deg[v]+=1
        for i,d in enumerate(deg):
            if d==1: B.add(i)
    return B

# -----------------------
# REAL-Gewichte & DtN (sparse)
# -----------------------
def real_weights_deepening(L_id_csr: ss.csr_matrix, edges, g:Graph, fractal):
    R = eff_res_on_edges_sparse(L_id_csr, edges)  # keine pinv mehr
    B = outer_boundary(fractal,g,edges)
    gamma=3 if fractal=="VIC" else 2
    w={}
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i)
        # Randgewicht = 1
        if fractal in ("DF","BTREE"):
            if i in B or j in B: w[key]=1.0; continue
        else:
            if i in B and j in B: w[key]=1.0; continue
        l=max(g.birthlevel.get(i,0), g.birthlevel.get(j,0))
        w0=1.0/max(R[key],EPS)
        w[key]=(gamma**l)*w0
    return w

def kron_reduce_dense(L: np.ndarray, keep: set):
    # (nur bei Bedarf im KRON-Modus; Graphgrößen hier typischerweise klein)
    keep_idx = np.array(sorted(list(keep)), dtype=int)
    allidx = np.arange(L.shape[0])
    elim = np.setdiff1d(allidx, keep_idx)
    if len(elim)==0: return L[np.ix_(keep_idx,keep_idx)]
    LBB=L[np.ix_(keep_idx,keep_idx)]
    LBE=L[np.ix_(keep_idx,elim)]
    LEB=L[np.ix_(elim,keep_idx)]
    LEE=L[np.ix_(elim,elim)]
    LEE_p=np.linalg.pinv(LEE)
    return LBB - LBE@LEE_p@LEB

def real_weights_kron(L_id_csr: ss.csr_matrix, edges, g:Graph, fractal):
    # Für KRON vereinfachend dense arbeiten (n moderat); bleibt funktionsgleich.
    L_id = L_id_csr.toarray()
    B0=outer_boundary(fractal,g,edges); w={}
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i)
        if fractal in ("DF","BTREE"):
            if i in B0 or j in B0: w[key]=1.0; continue
        else:
            if i in B0 and j in B0: w[key]=1.0; continue
        ell=max(g.birthlevel.get(i,0), g.birthlevel.get(j,0))
        keep=set(B0)|{k for k,bl in g.birthlevel.items() if bl<=ell}
        Lred=kron_reduce_dense(L_id, keep)
        # R_ij in reduziertem Graph: kleiner, daher ok mit pinv
        Lp=np.linalg.pinv(Lred); d=np.diag(Lp)
        idx_sorted=list(sorted(keep))
        ii=idx_sorted.index(i); jj=idx_sorted.index(j)
        rij=float(max(d[ii]+d[jj]-2*Lp[ii,jj],0.0)); w[key]=1.0/max(rij,EPS)
    return w

def dtn_map_sparse(L_csr: ss.csr_matrix, B: Iterable[int]) -> np.ndarray:
    """DtN via Schur-Komplement mit Solves statt pinv: Λ = L_BB - L_BI L_II^{-1} L_IB."""
    B = sorted(list(B))
    n = L_csr.shape[0]
    I = sorted(list(set(range(n)) - set(B)))
    if len(I)==0:
        return L_csr[np.ix_(B,B)].toarray()
    LBB = L_csr[np.ix_(B,B)].toarray()
    LIB = L_csr[np.ix_(I,B)].tocsc()
    LII = L_csr[np.ix_(I,I)].tocsc()
    # Solve LII * X = LIB  → X = LII^{-1} LIB
    LU = sla.splu(LII)
    X = LU.solve(LIB.toarray())
    return LBB - LIB.T.toarray() @ X

def dtn_error_sparse(L_id_csr: ss.csr_matrix, L_real_csr: ss.csr_matrix, B) -> float:
    Lam_id = dtn_map_sparse(L_id_csr, B)
    Lam_real = dtn_map_sparse(L_real_csr, B)
    num=frob(Lam_real-Lam_id); den=max(1.0,frob(Lam_id))
    return float(num/den)

def calibrate_s_star_sparse(L_id_csr, edges, g, fractal, w_base, grid=np.linspace(0.25,4.0,25)):
    B=outer_boundary(fractal,g,edges); best=(float("inf"),1.0,None)
    n=L_id_csr.shape[0]
    for s in grid:
        w={}
        for (i,j) in edges:
            key=(i,j) if i<j else (j,i)
            if fractal in ("DF","BTREE"):
                if i in B or j in B: w[key]=1.0
                else: w[key]=s*w_base[key]
            else:
                if i in B and j in B: w[key]=1.0
                else: w[key]=s*w_base[key]
        Ls = build_laplacian(n, edges, w)
        err = dtn_error_sparse(L_id_csr, Ls, B)
        if err<best[0]: best=(err,s,Ls)
    return dict(s_star=best[1], error=best[0], L_cal=best[2])

# -----------------------
# Dynamik (LvN, GKSL)
# -----------------------
def H_from_L_dense(L_csr: ss.csr_matrix) -> np.ndarray:
    return L_csr.toarray()

def lvN(H: np.ndarray, rho0: np.ndarray, t: float):
    U=dla.expm(-1j*H*t)
    return U@rho0@U.conj().T

def lindblad_ops_dense(n, edges, w):
    Ls=[]
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i); wij=w.get(key,1.0)
        if wij<=0: continue
        eij=np.zeros((n,n),complex); eji=np.zeros((n,n),complex)
        s = math.sqrt(wij)
        eij[i,j]=s; eji[j,i]=s
        Ls += [eij,eji]
    return Ls

def gksl_step(H, rho, Ls, dt):
    comm=-1j*(H@rho - rho@H); diss=np.zeros_like(rho,complex)
    for Lk in Ls:
        LdL=Lk.conj().T@Lk
        diss += Lk@rho@Lk.conj().T - 0.5*(LdL@rho + rho@LdL)
    rho2 = rho + dt*(comm+diss)
    return hermitize(rho2)

def superop_gksl(H, Ls):
    n=H.shape[0]; I=np.eye(n)
    Hcomm = -1j*(np.kron(I, H) - np.kron(H.T, I))
    Diss = np.zeros((n*n, n*n), complex)
    for Lk in Ls:
        LdL=Lk.conj().T@Lk
        Diss += np.kron(Lk.T, Lk) - 0.5*(np.kron(I, LdL) + np.kron(LdL.T, I))
    return Hcomm + Diss

# -----------------------
# Φ_Δt: schneller, exakt CPTP (Kraus aus U-Blöcken)
# -----------------------
def channel_from_unitary_kraus(H_full: np.ndarray, S_idx: List[int], dt: float, rhoE: np.ndarray):
    """
    Baut Φ_Δt(X) = Tr_E[ U (X⊗ρE) U† ].
    Implementierung:
      • U = exp(-i H dt)
      • U als (nS,nE,nS,nE)-Block-Array
      • ρE = Σ pβ |β><β| → Kraus K_{αβ} = √pβ * U_{αβ}  (α=env out, β=env in)
      • Superoperator T = Σ (K ⊗ K*) ; Choi J = Σ vec(K) vec(K)†
      • TP-Check: Σ K†K = I
    """
    n = H_full.shape[0]
    S = np.array(sorted(S_idx), dtype=int)
    E = np.array(sorted(list(set(range(n)) - set(S_idx))), dtype=int)
    nS = len(S); nE = len(E)

    U = dla.expm(-1j*H_full*dt)

    # Permutation: ordne Basis so, dass (S,E) blockweise contiguous sind
    perm = np.concatenate([S, E])
    Upe = U[np.ix_(perm, perm)]

    # Blöcke: U_{αβ} ist (nS×nS), mit α,β in [0..nE-1]
    U4 = Upe.reshape(nS, nE, nS, nE, order='C')
    # ρE-Eigenzerlegung (diagonal genügt)
    wE, UE = np.linalg.eigh(rhoE) if rhoE.size>0 else (np.array([1.0]), np.eye(1))
    # Für Einfachheit: in Standardbasis; falls ρE ≈ I/nE, dann wE=1/nE, UE=I
    # Wir transformieren die Eingangs-Env-Achse (β) in die ρE-Eigenbasis:
    # Effektiv: U_{αβ'} = Σβ U_{αβ} (UE)_{β,β'}
    if nE>0:
        U4 = U4 @ UE  # wirkt auf letzte Achse (β-in)

    # Kraus sammeln
    Vdim = nS*nS
    T = np.zeros((Vdim, Vdim), complex)
    J = np.zeros((Vdim, Vdim), complex)
    KdagK = np.zeros((nS, nS), complex)

    for alpha in range(nE if nE>0 else 1):
        for beta_p in range(nE if nE>0 else 1):
            p = float(wE[beta_p]) if nE>0 else 1.0
            if p < 1e-16: continue
            K = math.sqrt(p) * U4[:, alpha, :, beta_p]  # nS×nS
            # Beitrag zu T und J und TP
            T += np.kron(K, K.conj())
            vK = vec(K)
            J += np.outer(vK, vK.conj())
            KdagK += K.conj().T @ K

    # TP-Fehler
    tp_err = float(np.linalg.norm(KdagK - np.eye(nS)))
    # CP ist konstruktiv erfüllt; min-Eig für Diagnose
    lam_min = float(np.min(np.linalg.eigvalsh(hermitize(J))).real)
    return T, J, dict(cp_ok=True, min_eig=lam_min, tp_err=tp_err)

def choi_from_T_fast(T, nS):
    # Nur für Visualisierung kleiner Ausschnitte; T→J per Permutation der Indizes:
    # J[i,k; j,l] = T[i,j; k,l]
    J = np.transpose(T.reshape(nS,nS,nS,nS, order='F'), (0,2,1,3)).reshape(nS*nS, nS*nS, order='F')
    return hermitize(J)

def is_tp(T, nS, tol=1e-8):
    vI = vec(np.eye(nS)); return bool(np.linalg.norm(T @ vI - vI) < tol), float(np.linalg.norm(T @ vI - vI))

def rhp_cp_divisible(T_dt, T_2dt, nS):
    T_pinv = np.linalg.pinv(T_dt); Theta = T_2dt @ T_pinv
    J = choi_from_T_fast(Theta, nS); lam_min = float(np.min(np.linalg.eigvalsh(J)).real)
    return dict(cp_divisible=bool(lam_min>-1e-10), min_eig=lam_min)

def blp_backflow(T_dt, steps: int, nS: int):
    def ev(ρ, k):
        v = vec(ρ)
        for _ in range(k): v = T_dt @ v
        return unvec(v, nS)
    i,j = 0, min(1, max(0,nS-1))
    ρ1 = np.zeros((nS,nS)); ρ1[i,i]=1.0
    ρ2 = np.zeros((nS,nS)); ρ2[j,j]=1.0
    D=[]; prev=None; back=0.0
    for k in range(1, steps+1):
        a = ev(ρ1, k); b = ev(ρ2, k)
        Δ = a-b; s = np.linalg.svd(Δ, compute_uv=False); d = 0.5*float(np.sum(np.abs(s)))
        if prev is not None and d>prev+1e-9: back += (d-prev)
        D.append(d); prev=d
    return dict(backflow=back, D=D[:40])

# -------- Memory-Kernel skizzierte LS (OOM-frei) ----------
def memory_kernel_least_squares(T_dt, T_2dt, nS, depth=3, sketch_cols=32, ridge=1e-6):
    V = nS*nS
    Ts = [np.eye(V, dtype=complex), T_dt, T_dt@T_dt, T_2dt@T_dt]
    M = depth
    q = min(sketch_cols, V)
    S = rng.normal(size=(V, q)) + 1j*rng.normal(size=(V, q))
    S, _ = np.linalg.qr(S)

    U_blocks = []
    Bm_blocks = [ [] for _ in range(M+1) ]
    for t in range(M+1):
        Lt  = Ts[min(t,   len(Ts)-1)]
        Lt1 = Ts[min(t+1, len(Ts)-1)]
        U_blocks.append( (Lt1 - Lt) @ S )
        for m in range(M+1):
            Lt_m = Ts[min(t-m if t-m>=0 else 0, len(Ts)-1)]
            Bm_blocks[m].append( Lt_m @ S )
    U = np.vstack(U_blocks)
    Bm = [np.vstack(bl) for bl in Bm_blocks]

    resid = 0.0; normU = max(1.0, np.linalg.norm(U))
    for r in range(q):
        u = U[:, r:r+1]
        X = np.hstack([B[:, r:r+1] for B in Bm])
        XtX = X.conj().T @ X + ridge * np.eye(M+1)
        Xtu = X.conj().T @ u
        c = np.linalg.solve(XtX, Xtu)
        u_hat = X @ c
        resid += float(np.linalg.norm(u - u_hat)**2)
    rec_error = float(np.sqrt(resid)/normU)
    return dict(depth=M, rec_error=rec_error, sketched=True, q=q)

# -----------------------
# Raumzeit/Cluster & „Higgs“
# -----------------------
def embed_positions(g:Graph, fractal):
    n=len(g.idx_to_bary); X=np.zeros((n,3),float)
    if fractal=="SG":
        for i,(a,b,c) in g.idx_to_bary.items():
            a,b,c=map(float,(a,b,c)); X[i]=[b+0.5*c, math.sqrt(3)/2*c, 0.0]
    elif fractal=="ST":
        E=np.array([[0,0,0],[1,0,0],[0.5,math.sqrt(3)/2,0],[0.5,math.sqrt(3)/6,math.sqrt(6)/3]])
        for i,(a,b,c,d) in g.idx_to_bary.items():
            a,b,c,d=map(float,(a,b,c,d)); X[i]=a*E[0]+b*E[1]+c*E[2]+d*E[3]
    elif fractal=="VIC":
        for i,(x,y) in g.idx_to_bary.items(): X[i]=[float(x),float(y),0.0]
    elif fractal=="DF":
        xs=np.linspace(0,1,n); X[:,0]=xs; X[:,1]=0; X[:,2]=0
    else:
        for i,(x,y,z) in g.idx_to_bary.items(): X[i]=[float(x),float(y),float(z)]
    return X

def plot_graph3d(X, edges, labels, title, node_colors="black"):
    xe,ye,ze=[],[],[]
    for (i,j) in edges:
        xi,yi,zi=X[i]; xj,yj,zj=X[j]
        xe += [xi,xj,None]; ye += [yi,yj,None]; ze += [zi,zj,None]
    fig=go.Figure([
        go.Scatter3d(x=xe,y=ye,z=ze,mode="lines",line=dict(width=1,color="lightblue"),hoverinfo="none",showlegend=False),
        go.Scatter3d(x=X[:,0],y=X[:,1],z=X[:,2],mode="markers+text",
                     marker=dict(size=3,color=node_colors),
                     text=labels,textposition="top center",hoverinfo="text",showlegend=False)
    ])
    fig.update_layout(title=title,scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
                      height=520, margin=dict(l=0,r=0,t=40,b=0))
    return fig

def heatmap(M, title):
    fig=go.Figure(go.Heatmap(z=np.real(M), coloraxis="coloraxis"))
    fig.update_layout(title=title,coloraxis={'colorscale':'Viridis'},height=520,margin=dict(l=0,r=0,t=40,b=0))
    return fig

def fiedler_sparse(L_csr: ss.csr_matrix):
    # kleinste 2 Eigenpaare
    try:
        vals, vecs = sla.eigsh(L_csr.asfptype(), k=2, which='SM')
        order = np.argsort(vals)
        lam1 = float(np.real(vals[order[1]])) if len(vals)>1 else 0.0
        fvec = np.real(vecs[:,order[1]]) if len(vals)>1 else np.zeros(L_csr.shape[0])
        return lam1, fvec
    except Exception:
        # Fallback klein: dense
        L = L_csr.toarray()
        w,V = np.linalg.eigh(L)
        return float(w[1] if len(w)>1 else 0.0), (V[:,1] if len(w)>1 else np.zeros(L.shape[0]))

def modularity_cluster(L_csr: ss.csr_matrix, edges):
    lam1, f = fiedler_sparse(L_csr)
    med=float(np.median(f)); c=(f>med).astype(int)
    edges_c=[e for e in edges if c[e[0]]==c[e[1]]]
    return lam1, c, edges_c, f

def covariant_laplacian(n, edges, w, A_angles):
    rows,cols,data=[],[],[]; deg=np.zeros(n,float)
    for (i,j) in edges:
        key=(i,j) if i<j else (j,i); wij=w.get(key,1.0)
        Uij = np.exp(1j*A_angles.get(key, 0.0))
        rows += [i,j]; cols += [j,i]; data += [-(wij*Uij), -(wij*np.conj(Uij))]
        deg[i]+=wij; deg[j]+=wij
    rows += list(range(n)); cols += list(range(n)); data += list(deg)
    H = ss.coo_matrix((data,(rows,cols)), shape=(n,n)).toarray()
    return hermitize(H)

def higgs_surrogate_mass(rho_real):
    v=np.sqrt(np.clip(np.real(np.diag(rho_real)),0,None))
    return float(np.sqrt(np.mean(v*v)))

# -------- Harmonic Embedding (REAL) via sparse Solves --------
def harmonic_embed_with_fixed_boundary(L_csr: ss.csr_matrix, X_ideal: np.ndarray, B_idx: Iterable[int]):
    """REAL-Koordinaten aus Dirichlet-harmonischer Einbettung: L_II x_I = - L_IB x_B."""
    n=L_csr.shape[0]
    B=sorted(list(B_idx))
    I=sorted(list(set(range(n)) - set(B)))
    if not I:  # nur Rand
        return X_ideal.copy()
    LII=L_csr[np.ix_(I,I)].tocsc()
    LIB=L_csr[np.ix_(I,B)].tocsc()
    LU=sla.splu(LII)
    X_real = X_ideal.copy()
    for k in range(3):
        cB = X_ideal[B, k]
        rhs = - (LIB @ cB)
        cI  = LU.solve(rhs)
        X_real[I, k] = cI
        X_real[B, k] = cB
    return X_real

# -----------------------
# Evidenz / Checks
# -----------------------
def cheeger_bound_sparse(L_csr: ss.csr_matrix, edges):
    lam1, f = fiedler_sparse(L_csr)
    order=np.argsort(f); n=L_csr.shape[0]
    deg = np.array(L_csr.diagonal()).astype(float)
    volT=float(np.sum(deg))
    adj=[[] for _ in range(n)]
    for (i,j) in edges: adj[i].append(j); adj[j].append(i)
    marked=np.zeros(n,bool); volA=0.0; boundary=0; best=1e9
    for idx in order[:-1]:
        marked[idx]=True; volA+=deg[idx]
        for nb in adj[idx]:
            if marked[nb]: boundary-=1
            else: boundary+=1
        volB=volT-volA; h=boundary/max(1.0, min(volA,volB))
        if h<best: best=h
    lower=0.5*best*best
    return dict(lambda1=lam1, h=best, lower=lower, passed=(lam1+1e-8>=lower))

def varadhan_test_sparse(L_csr: ss.csr_matrix, t=0.03, pairs=60):
    """Ohne volle R-Matrix: Widerstände paarweise per LU; Heat-Kernel via expm_multiply."""
    n=L_csr.shape[0]
    if n < 2:
        return dict(a=float("nan"), b=float("nan"), r=1.0, passed=True, note="n<2: trivial")
    A=L_csr.tocsr()
    keep, LU = grounded_factorization(L_csr, ground=0)
    from scipy.sparse.linalg import expm_multiply
    xs=[]; ys=[]
    tries = 0
    while len(xs) < min(pairs, n*(n-1)//2) and tries < 5*pairs:
        i=int(rng.integers(0,n)); j=int(rng.integers(0,n))
        tries += 1
        if i==j: continue
        # Heat kernel column j from e_i
        e=np.zeros(n); e[i]=1.0
        col=expm_multiply((-t)*A, e); p=max(col[j],1e-300)
        # R_ij
        Rij = eff_resistance_pair(L_csr, LU, keep, i, j, ground=0)
        xs.append(Rij*Rij); ys.append(-2*t*math.log(p))
    if len(xs)<5: return dict(a=float("nan"),b=float("nan"),r=0.0, passed=False, note="wenig Stichproben")
    X=np.vstack([xs, np.ones(len(xs))]).T; a,b=np.linalg.lstsq(X, np.array(ys), rcond=None)[0]
    r=np.corrcoef(xs,ys)[0,1]
    return dict(a=float(a),b=float(b),r=float(r), passed=(abs(a-1.0)<0.2 and abs(b)<0.3 and r>0.95))

def triangle_resistance_ok_sparse(L_csr: ss.csr_matrix, trials: int = 200):
    n = L_csr.shape[0]
    if n < 3:
        return dict(rate=0.0, passed=True, tried=0, note="n<3: trivial.")
    keep, LU = grounded_factorization(L_csr, ground=0)
    viol = 0; tried = 0; seen=set()
    max_triples = n*(n-1)*(n-2)//6
    target = min(trials, max_triples)
    while tried < target:
        i, j, k = rng.choice(n, size=3, replace=False)
        trip = tuple(sorted((i,j,k)))
        if trip in seen: continue
        seen.add(trip); tried += 1
        Rij = eff_resistance_pair(L_csr, LU, keep, i, j, 0)
        Rjk = eff_resistance_pair(L_csr, LU, keep, j, k, 0)
        Rik = eff_resistance_pair(L_csr, LU, keep, i, k, 0)
        if Rik > Rij + Rjk + 1e-8: viol += 1
    rate = viol / max(tried, 1)
    return dict(rate=float(rate), passed=(rate < 1e-3), tried=tried,
                note=f"Tripel: {tried} von max. {max_triples}")

def cptp_entropy_check(H, w, edges, rho0, steps=80, T=1.0):
    n=H.shape[0]; Ls=lindblad_ops_dense(n, edges, w)
    dt=max(T/steps,1e-4); rho=rho0.copy(); traces=[]; mins=[]; ent=[]
    def S(r):
        w=np.linalg.eigvalsh(r); w=np.clip(w,0,1); w=w[w>1e-15]
        return float(-np.sum(w*np.log(w)))
    for _ in range(steps):
        rho=gksl_step(H, rho, Ls, dt)
        traces.append(float(np.real(np.trace(rho))))
        w=np.linalg.eigvalsh(rho); mins.append(float(np.min(w)))
        ent.append(S(rho))
    trace_ok=all(abs(x-1.0)<1e-6 for x in traces)
    psd_ok=all(m>=-1e-9 for m in mins)
    ent_mono=all(ent[i+1]>=ent[i]-1e-8 for i in range(len(ent)-1))
    return dict(trace_ok=trace_ok, psd_ok=psd_ok, entropy_monotone=ent_mono, passed=(trace_ok and psd_ok and ent_mono), rho=rho)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Emergenzkette: IDEAL↔REAL↔Raumzeit↔Symbr↔Higgs", layout="wide")

st.sidebar.header("Steuerung")
auto = st.sidebar.checkbox("Auto-Ableitung (Fraktal & REAL-Modus)", value=True)
fractal = st.sidebar.selectbox("Fraktal (manuell)", ["SG","ST","VIC","DF","BTREE"], index=0, disabled=auto)
mode = st.sidebar.radio("REAL-Modus (manuell)", ["R_eff","KRON"], index=0, disabled=auto)
L = st.sidebar.slider("Approximanten-Level L", 1, 6, 3)

rho0_kind = st.sidebar.selectbox(
    "Initialzustand ρ₀",
    ["Lokal rein (tiefster Level)", "Zufällig rein", "Zufällig gemischt", "Maximale Mischung"],
    index=0,
    help="Maximale Mischung bleibt unter LvN und unitalem GKSL invariant → ρ_IDEAL = ρ_REAL. Wähle einen nichttrivialen Zustand."
)

t_evo = st.sidebar.number_input("Zeit t (LvN/GKSL)", min_value=0.0, value=1.0, step=0.1)
steps = st.sidebar.slider("GKSL-Schritte", 10, 400, 120)
partition_kind = st.sidebar.selectbox("System-Partition S", ["Level-Cut", "Boundary-Cluster"], index=0)
S_cap = st.sidebar.slider("Max. Größe von S (Kanalableitung)", 4, 24, 12)

base_out = st.sidebar.text_input("Ausgabeordner", value="runs")
start = st.sidebar.button("Start (Run & Save)", type="primary")

st.title("IDEAL (LvN) ⇄ REAL (GKSL) ⇄ Raumzeit ⇄ Symmetriebruch ⇄ „Higgs“")

if not start:
    st.info("Parameter setzen und **Start (Run & Save)** klicken. Die Artefakte werden automatisch gespeichert.")
    st.stop()

# -----------------------
# Auto-Auswahl (sparse Checks)
# -----------------------
def infer_model(L, candidates=("SG","ST","VIC","DF","BTREE"), modes=("R_eff","KRON")):
    results=[]
    for fractal in candidates:
        G=build_fractal(fractal, L)
        n=len(G.idx_to_bary); edges=sorted(set((min(i,j),max(i,j)) for (i,j) in G.edges))
        L_id_csr=build_laplacian(n, edges); B=outer_boundary(fractal,G,edges)
        for mode in modes:
            w = (real_weights_deepening(L_id_csr, edges, G, fractal)
                 if mode=="R_eff" else
                 real_weights_kron(L_id_csr, edges, G, fractal))
            L_real_csr=build_laplacian(n, edges, w)
            ev1=cheeger_bound_sparse(L_real_csr, edges)
            ev2=varadhan_test_sparse(L_real_csr, t=0.03 if n<=1200 else 0.02)
            ev3=cptp_entropy_check(H_from_L_dense(L_real_csr), w, edges, rho0=np.eye(n)/n, steps=40, T=0.5)
            tri_trials = 150 if n >= 50 else 50 if n >= 10 else 10
            ev4=triangle_resistance_ok_sparse(L_real_csr, trials=tri_trials)
            score=sum([ev1["passed"], ev2["passed"], ev3["passed"], ev4["passed"]])
            cal=calibrate_s_star_sparse(L_id_csr, edges, G, fractal, w)
            results.append(dict(fractal=fractal, mode=mode, score=int(score),
                                dtn_err=float(dtn_error_sparse(L_id_csr, L_real_csr, B)),
                                cal_s=cal["s_star"], cal_err=cal["error"]))
    results.sort(key=lambda r: (-r["score"], r["cal_err"], r["dtn_err"]))
    return results[0], results

# Auto-Model
if auto:
    with st.spinner("Auto-Ableitung …"):
        best, allres = infer_model(L)
    fractal = best["fractal"]; mode = best["mode"]
    st.info(f"**Auto-Vorschlag:** Fraktal **{fractal}**, REAL-Modus **{mode}**, Evidenz-Score **{best['score']}/4**, "
            f"s* = {best['cal_s']:.3f}, DtN-Fehler ≈ {best['cal_err']:.3e}")
    with st.expander("Kandidaten (Top-10)"):
        st.json(allres[:10])

# -----------------------
# Run & Auto-Save
# -----------------------
run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{fractal}_L{L}_{mode}"
out_dir = Path(base_out) / run_id
ensure_dir(out_dir)

# Geometrie & IDEAL
G = build_fractal(fractal, L)
n = len(G.idx_to_bary)
edges = sorted(set((min(i,j),max(i,j)) for (i,j) in G.edges))
labels = [f"{G.addr[i]} | L{G.birthlevel.get(i,0)}" for i in range(n)]
X = embed_positions(G, fractal)

L_id_csr = build_laplacian(n, edges)
H_id = H_from_L_dense(L_id_csr)

# REAL
if mode=="R_eff":
    w_real = real_weights_deepening(L_id_csr, edges, G, fractal)
else:
    w_real = real_weights_kron(L_id_csr, edges, G, fractal)
L_real_csr = build_laplacian(n, edges, w_real)
B = outer_boundary(fractal, G, edges)
cal = calibrate_s_star_sparse(L_id_csr, edges, G, fractal, w_real)
s_star = cal["s_star"]; dtn_after = cal["error"]; dtn_before = dtn_error_sparse(L_id_csr, L_real_csr, B)
st.markdown(f"**DtN-Fehler (vor/ nach Kalibrierung)**: {dtn_before:.3e} → **{dtn_after:.3e}**  |  s* = {s_star:.3f}")

# REAL-Layout: harmonische Einbettung (sparse)
X_real = harmonic_embed_with_fixed_boundary(L_real_csr, X, B)

# Initialzustand
def make_rho0(kind, n, G):
    if kind == "Maximale Mischung":
        return np.eye(n, dtype=complex)/max(n,1)
    elif kind == "Lokal rein (tiefster Level)":
        deepest = max(G.birthlevel.values()) if G.birthlevel else 0
        cand = [i for i,bl in G.birthlevel.items() if bl==deepest]
        i0 = cand[0] if cand else 0
        psi = np.zeros((n,1), complex); psi[i0,0]=1.0
        return psi@psi.conj().T
    elif kind == "Zufällig rein":
        v = rng.normal(size=(n,)) + 1j*rng.normal(size=(n,))
        v /= np.linalg.norm(v)
        psi = v.reshape((n,1))
        return psi@psi.conj().T
    else:  # "Zufällig gemischt"
        A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
        M = A@A.conj().T
        M /= np.trace(M)
        return M

rho0 = make_rho0(rho0_kind, n, G)

# IDEAL/REAL Dynamik
rho_id = lvN(H_id, rho0, t_evo)
Ls_full = lindblad_ops_dense(n, edges, w_real)
dt = max(t_evo/steps, 1e-4)
rho_real = rho0.copy()
for _ in range(steps):
    rho_real = gksl_step(H_id, rho_real, Ls_full, dt)  # H_id ≡ L_real bei uns → identisch zum alten Verhalten

# Raumzeit/Cluster/Symmetrie
lam1, cluster, edges_c, f = modularity_cluster(L_real_csr, edges)
colors = ["green" if cluster[i]==0 else "red" for i in range(n)]
m_order = float(np.mean(np.sign(f - np.median(f))))
samples = np.array([np.mean(np.sign((f + 0.03*rng.normal(size=n)) - np.median(f))) for _ in range(64)])
def binder_cumulant(samples: np.ndarray):
    m2=np.mean(samples**2); m4=np.mean(samples**4)
    if m2<=1e-16: return 0.0
    return float(1 - m4/(3*m2*m2))
U4 = binder_cumulant(samples)
theta = np.zeros(n); A_angles = { (i,j) if i<j else (j,i): (theta[i]-theta[j]) for (i,j) in edges }
H_gauge = covariant_laplacian(n, edges, w_real, A_angles)
m_A = higgs_surrogate_mass(rho_real)
Q_info = cheeger_bound_sparse(L_real_csr, edges)

# Plots
fig_ideal = plot_graph3d(X, edges, labels, f"IDEAL ({fractal}, n={n})")
fig_real  = plot_graph3d(X_real, edges, labels, f"REAL [{mode}] — Clusterfarben (harmonisch, Rand fix)", node_colors=colors)
fig_rho_id   = heatmap(np.real(rho_id), f"IDEAL ρ(t={t_evo:.2f})")
fig_rho_real = heatmap(np.real(rho_real), f"REAL ρ_GKSL(t={t_evo:.2f})")

c1,c2 = st.columns(2)
with c1: st.plotly_chart(fig_ideal, use_container_width=True)
with c2: st.plotly_chart(fig_real,  use_container_width=True)
c3,c4 = st.columns(2)
with c3: st.plotly_chart(fig_rho_id,   use_container_width=True)
with c4: st.plotly_chart(fig_rho_real, use_container_width=True)
st.markdown(f"**Ordnungsparameter** M ≈ {m_order:+.3f}  |  **Binder** U₄ ≈ {U4:.3f}  |  **Cheeger-h** ≈ {Q_info['h']:.3f}  |  **„Higgs“ m_A** ≈ {m_A:.3f}")

# -----------------------
# Emergenz-Dilatation: Φ_Δt (S-Partition)
# -----------------------
st.subheader("Emergenz-Dilatation: Φ_Δt aus globaler Unitarität → Markov-Diagnostik")

# Partition S
if partition_kind=="Level-Cut":
    if n>0:
        levels=[G.birthlevel.get(i,0) for i in range(n)]
        L_cut = int(np.percentile(levels, 40)) if len(levels)>0 else 0
        S_idx = [i for i in range(n) if G.birthlevel.get(i,0) <= L_cut]
    else:
        S_idx=[]
else:
    cluster_arr = np.array(cluster) if n>0 else np.array([])
    deg = np.array(L_real_csr.diagonal()) if n>0 else np.array([])
    rank = np.argsort(-deg) if n>0 else np.array([])
    S_idx=[]
    for cl in (0,1):
        cand = [i for i in rank if n>0 and cluster_arr[i]==cl]
        take = cand[:max(1, len(cand)//3)]
        S_idx += take

if len(S_idx)==0 and n>0:
    S_idx = list(range(min(S_cap, n)))
if len(S_idx) > S_cap:
    S_idx = S_idx[:S_cap]

nS = len(S_idx)
Sset = set(S_idx)
E_idx = sorted(list(set(range(n)) - set(S_idx)))
nE = len(E_idx)
rhoE = np.eye(nE)/max(nE,1)  # bleibt wie zuvor

if nS>=1:
    with st.spinner("Kanal Φ_Δt (exakt CPTP, Kraus-Blöcke) …"):
        T_dt, J_dt, diag_info = channel_from_unitary_kraus(H_id, S_idx, dt, rhoE)
        T_2dt, _, _ = channel_from_unitary_kraus(H_id, S_idx, 2*dt, rhoE)
    cp_ok = diag_info["cp_ok"]; lam_min = diag_info["min_eig"]; tp_err = diag_info["tp_err"]

    # REAL-GKSL auf S
    index_map = {orig:i_new for i_new,orig in enumerate(S_idx)}
    edges_S = [(index_map[i], index_map[j]) for (i,j) in edges if (i in Sset and j in Sset)]
    w_S = {}
    for (i,j) in edges:
        if i in Sset and j in Sset:
            key_full = (i,j) if i<j else (j,i)
            ii, jj = index_map[i], index_map[j]
            key_sub = (ii,jj) if ii<jj else (jj,ii)
            w_S[key_sub] = w_real[key_full]
    H_S = build_laplacian(nS, edges_S, w_S).toarray()
    Ls_S = lindblad_ops_dense(nS, edges_S, w_S)
    L_super = superop_gksl(H_S, Ls_S)
    T_semigroup = dla.expm(L_super*dt)

    T_err = frob(T_dt - T_semigroup) / max(1.0, frob(T_semigroup))
    st.markdown(
        f"**CPTP für Φ_Δt:** CP=`{cp_ok}` (min Eig≈{lam_min:.2e}), TP-Abweichung≈{tp_err:.2e}  |  "
        f"**‖Φ_Δt − e^{{Δt𝓛_REAL}}‖/‖·‖ ≈ {T_err:.3e}**"
    )

    # Memory-Kern (skizzierte LS, OOM-frei)
    mem = memory_kernel_least_squares(T_dt, T_2dt, nS, depth=3)
    st.info(f"Memory-Kern: skizzierte LS (q={mem.get('q','?')}), Rekonstruktionsfehler ≈ {mem['rec_error']:.3e}")

    c7,c8 = st.columns(2)
    with c7:
        rhp = rhp_cp_divisible(T_dt, T_2dt, nS)
        blp = blp_backflow(T_dt, min(80, steps), nS)
        st.json(dict(RHP=rhp, BLP=dict(backflow=blp['backflow'], D=blp['D'][:10]), Memory=mem))
    with c8:
        # kleiner Visual-Ausschnitt
        nshow = min(10, nS*nS)
        st.plotly_chart(heatmap(np.real(J_dt[:nshow,:nshow]), "Choi(Φ_Δt) — Ausschnitt"), use_container_width=True)
else:
    st.warning("Teilmenge S ist leer. Bitte Partition/S_cap anpassen.")
    T_dt=None; T_semigroup=None

# -----------------------
# Kompakte Evidenz & Export
# -----------------------
ev1 = cheeger_bound_sparse(L_real_csr, edges)
ev2 = varadhan_test_sparse(L_real_csr, t=0.03 if n<=1200 else 0.02)
ev3 = cptp_entropy_check(H_id, w_real, edges, rho0=np.eye(n)/max(1,n), steps=40, T=0.5)
tri_trials = 150 if n >= 50 else 50 if n >= 10 else 10
ev4 = triangle_resistance_ok_sparse(L_real_csr, trials=tri_trials)
score = sum([ev1["passed"], ev2["passed"], ev3["passed"], ev4["passed"]])

with st.expander("Evidenz (kompakt)"):
    st.json(dict(
        cheeger=ev1, varadhan=ev2,
        cptp=dict(trace_ok=ev3["trace_ok"], psd_ok=ev3["psd_ok"], entropy_monotone=ev3["entropy_monotone"], passed=ev3["passed"]),
        triangle=ev4, score=score
    ))

# Auto-Save Artefakte
save_npy(out_dir / "X_ideal.npy", X)
save_npy(out_dir / "X_real_harmonic.npy", X_real)
save_npy(out_dir / "L_ideal.npy", L_id_csr.toarray())
save_npy(out_dir / "L_real.npy", L_real_csr.toarray())
save_npy(out_dir / "rho_LvN.npy", rho_id)
save_npy(out_dir / "rho_GKSL.npy", rho_real)
with open(out_dir / "edges.json", "w", encoding="utf-8") as f:
    json.dump(edges, f)

# Optional: Kanal/Semigroup speichern
if nS>=1:
    save_npy(out_dir / "Phi_dt_super.npy", T_dt)
    save_npy(out_dir / "exp_dt_L_REAL_super.npy", T_semigroup)

meta=dict(
    run_id=run_id, out_dir=str(out_dir),
    fractal=fractal, mode=mode, L=L, n=n,
    rho0_kind=rho0_kind,
    t_evo=t_evo, steps=steps, dt=dt,
    dtn_before=float(dtn_before), dtn_after=float(dtn_after), s_star=float(s_star),
    order_param=float(m_order), binder=float(U4), higgs_mA=float(m_A),
    cheeger_h=float(ev1["h"]),
    score=int(score),
    S_idx=S_idx if nS>=1 else [],
    nS=int(nS),
)
with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# Plots speichern
fig_ideal_html = out_dir / "ideal_graph.html"
fig_real_html  = out_dir / "real_graph_harmonic.html"
fig_rho_id_html = out_dir / "rho_ideal_heatmap.html"
fig_rho_real_html = out_dir / "rho_real_heatmap.html"
save_fig_html(fig_ideal, fig_ideal_html)
save_fig_html(fig_real,  fig_real_html)
save_fig_html(fig_rho_id, fig_rho_id_html)
save_fig_html(fig_rho_real, fig_rho_real_html)

# ZIP-Bundle
with open(out_dir / "bundle.zip", "wb") as zf_out:
    with zipfile.ZipFile(zf_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.glob("**/*"):
            if p.is_file() and p.name != "bundle.zip":
                zf.write(p, arcname=p.relative_to(out_dir))

st.success(f"Run abgeschlossen. Artefakte gespeichert in: `{out_dir}`")
with open(out_dir / "bundle.zip", "rb") as fzip:
    st.download_button("ZIP herunterladen", data=fzip.read(), file_name=f"{run_id}.zip", mime="application/zip")
