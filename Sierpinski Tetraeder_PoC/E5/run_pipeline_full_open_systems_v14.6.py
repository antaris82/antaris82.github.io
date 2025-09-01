#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline_full_open_systems.py — TI-on-ST (FULL Lindblad) v14.0

- ST-Graph (Sierpinski-Tetraeder) + Adressbaum
- Heat-Kernel K_t, Gibbs-Zustand ρ(β), σ=C ρ C^† auf Pixelmoden
- Pixel-POVM aus JSON-Detektor-Geometrie + optionale unitäre Mischung (z.B. 50/50-Beamsplitter)
- Schrödinger-Dispersion (unitär) und OFFENES SYSTEM mit **voller** Lindblad-Summe:
  * Site-Dephasing als Σ_i L_i ρ L_i - 1/2 {L_i^†L_i, ρ}, L_i = sqrt(γ_site)|i><i|
  * Pixel-Dephasing/Detektoren als Rang-1 Sprungoperatoren |χ_k><χ_k|
- (Optional) MCWF-Trajektorien (observed channels = Detektoren)
- Sub-Gaussian-Checks, Refutation-Suite (KL/χ²/Wilson, d_s-Fit)
- **Neue Ausgaben**: L-Heatmaps/Sparsity, Ks/Kts/Pt-Visuals, eigs_L.csv, dispersion_width.csv,
  TI/rho_open_final.npy, TI/rho_open_series.npz, TI/mode_vecs.npy, TI/unitary_U.npy,
  TI/SigmaK.npy, TI/SigmaK_mixed.npy, SG/R_diag.npy, Trajektorien-Events.csv

Hinweis:
- Diese FULL-Variante ist absichtlich „langsam“ für γ_site>0 (O(N) Sprungoperatoren).
  Die FAST-Variante ersetzt Site-Dephasing durch die geschlossene Form γ(diag(ρ)-ρ).
"""

from __future__ import annotations
import os, sys, re, json, math, argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, TwoSlopeNorm

# ===== LIMITS =====
MAX_LEVEL     = 7
MAX_NODES     = 50000
MAX_PIXELS    = 64
MAX_TRAJ      = 200000
MAX_STEPS     = 200000
MAX_SPEC_LVLS = 10
SAFE_PATH_MAX = 260

# ===== UTILS / LOG =====
def ensure_dir(p: str):
    p = os.path.normpath(p)
    if len(p) > SAFE_PATH_MAX: raise ValueError("Output path too long.")
    os.makedirs(p, exist_ok=True); return p
def info(msg: str): print(msg, flush=True)
def progress_start(stage: str, total_steps: int): info(f"[{stage}] starting … 0/{total_steps}")
def progress_step(stage: str, step: int, total_steps: int, msg: str): info(f"[{stage}] {step}/{total_steps}  {msg}")
def progress_done(stage: str): info(f"[{stage}] ✓ done")
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def assert_range(name: str, val: float, lo: float, hi: float):
    if not (lo <= val <= hi): raise ValueError(f"Argument {name}={val} out of range [{lo},{hi}]")
def normalized(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)); return v / n if n > 0 else v
def is_psd(M: np.ndarray, tol=1e-10):
    Ms = (M + M.T.conj()) / 2
    w = np.linalg.eigvalsh(Ms)
    return bool(np.all(w >= -tol)), float(w.min()), float(w.max())
def as_complex(x: np.ndarray) -> np.ndarray:
    return x.astype(np.complex128, copy=False) if not np.iscomplexobj(x) else x.astype(np.complex128, copy=False)

# ===== SAFE COMPLEX PARSER =====
_ALLOWED_CHARS = set("0123456789+-eE.jJ ")
def _parse_complex_scalar(x: Any) -> complex:
    if isinstance(x, complex): return complex(x)
    if isinstance(x, (int, float, np.number)): return complex(float(x))
    if isinstance(x, (list, tuple)) and len(x) == 2: return complex(float(x[0]), float(x[1]))
    if isinstance(x, str):
        s = x.strip()
        if len(s) > 64: raise ValueError("complex string too long")
        if not set(s) <= _ALLOWED_CHARS: raise ValueError("invalid char in complex string")
        return complex(s)
    raise TypeError(f"Unsupported complex scalar: {type(x)}")
def parse_unitary(obj: Any) -> Optional[np.ndarray]:
    if obj is None: return None
    arr = np.array(obj, dtype=object)
    if arr.ndim == 3 and arr.shape[-1]==2:
        K = arr.shape[0]
        U = np.empty((K,K), dtype=np.complex128)
        for i in range(K):
            for j in range(K):
                U[i,j] = complex(float(arr[i,j,0]), float(arr[i,j,1]))
        return U
    if arr.ndim == 2 and arr.shape[0]==arr.shape[1]:
        K=arr.shape[0]; U=np.empty((K,K), dtype=np.complex128)
        for i in range(K):
            for j in range(K):
                U[i,j] = _parse_complex_scalar(arr[i,j])
        return U
    raise ValueError("unitary must be KxK or KxKx2 (Re,Im)")

# ===== exp for symmetric =====
def expm_sym(M: np.ndarray, scale: float) -> np.ndarray:
    M = np.array(M, dtype=float, copy=False)
    w, V = np.linalg.eigh(M)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        E = np.exp(scale * w)
    E[~np.isfinite(E)] = 0.0
    return (V * E) @ V.T

# ===== ST GRAPH =====
def build_st_with_addresses(level: int):
    if level < 0 or level > MAX_LEVEL: raise ValueError(f"level must be 0..{MAX_LEVEL}")
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3)/2, 0.0])
    V3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])
    corners = [V0, V1, V2, V3]
    nodes = [V0.copy(), V1.copy(), V2.copy(), V3.copy()]
    addrs = ["" for _ in nodes]
    edges = {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
    def key_of(p, s): return tuple(np.rint(p*s).astype(int))
    scale = 1
    for _ in range(level):
        scale *= 2
        key2: Dict[Tuple[int,int,int], List[Any]] = {}
        new_edges = set()
        for ci, Vi in enumerate(corners):
            local = {}
            for uid, up in enumerate(nodes):
                p = 0.5*(up + Vi)
                k = key_of(p, scale)
                adr = addrs[uid] + str(ci)
                if k not in key2:
                    key2[k] = [len(key2), p, adr]
                else:
                    if adr < key2[k][2]: key2[k][2] = adr
                local[uid] = key2[k][0]
            for (a,b) in edges:
                na, nb = local[a], local[b]
                if na != nb:
                    i, j = (na, nb) if na < nb else (nb, na)
                    new_edges.add((i,j))
        N = len(key2)
        if N > MAX_NODES: raise MemoryError(f"Graph too large (N={N} > {MAX_NODES}). Reduce --level.")
        nodes = [None]*N; addrs = [None]*N
        for _, (nid, p, adr) in key2.items():
            nodes[nid] = p; addrs[nid] = adr
        edges = new_edges
    N = len(nodes)
    A = np.zeros((N,N), float)
    for (i,j) in edges: A[i,j] = 1.0; A[j,i] = 1.0
    coords = np.array(nodes, float)
    boundary = [0,1,2,3] if N >= 4 else list(range(min(4, N)))
    return A, coords, addrs, boundary

# ===== RENORM =====
def estimate_r_harmonic(A: np.ndarray, boundary: List[int]) -> float:
    N = A.shape[0]; bset=set(boundary)
    interior = [i for i in range(N) if i not in bset]
    if not interior: return 1.0
    D = np.diag(A.sum(axis=1)); L = D - A
    Lii = L[np.ix_(interior, interior)]
    Lib = L[np.ix_(interior, boundary)]
    def energy(u):
        dif = u[:,None] - u[None,:]
        return 0.5 * float(np.sum(A * (dif**2)))
    ratios = []
    patterns = [np.array(x, float) for x in
                ([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                 [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1])]
    for ub in patterns:
        rhs = -Lib @ ub
        try: ui = np.linalg.solve(Lii, rhs)
        except np.linalg.LinAlgError: ui = np.linalg.lstsq(Lii, rhs, rcond=None)[0]
        u = np.zeros(N); u[boundary] = ub; u[interior] = ui
        E = energy(u); Eb = 0.0
        for i in range(4):
            for j in range(i+1,4): Eb += (ub[i]-ub[j])**2
        if Eb > 0: ratios.append(E/Eb)
    return float(np.median(ratios)) if ratios else 1.0

# ===== SELECTORS / PIXEL-POVM =====
def point_in_polyhedron_convex(pt: np.ndarray, faces: List[np.ndarray]) -> bool:
    if not faces: return False
    allv = np.vstack(faces); c = allv.mean(axis=0)
    for face in faces:
        if face.shape[0] < 3: continue
        p0, p1, p2 = face[0], face[1], face[2]
        n = np.cross(p1-p0, p2-p0); nrm = float(np.linalg.norm(n))
        if nrm < 1e-14: continue
        if np.dot(n, c - p0) > 0: n = -n
        if np.dot(n, pt - p0) > 1e-12: return False
    return True

def sel_mask(sel: Dict[str,Any], N:int, coords:np.ndarray, addrs:List[str]) -> np.ndarray:
    v = np.zeros(N, float)
    t = str(sel.get("type","")).lower()
    if t == "nodes":
        for i in sel.get("ids",[]):
            i=int(i)
            if 0<=i<N: v[i]+=1.0
    elif t == "address":
        pref = sel.get("prefix", []); pref = [pref] if isinstance(pref, str) else pref
        pref = [str(p) for p in pref]
        for i,a in enumerate(addrs):
            if any(a.startswith(p) for p in pref): v[i]+=1.0
    elif t == "sphere":
        c = np.array(sel.get("center",[0,0,0]), float); r = float(sel.get("radius",0.0))
        if r>0:
            d2 = np.sum((coords - c[None,:])**2, axis=1); v[d2 <= r*r] += 1.0
    elif t == "gauss3d":
        c = np.array(sel.get("center",[0,0,0]), float)
        sig = sel.get("sigma",0.2)
        if isinstance(sig,(int,float)): Sigma = np.eye(3)*(float(sig)**2)
        else:
            s = np.array(sig,float).ravel(); Sigma = np.diag(s**2) if s.size==3 else np.eye(3)*(0.2**2)
        try: SigInv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError: SigInv = np.linalg.pinv(Sigma)
        x = coords - c[None,:]; v += np.exp(-0.5*np.sum(x*(x@SigInv), axis=1))
    elif t == "cone":
        apex = np.array(sel.get("apex",[0,0,0]), float)
        axis = np.array(sel.get("axis",[0,0,1]), float)
        n = float(np.linalg.norm(axis)); axis = axis/n if n>0 else np.array([0,0,1.0])
        ang = float(sel.get("angle_deg",20.0))*math.pi/180.0
        rmin = float(sel.get("r_min",0.0)); rmax=float(sel.get("r_max",np.inf))
        mode = str(sel.get("mode","binary")).lower()
        ang_sigma = float(sel.get("angle_sigma_deg", max(5.0, ang*180/math.pi/2)))*math.pi/180.0
        r_sigma = float(sel.get("r_sigma",0.0))
        dx = coords - apex[None,:]
        r = np.linalg.norm(dx,axis=1)+1e-15
        cos_th = (dx@axis)/r; cos_th=np.clip(cos_th,-1,1); th=np.arccos(cos_th)
        in_r = (r>=rmin-1e-12)&(r<=rmax+1e-12)
        if mode=="gaussian":
            w_ang = np.exp(-0.5*(th/ang_sigma)**2)
            w_rad = np.ones_like(r) if r_sigma<=0 else np.exp(-0.5*((r-(rmin+rmax)/2)/r_sigma)**2)
            v += w_ang*w_rad*in_r.astype(float)
        else:
            v += ((th<=ang)&in_r).astype(float)
    elif t == "polyhedron":
        faces=[]
        for F in sel.get("faces",[]):
            pts = np.array(F.get("points",[]), float)
            if pts.ndim==2 and pts.shape[0]>=3: faces.append(pts)
        if faces:
            for i in range(N):
                if point_in_polyhedron_convex(coords[i], faces): v[i]+=1.0
    return v

def pixel_mode_projection(pixels: List[np.ndarray]) -> np.ndarray:
    K = len(pixels); N = len(pixels[0])
    if K<1 or K>MAX_PIXELS: raise ValueError(f"pixels K must be 1..{MAX_PIXELS}")
    C = np.zeros((K,N), np.complex128)
    for k,p in enumerate(pixels):
        s = float(p.sum()); w = p / (s+1e-16)
        C[k,:] = np.sqrt(w.astype(np.complex128))
    return C

def povm_from_pixels(pixels: List[np.ndarray], effs: np.ndarray):
    N = len(pixels[0]); K = len(pixels)
    if effs.size != K: raise ValueError("effs size mismatch")
    W = np.zeros((K,N))
    for k,p in enumerate(pixels):
        w = p/(p.sum()+1e-16)
        W[k,:] = float(effs[k])*w
    s = W.sum(axis=0)
    overflow = s>1.0+1e-12
    if np.any(overflow): W[:,overflow] = W[:,overflow]/s[overflow]
    return [np.diag(W[k,:]) for k in range(K)]

# ===== JSON PIXELS =====
def load_pixels_from_json(path_json: str, N:int, coords:np.ndarray, addrs:List[str]):
    if not path_json: return None
    if not os.path.exists(path_json): raise FileNotFoundError(f"pixels_json not found: {path_json}")
    with open(path_json,"r",encoding="utf-8-sig") as f:
        try: cfg=json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON ({e.lineno}:{e.colno}) {e.msg} — JSON must use double quotes; no comments/trailing commas.")
    combine = str(cfg.get("combine","union")).lower()
    pixels,effs,names=[],[],[]
    px = cfg.get("pixels",[])
    if not isinstance(px,list) or len(px)==0: raise ValueError("pixels_json: field 'pixels' must be non-empty list")
    if len(px) > MAX_PIXELS: raise ValueError(f"too many pixels (>{MAX_PIXELS})")
    for p in px:
        sel = p.get("selectors",[])
        if not isinstance(sel,list) or len(sel)==0: raise ValueError("pixel without selectors")
        masks=[sel_mask(s,N,coords,addrs) for s in sel]
        if combine=="sum": m=np.sum(masks,axis=0)
        elif combine=="max": m=np.max(masks,axis=0)
        else: m=np.clip(np.sum([(mm>0).astype(float) for mm in masks],axis=0),0,1)
        if m.sum() <= 0: raise ValueError("pixel selects no nodes")
        pixels.append(m); effs.append(float(p.get("efficiency",1.0))); names.append(str(p.get("name", f"pix{len(names)}")))
    U=None
    if "unitary" in cfg:
        try: U = parse_unitary(cfg["unitary"])
        except Exception as e: raise ValueError(f"invalid 'unitary' in JSON: {e}")
    return pixels, np.array(effs,float), names, U

def beamsplitter_50_50() -> np.ndarray:
    return (1.0/np.sqrt(2.0))*np.array([[1.0, 1.0j],[1.0j, 1.0]], np.complex128)

# ===== SCHRÖDINGER =====
def schrodinger_propagate(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    H = as_complex(H); psi0 = as_complex(psi0)
    w,V=np.linalg.eigh((H+H.conj().T)/2)
    ph=np.exp(-1j*t*w)
    return (V*ph) @ (V.T.conj() @ psi0)
def gaussian_on_coords(coords,c,sigma):
    d2=np.sum((coords-c[None,:])**2,axis=1)
    g=np.exp(-0.5*d2/(sigma**2))
    return normalized(g)
def packet_width_from_prob(prob, coords, c):
    d2 = np.sum((coords-c[None,:])**2,axis=1)
    return float(np.sum(prob*d2))

# ===== VISUALIZATION HELPERS (downsample-only; raw bleibt unverändert) =====
def _add_cbar(im, label:str="", shrink=0.85, log=False):
    cb=plt.colorbar(im, shrink=shrink)
    if label: cb.set_label(label)
    return cb
def _downsample2d(M: np.ndarray, max_dim: int = 1200, mode: str = "auto") -> np.ndarray:
    M = np.asarray(M)
    if mode == "full":
        return M
    h, w = M.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return M
    s = int(math.ceil(m / float(max_dim)))
    if mode in ("auto", "stride"):
        return M[::s, ::s]
    elif mode == "block":
        h2 = (h // s) * s
        w2 = (w // s) * s
        M2 = M[:h2, :w2]
        M2 = M2.reshape(h2//s, s, w2//s, s).mean(axis=(1,3))
        return M2
    else:
        return M

def safe_imshow_log(M, cmap="plasma", label="value (log)", fname=None,
                    viz_mode="auto", viz_max_dim=1200, viz_dpi=160):
    if viz_mode == "off": return
    M_plot = _downsample2d(np.array(M, float, copy=False), max_dim=int(viz_max_dim), mode=str(viz_mode))
    M_plot = M_plot.astype(np.float32, copy=False)
    Mpos = np.maximum(M_plot, np.nanmax(M_plot[M_plot>0])*1e-16 if np.any(M_plot>0) else 1e-16)
    vmin = float(np.nanmin(Mpos)); vmax=float(np.nanmax(Mpos))
    if not (vmin < vmax): vmax = vmin*10
    plt.figure(figsize=(6,5))
    im=plt.imshow(Mpos, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    _add_cbar(im, label=label, shrink=0.9)
    if fname: plt.tight_layout(); plt.savefig(fname, dpi=int(viz_dpi)); plt.close()

# ===== RESISTANCE & SG =====
def resistance_matrix_from_L(L: np.ndarray) -> np.ndarray:
    N=L.shape[0]
    try: Linv=np.linalg.inv(L)
    except np.linalg.LinAlgError:
        w,V=np.linalg.eigh(L); w_plus=np.where(w>1e-12,1.0/w,0.0); Linv=(V*w_plus)@V.T
    d=np.diag(Linv); R=d[:,None]+d[None,:]-2*Linv
    return np.maximum(R,0.0)

def subgaussian_check(Kt_list: List[np.ndarray], t_list: List[float], R: np.ndarray,
                      d_w: float, out_dir: str, ref_idx: int=0):
    rows=[]
    for Kt,t in zip(Kt_list,t_list):
        p=np.maximum(np.array(Kt[ref_idx,:], float),1e-300)
        d=R[ref_idx,:]
        X=(np.power(d,d_w)/max(t,1e-12))**(1.0/(d_w-1.0))
        Y=-np.log(p)
        m=(np.isfinite(X)&np.isfinite(Y)&(d>1e-14)); Xf,Yf=X[m],Y[m]
        if Xf.size>=2:
            A=np.vstack([Xf,np.ones_like(Xf)]).T
            slope,intercept=np.linalg.lstsq(A,Yf,rcond=None)[0]
            Yhat=slope*Xf+intercept
            ssres=float(np.sum((Yf-Yhat)**2)); sstot=float(np.sum((Yf-Yf.mean())**2))
            R2=1.0-ssres/max(sstot,1e-16)
        else:
            slope=intercept=R2=float("nan")
        rows.append({"t":t,"slope":slope,"intercept":intercept,"R2":R2,"n":int(Xf.size)})
        plt.figure(figsize=(6,3.8)); plt.scatter(Xf,Yf,s=10)
        if np.isfinite(slope):
            xs=np.linspace(Xf.min(),Xf.max(),100); plt.plot(xs,slope*xs+intercept,lw=2,label=f"slope={slope:.3g}, R²={R2:.3f}"); plt.legend()
        plt.title(f"Sub-Gaussian check @ t={t:.3g} (ref idx={ref_idx})")
        plt.xlabel(r"X=((d_R^{d_w})/t)^{1/(d_w-1)}"); plt.ylabel(r"Y=-\log p_t(x,y)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"sg_t_{t:.3g}.png"), dpi=160); plt.close()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"subgaussian_fits.csv"), index=False)

# ===== GKSL (FULL) & TRAJ =====
def lindblad_rhs_full(rho: np.ndarray, H: np.ndarray,
                      L_ops: List[Tuple[float, np.ndarray]]) -> np.ndarray:
    """
    dρ/dt = -i[H,ρ] + ∑_m γ_m (L_m ρ L_m^† - 1/2{L_m^†L_m, ρ})
    FULL: beinhaltet explizit alle Site-Dephasing-L_m = |i><i| * sqrt(γ_site)
    """
    rho = as_complex(rho); H = as_complex(H)
    comm = -1j*(H @ rho - rho @ H)
    dissip = np.zeros_like(rho)
    for g, L in L_ops:
        if g <= 0: continue
        L = as_complex(L)
        Jr = L @ rho @ L.conj().T
        M  = L.conj().T @ L
        dissip += float(g)*(Jr - 0.5*(M @ rho + rho @ M))
    return comm + dissip

def integrate_master_rk4_full(rho0: np.ndarray, H: np.ndarray, L_ops: List[Tuple[float,np.ndarray]],
                              dt: float, steps: int, store_stride:int=0, renorm_trace:bool=True):
    rho = as_complex(rho0).copy(); series = []
    for n in range(steps):
        k1 = lindblad_rhs_full(rho, H, L_ops)
        k2 = lindblad_rhs_full(rho + 0.5*dt*k1, H, L_ops)
        k3 = lindblad_rhs_full(rho + 0.5*dt*k2, H, L_ops)
        k4 = lindblad_rhs_full(rho + dt*k3, H, L_ops)
        rho = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        rho = (rho + rho.conj().T)/2
        if renorm_trace:
            tr = float(np.trace(rho).real)
            if tr>0: rho = rho / tr
        if store_stride and (n % store_stride == 0): series.append(rho.copy())
    return rho, series

def mcwf_trajectory_full(psi0: np.ndarray, H: np.ndarray,
                         observed: List[Tuple[str,float,np.ndarray]],
                         unobserved: List[Tuple[float,np.ndarray]],
                         dt: float, tmax: float, rng: np.random.Generator):
    """
    MCWF mit beobachteten (Detektor) und unbeobachteten Kanälen.
    Für FULL-Variante werden unbeobachtete als no-jump via H_eff berücksichtigt (Effizienz).
    """
    psi = as_complex(psi0).copy()
    # Effektives Heff = H - (i/2) Σ L_m^†L_m für unbeobachtete
    Heff = H.copy()
    for g,L in unobserved:
        if g<=0: continue
        M = (L.conj().T @ L).astype(np.complex128)
        Heff = Heff - 0.5j * float(g) * M
    t = 0.0
    while t < tmax - 1e-15:
        psi = psi - 1j*dt*(Heff @ psi)
        n2 = float(np.vdot(psi, psi).real)
        if not np.isfinite(n2) or n2 <= 0: return {"clicked":False, "channel":"lost", "time":t}
        psi = psi / math.sqrt(n2)
        # beobachtete Sprünge
        weights = []
        amps = []
        for lbl, g, L in observed:
            v = L @ psi
            p = float(g) * dt * float(np.vdot(v, v).real)
            weights.append(max(0.0, p))
            amps.append(v)
        tot = sum(weights)
        if tot > 1.0:
            s = 1.0/tot; weights=[w*s for w in weights]; tot=1.0
        r = rng.random()
        if r < tot:
            acc=0.0
            for (lbl,g,L), w, v in zip(observed, weights, amps):
                acc += w
                if r <= acc:
                    psi = v / (np.linalg.norm(v)+1e-16)
                    if lbl.startswith("pixel:"):
                        return {"clicked":True, "channel":lbl.split(":",1)[1], "time":t+dt}
                    break
        t += dt
    return {"clicked":False, "channel":None, "time":tmax}

# ===== BUILD =====
def build_artifacts(artifacts_dir: str, level:int, t:float, s:float, beta:float,
                    bc:str, use_fixed_r:bool):
    progress_start("BUILD", 5)
    A, coords, addrs, bndry = build_st_with_addresses(level); progress_step("BUILD",1,5,"ST graph built")
    if A.shape[0] > MAX_NODES: raise MemoryError("Graph exceeds MAX_NODES")
    D=np.diag(A.sum(axis=1)); L=D-A
    if bc.lower().startswith("dir"):
        keep=[i for i in range(A.shape[0]) if i not in set(bndry)]
        idx=np.array(keep,int)
        Aop=A[np.ix_(idx,idx)]; Lop=L[np.ix_(idx,idx)]
        coords_eff=coords[idx]; addrs_eff=[addrs[i] for i in idx]
    else:
        Aop=A; Lop=L; coords_eff=coords; addrs_eff=addrs
    progress_step("BUILD",2,5,"boundary conditions applied")
    r_est = estimate_r_harmonic(A,bndry)
    r_use = (2.0/3.0) if use_fixed_r else r_est
    time_scale = r_use**(-level)
    progress_step("BUILD",3,5,f"renorm r_est={r_est:.6g}, r_use={r_use:.6g}")

    Kt = expm_sym(Lop, -float(t)*time_scale)
    Ks = expm_sym(Lop, -float(s)*time_scale)
    Kts= expm_sym(Lop, -(float(t)+float(s))*time_scale)
    deg=Aop.sum(axis=1); Dm=np.diag(1/np.sqrt(np.maximum(deg,1.0))); Dh=np.diag(np.sqrt(np.maximum(deg,1.0)))
    S=Dm@Aop@Dm - np.eye(Aop.shape[0]); Pt= Dm @ expm_sym(S, +float(t)) @ Dh
    G=expm_sym(Lop,-float(beta)*time_scale); Z=float(np.trace(G).real); rho=(G/Z).astype(np.complex128)

    # zwei Pixel-Probe-Moden für sigma
    score=coords_eff[:,0] + 0.3*coords_eff[:,1] - 0.2*coords_eff[:,2]
    i1=int(np.argmin(score)); i2=int(np.argmax(score))
    e1=np.zeros(Lop.shape[0]); e1[i1]=1.0; e2=np.zeros(Lop.shape[0]); e2[i2]=1.0
    C2=np.vstack([normalized(e1), normalized(e2)]); sigma=(C2@rho@C2.conj().T).astype(np.complex128)

    ensure_dir(artifacts_dir)
    np.save(os.path.join(artifacts_dir,"A.npy"),Aop)
    np.save(os.path.join(artifacts_dir,"L.npy"),Lop)
    np.save(os.path.join(artifacts_dir,"K_t.npy"),Kt)
    np.save(os.path.join(artifacts_dir,"K_s.npy"),Ks)
    np.save(os.path.join(artifacts_dir,"K_ts.npy"),Kts)
    np.save(os.path.join(artifacts_dir,"P_t.npy"),Pt)
    np.save(os.path.join(artifacts_dir,"rho_beta.npy"),rho)
    np.save(os.path.join(artifacts_dir,"sigma.npy"),sigma)
    np.save(os.path.join(artifacts_dir,"coords.npy"),coords_eff)
    with open(os.path.join(artifacts_dir,"addresses.json"),"w",encoding="utf-8") as f:
        json.dump({"addresses":addrs_eff},f,indent=2)
    with open(os.path.join(artifacts_dir,"summary.json"),"w",encoding="utf-8") as f:
        json.dump({"level":level,"r_estimate":r_est,"r_used":r_use,"n_effective":int(Lop.shape[0]),"bc":bc}, f, indent=2)
    progress_step("BUILD",4,5,"kernels and ρ constructed")
    progress_step("BUILD",5,5,"artifacts saved"); progress_done("BUILD")

# [PATCH B] --- Stats helpers for robust TI validation (exact p, G-test, CIs, TOST) ---
def _try_scipy_chi2_sf(x, df):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:
        return None

def chi2_p_exact(chi2_stat: float, df: int) -> float:
    """Exact chi^2 tail if SciPy available, else stable normal-approx fallback."""
    p = _try_scipy_chi2_sf(chi2_stat, df)
    if p is not None:
        return p
    z = (chi2_stat - df) / math.sqrt(2.0 * df)
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def g_test(counts: List[int], probs: List[float]):
    """Likelihood-ratio G-test (goodness-of-fit) with exact chi^2 tail if possible."""
    N = float(sum(counts))
    exp = [max(N * p, 1e-12) for p in probs]
    g = 0.0
    for o, e in zip(counts, exp):
        if o > 0:
            g += 2.0 * o * math.log(o / e)
    df = max(len(counts) - 1, 1)
    return g, chi2_p_exact(g, df)

def cramers_v(chi2_stat: float, N: int, k: int) -> float:
    if N <= 0 or k <= 1:
        return float('nan')
    return math.sqrt(chi2_stat / (N * (k - 1)))

def js_divergence(p: List[float], q: List[float]) -> float:
    """Jensen–Shannon divergence (base e)."""
    def _safe_log(x): return math.log(x) if x > 0 else -1e9
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0:
                s += ai * (_safe_log(ai) - _safe_log(bi if bi > 0 else 1e-300))
        return s
    return 0.5 * (kl(p, m) + kl(q, m))

def total_variation(p: List[float], q: List[float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))

def _norm_ppf(u: float) -> float:
    # Acklam-Approximation für Invers-Normal (ausreichend für CI-Bounds)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
          138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
          66.80131188771972, -13.28068155288572]
    c = [-7.784894002430293e-3, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734,  4.374664141464968,  2.938163982698783]
    d = [ 7.784695709041462e-3,  0.3224671290700398,  2.445134137142996,
          3.754408661907416]
    if not (0.0 < u < 1.0):
        return float('inf') if u > 0 else float('-inf')
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow:
        q = math.sqrt(-2.0 * math.log(u))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if u > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - u))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = u - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else abs(_norm_ppf(1 - alpha/2))
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z*z)/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def tost_equivalence_binomial(k: int, n: int, p0: float, eps_abs: float, alpha: float = 0.05):
    """TOST (z-approx) für eine Binomialproportion vs. p0 mit absoluter Marge eps_abs."""
    if n == 0:
        return False, float('nan'), float('nan')
    phat = k / n
    se = math.sqrt(p0*(1-p0)/n) if n > 0 else float('inf')
    # zwei einseitige Tests: wir wollen BEIDE H0 verwerfen
    z_lo = ((phat - p0) + eps_abs) / se   # H0: phat - p0 <= -eps_abs
    z_hi = ((phat - p0) - eps_abs) / se   # H0: phat - p0 >=  eps_abs
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p1 = 1.0 - Phi(z_lo)  # one-sided
    p2 =      Phi(z_hi)   # one-sided
    return (p1 < alpha and p2 < alpha), p1, p2

def exact_binom_p(k: int, n: int, p0: float) -> float:
    """Two-sided exact binomial p-value (SciPy-style) in log-space to avoid overflow.
    Definition: sum of all probabilities with pmf(x) <= pmf(k).
    """
    import math
    import numpy as _np

    if n <= 0:
        return 1.0
    if p0 <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p0 >= 1.0:
        return 1.0 if k == n else 0.0

    # log pmf using lgamma (avoids comb overflow):
    # log C(n,x) + x log p0 + (n-x) log(1-p0)
    lgamma = math.lgamma
    log_p0 = math.log(p0)
    log_q0 = math.log(1.0 - p0)
    logC_n = lgamma(n + 1)

    def logpmf(x: int) -> float:
        return (logC_n - lgamma(x + 1) - lgamma(n - x + 1)
                + x * log_p0 + (n - x) * log_q0)

    log_pk = logpmf(k)

    # Collect all x with logpmf(x) <= log_pk (within tiny tolerance)
    logs = []
    tol = 1e-12
    # We iterate once over 0..n; cost ~O(n) with lgamma ops but still fine for a few 1e4.
    for x in range(0, n + 1):
        lx = logpmf(x)
        if lx <= log_pk + tol:
            logs.append(lx)

    if not logs:
        return 0.0

    # stable log-sum-exp
    mlog = max(logs)
    ssum = sum(math.exp(lx - mlog) for lx in logs)
    return float(math.exp(mlog) * ssum)
def _spectral_dimension_from_diagK(L: np.ndarray, times: np.ndarray, idx: int = 0, scale: float = 1.0):
    """Estimate spectral dimension d_s from the return probability K_t(ii).
    Uses eigen-decomposition of symmetric L once, then fits log K(ii,t) ~ a + b log t -> d_s = -2 b.
    Returns: (ds, R2, times, diagK_series)
    """
    L = np.array(L, dtype=float, copy=False)
    # Eigendecomposition once (L is symmetric)
    w, V = np.linalg.eigh(L)
    v2 = (V[idx, :]**2).astype(float)  # squared components for diagonal element
    times = np.asarray(times, dtype=float)
    diagK = []
    for t in times:
        Et = np.exp(-float(t) * float(scale) * w)
        diagK.append(float(np.dot(v2, Et)))
    diagK = np.array(diagK, dtype=float)

    # Prepare regression on valid (positive) values
    mask = (diagK > 0) & np.isfinite(diagK) & (times > 0)
    if mask.sum() < 3:
        return float('nan'), 0.0, times, diagK
    x = np.log(times[mask])
    y = np.log(diagK[mask])
    # Linear fit y = a + b x
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    b, a = sol[0], sol[1]
    yhat = A @ sol
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2)) if y.size > 1 else 0.0
    R2 = 1.0 - (ss_res/ss_tot) if ss_tot > 0 else 0.0
    ds = -2.0 * b
    return float(ds), float(R2), times, diagK

def run_refutation_suite(out_dir, ti_dir, names, p_ti, traj_clicks_dict, total_traj,
                         L, time_scale, idx_diag=0, alpha=None, eq_margin_abs=None):
    """
    Robuste TI-Validierung:
    - exakte chi^2-Tails (falls SciPy vorhanden; sonst stabiler Fallback),
    - G-Test (LRT, direkt an 2N*KL gekoppelt),
    - Wilson-CIs & TOST-Äquivalenz je Pixel,
    - Effektstärken (Cramér's V, JSD, TV),
    - rückwärtskompatible JSON-Felder + zusätzliche Kennzahlen.
    """
    K = len(names)
    counts = [int(traj_clicks_dict.get(names[i], 0)) for i in range(K)]
    N = int(sum(counts))

    # Normalisierte TI-Probs
    p_ti = np.asarray(p_ti, float)
    p_ti = np.clip(p_ti, 1e-15, 1.0)
    p_ti = p_ti / p_ti.sum()

    # Pearson-chi^2 (exaktes tail)
    exp = [max(N * pk, 1e-12) for pk in p_ti]
    chi2_stat = 0.0
    for o, e in zip(counts, exp):
        chi2_stat += (o - e)**2 / e
    df = max(K - 1, 1)
    p_chi2 = chi2_p_exact(chi2_stat, df)

    # LRT (G-Test)
    g_stat, p_g = g_test(counts, p_ti)

    # Empirie + Maße
    p_emp = [c / N if N > 0 else 0.0 for c in counts]
    Dkl = float(sum(pe * math.log(pe / max(pt,1e-300)) for pe, pt in zip(p_emp, p_ti) if pe > 0))
    jsd = js_divergence(p_emp, p_ti)
    tv  = total_variation(p_emp, p_ti)
    V   = cramers_v(chi2_stat, N, K)

    # Wilson & TOST je Pixel
    alpha_use = 0.05 if (alpha is None) else float(alpha)
    tost_flags, wilson_flags, wilson_cis, margins_used = [], [], [], []
    for ck, pk in zip(counts, p_ti):
        lo, hi = wilson_ci(ck, N, alpha=alpha_use)
        wilson_cis.append((lo, hi))
        wilson_flags.append(lo <= pk <= hi)
        if (eq_margin_abs is None) or (float(eq_margin_abs) <= 0.0):
            se = math.sqrt(pk*(1-pk)/max(N,1))
            eps = 1.959963984540054 * se  # z_0.975
        else:
            eps = float(eq_margin_abs)
        margins_used.append(eps)
        ok, _, _ = tost_equivalence_binomial(ck, N, pk, eps, alpha=alpha_use)
        tost_flags.append(ok)

    # K=2: exakter Binomialtest
    p_binom = None
    if K == 2:
        p_binom = exact_binom_p(counts[0], N, p_ti[0])

    # Bestehendes (kompatibles) Kriterium beibehalten
    pass_basic = (Dkl < 0.02) and (p_chi2 >= 0.05)

    # Strenge Äquivalenz-Aussage
    ti_equiv_pass = ((p_chi2 >= alpha_use) or (p_g >= alpha_use)) and all(tost_flags) and all(wilson_flags)

    # Spektraldimension (bestehende Methode)
    times = np.geomspace(1e-2, 1.0, 12)
    ds, R2, T, diagK = _spectral_dimension_from_diagK(L, times, idx=idx_diag, scale=time_scale)

    # JSON: rückwärtskompatibel + neue Felder
    res = {
        "traj_vs_ti": {
            "Dkl": Dkl,
            "chi2": chi2_stat,
            "df": df,
            "p_value_approx": float(p_chi2),             # rückwärtskompatibler Key
            "p_chi2_exact": float(p_chi2),               # präziser Name
            "p_g": float(p_g),
            "clicks_total": int(total_traj),
            "bins": [{
                "name": nm,
                "rel": (counts[i] / max(N,1)),
                "p_ti": float(p_ti[i]),
                "wilson_low": wilson_cis[i][0],
                "wilson_high": wilson_cis[i][1],
                "count": int(counts[i])
            } for i, nm in enumerate(names)],
            "passed": bool(pass_basic),
            "criteria": {"Dkl<": 0.02, "p_value>": 0.05},
            # neu:
            "tost_all_pixels": bool(all(tost_flags)),
            "wilson_all_pixels": bool(all(wilson_flags)),
            "alpha": alpha_use,
            "eq_margin_mode": "auto" if (eq_margin_abs is None or float(eq_margin_abs) <= 0.0) else "absolute",
            "eq_margin_abs_used_per_pixel": margins_used,
            "cramers_V": V,
            "JSD": jsd,
            "TV": tv,
            "p_binom_two_sided": (None if p_binom is None else float(p_binom)),
            "TI_equivalence_pass": bool(ti_equiv_pass)
        },
        "spectral_dimension": {
            "ds_est": float(ds), "R2": float(R2),
            "t_min": float(T.min()), "t_max": float(T.max())
        }
    }
    with open(os.path.join(ti_dir, "refutation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # PDF-Report
    pdf_path = os.path.join(ti_dir, "Refutation_Report.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
        s = (f"Refutation Suite — Summary\n\n"
             f"KL(rel || TI) = {Dkl:.4g}\n"
             f"Chi² = {chi2_stat:.3f} (df={df}),  p_chi2 ≈ {p_chi2:.2e},  p_G ≈ {p_g:.2e}\n"
             f"Spectral dimension d_s ≈ {ds:.3f} (R²={R2:.3f})\n"
             f"Total trajectories: {total_traj}\n"
             f"Pass (thresholds): KL<0.02 & p>0.05 → {pass_basic}\n"
             f"TI equivalence (strict): {ti_equiv_pass}")
        plt.text(0.07, 0.92, s, fontsize=11, va='top'); pdf.savefig(); plt.close()

    return pdf_path


# ===== ANALYZE =====
def try_small_spectrum(L: np.ndarray, mode: str, k: int):
    mode = str(mode).lower()
    if mode == "skip":
        return None
    if mode == "small":
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            S = csr_matrix(L)
            k_eff = int(clamp(k, 2, max(2, min(L.shape[0]-2, 256))))
            vals = eigsh(S, k=k_eff, which="SM", return_eigenvectors=False)
            return np.sort(np.real(vals))
        except Exception:
            pass
        vals = np.linalg.eigvalsh(L)
        return np.sort(vals)[:min(len(vals), k)]
    return np.sort(np.linalg.eigvalsh(L))

def analyze(artifacts_dir: str, out_dir: str, pixels_json: str, auto_pixels: int, auto_eff: float,
            mixer_mode: str, do_schro: bool, ham: str, gamma: float, sigma0: float,
            tmax: float, nt: int, do_sg: bool, sg_tmin: float, sg_tmax: float, sg_nt: int, d_w: float,
            do_branch: bool, do_spec: bool, spec_levels: List[int],
            do_open: bool, dephase_site: float, dephase_pixel: float, det_gamma: float, loss_gamma: float,
            do_traj: bool, n_traj: int, dt_traj: float, tmax_traj: float, seed: int, traj_scheme: str,
            viz_mode: str, viz_max_dim: int, viz_dpi: int,
            eigs_mode: str, eigs_k: int, traj_unobs_mode: str,
            # --- neue Save-Flags ---
            save_L: bool, save_kernels: bool, save_eigs_csv: bool, save_disp_csv: bool,
            save_open_rho: bool, open_series_stride: int, save_modes: bool, save_R_diag: bool, save_traj_events: bool,
            alpha: Optional[float] = None, eq_margin_abs: Optional[float] = None):

    progress_start("ANALYZE", 14)
    ensure_dir(out_dir); ti_dir = ensure_dir(os.path.join(out_dir,"TI"))

    # 1) load
    def load_npy(name): return np.load(os.path.join(artifacts_dir, name))
    A  = load_npy("A.npy"); L  = load_npy("L.npy")
    Kt = load_npy("K_t.npy"); Ks = load_npy("K_s.npy"); Kts=load_npy("K_ts.npy")
    Pt = load_npy("P_t.npy"); rho= load_npy("rho_beta.npy"); sigma=load_npy("sigma.npy")
    coords = load_npy("coords.npy")
    with open(os.path.join(artifacts_dir,"addresses.json"),"r",encoding="utf-8") as f: addrs=json.load(f)["addresses"]
    with open(os.path.join(artifacts_dir,"summary.json"),"r",encoding="utf-8") as f: summary=json.load(f)
    N=L.shape[0]
    progress_step("ANALYZE",1,14,"loaded artifacts")

    # 2) checks
    psd_L, Lmin, Lmax = is_psd(L)
    semi_err = float(np.linalg.norm(Kt @ Ks - Kts, ord=np.inf))
    pos_ok = bool((Kt >= -1e-12).all())
    rowsum_err = float(np.max(np.abs(Pt.sum(axis=1) - 1.0)))
    nonneg_Pt = bool((Pt >= -1e-12).all())
    rho_tr = float(np.trace(rho).real); psd_rho, rmin, rmax = is_psd(rho)
    psd_sig, _, _ = is_psd(sigma); sig_tr = float(np.trace(sigma).real)
    eigs = try_small_spectrum(L, eigs_mode, eigs_k)
    progress_step("ANALYZE",2,14,"core checks done")

    # 3) pixels + mixer
    if pixels_json:
        pixels,effs,names,U_json = load_pixels_from_json(pixels_json,N,coords,addrs)
    else:
        K = int(clamp(auto_pixels,1,MAX_PIXELS))
        blocks=np.array_split(np.arange(N), K)
        pixels,names=[],[]
        for b in blocks:
            v=np.zeros(N); v[b]=1.0; pixels.append(v); names.append(f"pix{len(names)}")
        effs=np.full(len(pixels), float(auto_eff)); U_json=None
    K=len(pixels)
    if K>MAX_PIXELS: raise ValueError("K>MAX_PIXELS")
    povm = povm_from_pixels(pixels, effs)
    p_povm=np.array([float(np.trace(rho@Ek).real) for Ek in povm])
    C = pixel_mode_projection(pixels)
    mode_vecs = C.copy()

    U = None
    if (U_json is None) and (mixer_mode.lower() in ["bs","unitary"]) and K==2:
        U = beamsplitter_50_50()
    elif U_json is not None:
        U = U_json
    if U is not None:
        if not (isinstance(U,np.ndarray) and U.ndim==2 and U.shape==(K,K)):
            raise ValueError(f"invalid unitary shape {getattr(U,'shape',None)}; expected {(K,K)}.")
        if not np.allclose(U.conj().T@U, np.eye(K), atol=1e-8):
            raise ValueError("unitary not unitary: U^†U != I")
        SigmaK = (mode_vecs) @ rho @ (mode_vecs.conj().T)
        SigmaK_mixed = U @ SigmaK @ U.conj().T
        p_unitary = effs * np.real(np.diag(SigmaK_mixed))
        mode_vecs = (U @ mode_vecs)
    else:
        SigmaK = (mode_vecs) @ rho @ (mode_vecs.conj().T)
        SigmaK_mixed = SigmaK.copy()
        p_unitary = p_povm.copy()
    if save_modes:
        np.save(os.path.join(ti_dir,"mode_vecs.npy"), mode_vecs)
        np.save(os.path.join(ti_dir,"SigmaK.npy"), SigmaK)
        np.save(os.path.join(ti_dir,"SigmaK_mixed.npy"), SigmaK_mixed)
        if U is not None: np.save(os.path.join(ti_dir,"unitary_U.npy"), U)
    progress_step("ANALYZE",3,14,"POVM and mixer applied")

    # 4) visuals (K_t already; add L, Ks, Kts, Pt)
    safe_imshow_log(Kt, label="K_t (log)",
                    fname=os.path.join(out_dir,"heatmap_Kt_log.png"),
                    viz_mode=viz_mode, viz_max_dim=int(viz_max_dim), viz_dpi=int(viz_dpi))
    if save_kernels:
        safe_imshow_log(Ks, label="K_s (log)",
                        fname=os.path.join(out_dir,"heatmap_Ks_log.png"),
                        viz_mode=viz_mode, viz_max_dim=int(viz_max_dim), viz_dpi=int(viz_dpi))
        safe_imshow_log(Kts, label="K_{t+s} (log)",
                        fname=os.path.join(out_dir,"heatmap_Kts_log.png"),
                        viz_mode=viz_mode, viz_max_dim=int(viz_max_dim), viz_dpi=int(viz_dpi))
        # Pt linear (keine LogNorm, enthält evtl. Nullen)
        if viz_mode != "off":
            P_plot = _downsample2d(np.array(Pt, float, copy=False),
                                   max_dim=int(viz_max_dim), mode=str(viz_mode))
        else:
            P_plot = np.array(Pt, float, copy=False)
        plt.figure(figsize=(6,5))
        im=plt.imshow(P_plot, cmap="viridis", interpolation="nearest")
        _add_cbar(im, label="P_t entries")
        plt.title("P_t (degree-normalized)"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"heatmap_Pt.png"), dpi=int(viz_dpi)); plt.close()
    if save_L:
        # signierte Heatmap um 0
        if viz_mode != "off":
            L_plot = _downsample2d(np.array(L, float, copy=False),
                                   max_dim=int(viz_max_dim), mode=str(viz_mode))
        else:
            L_plot = np.array(L, float, copy=False)
        v = float(np.max(np.abs(L_plot))) if L_plot.size else 1.0
        if v<=0: v=1.0
        plt.figure(figsize=(6,5))
        im = plt.imshow(L_plot, cmap="coolwarm",
                        norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
                        interpolation="nearest")
        _add_cbar(im, label="L entries (diag=deg, offdiag=-1 on edges)")
        plt.title("Graph Laplacian L (signed)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,"heatmap_L_signed.png"), dpi=int(viz_dpi)); plt.close()
        # Sparsity
        plt.figure(figsize=(6,5)); plt.spy(np.abs(L)>1e-12, markersize=1)
        plt.title("L sparsity pattern (non-zeros)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,"heatmap_L_sparsity.png"), dpi=int(viz_dpi)); plt.close()
    # Eigenspektrum
    if eigs is not None and eigs.size>0:
        plt.figure(figsize=(6,3.5)); plt.plot(np.arange(len(eigs)),eigs,'.',ms=3)
        plt.title(f"Eigenvalues of L ({eigs_mode})"); plt.xlabel("index"); plt.ylabel("eigenvalue")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,"eigs_L.png"), dpi=160); plt.close()
        if save_eigs_csv:
            pd.DataFrame({"eig": np.array(eigs, float)}).to_csv(os.path.join(out_dir,"eigs_L.csv"), index=False)
    progress_step("ANALYZE",4,14,"core visuals saved")

    # 5) Schrödinger dispersion (optional)
    ts=None; r2=None
    if do_schro:
        H = ((-float(gamma)) * (L if ham.lower().startswith("lap") else A)).astype(np.complex128)
        c0=coords.mean(axis=0); psi0=gaussian_on_coords(coords,c0,float(sigma0))
        assert_range("--tmax", tmax, 0.0, 1e6); nt=int(clamp(nt,2,MAX_STEPS))
        ts=np.linspace(0.0, float(tmax), int(nt)); r2=[]
        for tt in ts:
            psi_t=schrodinger_propagate(H,psi0,tt)
            prob=(psi_t.conj()*psi_t).real
            r2.append(packet_width_from_prob(prob,coords,c0))
        r2=np.array(r2)
        plt.figure(figsize=(6,3.5)); plt.plot(ts,r2,marker='.')
        plt.title("Wavepacket width ⟨|x−x₀|²⟩ vs time (unitary)")
        plt.xlabel("time"); plt.ylabel("<r^2>")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,"dispersion_width.png"), dpi=160); plt.close()
        if save_disp_csv:
            pd.DataFrame({"t":ts, "r2":r2}).to_csv(os.path.join(out_dir,"dispersion_width.csv"), index=False)
    progress_step("ANALYZE",5,14,"schro layer done")

    # 6) Open-system & (optional) trajectories
    progress_step("ANALYZE",6,14,"open-system init")
    if do_open or do_traj:
        H_open = ((-float(gamma)) * (L if ham.lower().startswith("lap") else A)).astype(np.complex128)

        # L_ops FULL: Site-Dephasing (N Stück), Pixel-Dephasing (K Stück), Detektor (K Stück)
        L_ops_full : List[Tuple[float,np.ndarray]] = []
        if dephase_site>0:
            for i in range(N):
                e = np.zeros(N); e[i]=1.0
                L_i = np.diag(e).astype(np.complex128)  # |i><i|
                L_ops_full.append( (float(dephase_site), L_i) )
        # Pixel-/Detektor-Projektionen
        Jpix = []
        if (dephase_pixel>0 and K>0) or (det_gamma>0 and K>0):
            for k in range(K):
                chi = mode_vecs[k,:].conj().reshape(N,1)
                J = (chi @ chi.conj().T).astype(np.complex128)
                if dephase_pixel>0: L_ops_full.append( (float(dephase_pixel), J) )
                if det_gamma>0:    Jpix.append( (f"pixel:{names[k]}", float(det_gamma)*float(effs[k]), J) )

        # Startzustand für Open-System
        c0=coords.mean(axis=0); psi_init = gaussian_on_coords(coords, c0, float(sigma0))
        rho0 = (np.outer(psi_init, psi_init.conj())).astype(np.complex128)

        # Master-Integration (voll)
        dtm   = float(clamp(dt_traj, 1e-5, 1.0))  # wir nutzen dt_traj auch für Master
        steps = int(min(MAX_STEPS, max(1, math.ceil(tmax/dtm))))
        rec_stride = max(1, steps//50) if open_series_stride<=0 else int(open_series_stride)
        rho_t, rho_series = integrate_master_rk4_full(
            rho0, H_open, L_ops_full, dt=dtm, steps=steps, store_stride=rec_stride, renorm_trace=True
        )
        times = np.linspace(0.0, float(tmax), len(rho_series))
        r2_open=[]
        for R in rho_series:
            prob = np.maximum(np.real(np.diag(R)), 0.0)
            r2_open.append(packet_width_from_prob(prob, coords, c0))
        pd.DataFrame({"t":times, "width":r2_open}).to_csv(os.path.join(ti_dir,"open_width.csv"), index=False)
        plt.figure(figsize=(6,3.5))
        if ts is not None and r2 is not None: plt.plot(ts, r2, 'o-', ms=3, label="unitary")
        plt.plot(times, r2_open, '.-', ms=3, label="open (Lindblad FULL)")
        plt.title("Packet width vs time: unitary vs open"); plt.xlabel("time"); plt.ylabel("<r^2>")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(ti_dir,"open_dispersion.png"), dpi=180); plt.close()
        if save_open_rho:
            np.save(os.path.join(ti_dir,"rho_open_final.npy"), rho_t)
            # dünne Serie
            np.savez_compressed(os.path.join(ti_dir,"rho_open_series.npz"),
                                times=times, series=np.array(rho_series, dtype=np.complex128))

        if do_traj:
            rng = np.random.default_rng(int(seed))
            clicks = {nm:0 for nm in names}; lost=0; none=0; times_click=[]
            # beobachtete = Detektoren; unbeobachtete = Dephasing-Kanäle
            observed = [(lbl, g, J) for (lbl,g,J) in Jpix]
            unobserved = []
            if dephase_site>0:
                # Für Effizienz approximieren wir unbeobachtete site-dephasing via Heff (kein expliziter Jump)
                # (Liste bleibt leer; Heff wird in mcwf_trajectory_full aus den L_ops erzeugt)
                for i in range(N):
                    e = np.zeros(N); e[i]=1.0; L_i = np.diag(e).astype(np.complex128)
                    unobserved.append( (float(dephase_site), L_i) )
            if dephase_pixel>0:
                for k in range(K):
                    chi = mode_vecs[k,:].conj().reshape(N,1)
                    J = (chi @ chi.conj().T).astype(np.complex128)
                    unobserved.append( (float(dephase_pixel), J) )

            n_traj = int(clamp(n_traj, 1, MAX_TRAJ))
            report_every = max(1, n_traj // 20)
            events_path = os.path.join(ti_dir,"traj_events.csv") if save_traj_events else None
            if events_path:
                with open(events_path,"w",encoding="utf-8") as f: f.write("traj_id,t_click,channel\n")
            info(f"[ANALYZE] trajectories 0/{n_traj} …")
            for it in range(n_traj):
                res = mcwf_trajectory_full(psi_init, H_open, observed, unobserved, float(dt_traj), float(tmax_traj), rng)
                if res["clicked"]:
                    nm = res["channel"]; clicks[nm] = clicks.get(nm,0) + 1; times_click.append(res["time"])
                    if events_path:
                        with open(events_path,"a",encoding="utf-8") as f: f.write(f"{it},{res['time']},{nm}\n")
                else:
                    if res["channel"]=="lost": lost += 1
                    else: none += 1
                if (it+1) % report_every == 0:
                    info(f"[ANALYZE] trajectories {it+1}/{n_traj} …")

            total = sum(clicks.values())
            df = pd.DataFrame({"pixel":names,
                               "traj_rel":[clicks[nm]/max(total,1) for nm in names],
                               "p_ti":list(p_unitary)})
            df.to_csv(os.path.join(ti_dir,"traj_vs_povm.csv"), index=False)
            plt.figure(figsize=(7,3.6))
            idx=np.arange(len(names)); w=0.4
            plt.bar(idx-w/2, [clicks[nm]/max(total,1) for nm in names], width=w, label="trajectories (rel.)")
            plt.bar(idx+w/2, p_unitary, width=w, label="TI tr(ρE')")
            plt.xticks(idx, names, rotation=45, ha='right'); plt.legend()
            plt.title("Trajectories vs TI prediction"); plt.tight_layout(); plt.savefig(os.path.join(ti_dir,"traj_hist_vs_TI.png"), dpi=200); plt.close()
            if len(times_click)>0:
                plt.figure(figsize=(6,3.4)); plt.hist(times_click, bins=30)
                plt.title("Detection time distribution"); plt.xlabel("t"); plt.ylabel("counts")
                plt.tight_layout(); plt.savefig(os.path.join(ti_dir,"traj_times.png"), dpi=180); plt.close()
            with open(os.path.join(ti_dir,"traj_hist.json"),"w",encoding="utf-8") as f:
                json.dump({"clicks":clicks, "total":int(total), "lost":int(lost), "none":int(none), "names":names}, f, indent=2)
    progress_step("ANALYZE",7,14,"open-system layer done")

    # 7) TI visuals
    abs_rho = np.abs(rho); abs_rho = abs_rho / (abs_rho.max() + 1e-16)
    safe_imshow_log(np.maximum(abs_rho,1e-16), label="|ρ| (log)",
                    fname=os.path.join(ti_dir,"rho_abs_log.png"),
                    viz_mode=viz_mode, viz_max_dim=int(viz_max_dim), viz_dpi=int(viz_dpi))
    plt.figure(figsize=(6,3.5)); idx=np.arange(len(p_povm)); w=0.4
    plt.bar(idx-w/2, p_povm, width=w, label="⟨E_k⟩ (POVM)")
    plt.bar(idx+w/2, p_unitary, width=w, label="⟨E'_k⟩ (after U)")
    plt.xticks(idx, [f"{i}:{n}" for i,n in enumerate(names)], rotation=45, ha='right'); plt.legend()
    plt.title("Detector probabilities (TI)"); plt.tight_layout(); plt.savefig(os.path.join(ti_dir,"povm_vs_unitary.png"), dpi=200); plt.close()
    progress_step("ANALYZE",8,14,"TI visuals saved")

    # 8) Sub-Gaussian (optional)
    if do_sg:
        sg_dir = ensure_dir(os.path.join(out_dir,"SG"))
        r_used = float(summary.get("r_used",1.0)); level = int(summary.get("level",0))
        time_scale = r_used**(-level)
        R = resistance_matrix_from_L(L)
        if save_R_diag: np.save(os.path.join(sg_dir,"R_diag.npy"), np.diag(R))
        t_list = list(np.geomspace(float(sg_tmin), float(sg_tmax), int(clamp(sg_nt,2,1000))))
        Kt_list = [expm_sym(L, -time_scale*t) for t in t_list]
        subgaussian_check(Kt_list, t_list, R, float(d_w), sg_dir, ref_idx=0)
        progress_step("ANALYZE",9,14,"Sub-Gaussian fits saved")
    else:
        progress_step("ANALYZE",9,14,"Sub-Gaussian skipped")

    # 9) Refutation Suite
    r_used = float(summary.get("r_used",1.0)); level = int(summary.get("level",0))
    time_scale = r_used**(-level)
    traj_clicks = {}; total = 0
    p_hist = os.path.join(ti_dir,"traj_hist.json")
    if os.path.exists(p_hist):
        with open(p_hist,"r",encoding="utf-8") as f: dat=json.load(f)
        total = int(dat.get("total",0)); clicksD = dat.get("clicks", {})
        # map name->count
        traj_clicks = clicksD
    ref_pdf = run_refutation_suite(out_dir, ti_dir, names, p_unitary, traj_clicks, total, L, time_scale,
                               idx_diag=0, alpha=alpha, eq_margin_abs=eq_margin_abs)

    progress_step("ANALYZE",10,14,"Refutation report saved")

    # 10) metrics & CSV
    df=pd.DataFrame([{
        "level":summary.get("level"), "r_estimate":summary.get("r_estimate"),
        "r_used":summary.get("r_used"), "n_effective":summary.get("n_effective"),
        "L_psd_ok":psd_L, "L_min_eig":Lmin, "L_max_eig":Lmax,
        "semigroup_err_inf":semi_err, "K_t_positivity_ok":pos_ok,
        "P_t_rowsum_inf_err":rowsum_err, "P_t_nonneg_ok":nonneg_Pt,
        "rho_trace":rho_tr, "rho_psd_ok":psd_rho, "rho_min_eig":rmin,
        "sigma_psd_ok":psd_sig, "sigma_trace":sig_tr
    }])
    df.to_csv(os.path.join(out_dir,"metrics.csv"), index=False)
    pd.DataFrame({"pixel":names,"efficiency":effs,"p":p_povm,"p_after_U":p_unitary}).to_csv(
        os.path.join(out_dir,"pixel_probs.csv"), index=False)
    progress_step("ANALYZE",11,14,"metrics & CSV exported")

    # 11) PDFs
    pdf_ti = os.path.join(ti_dir, "TI_comparison.pdf")
    with PdfPages(pdf_ti) as pdf:
        plt.figure(figsize=(8.27,11.69)); plt.axis('off')
        txt = ("TI: ρ-Kern & q-Erwartungen\nOpen-system: FULL GKSL (site/pixel dephasing), observed pixel channels → clicks.\n"
               "MCWF trajectories vs TI tr(ρE'), Beamsplitter U auf Pixelmoden.\nRefutation: KL/χ²/Wilson; d_s fit.")
        plt.text(0.07,0.93, txt, fontsize=12, va='top'); pdf.savefig(); plt.close()
        for fn in ["rho_abs_log.png","povm_vs_unitary.png","open_dispersion.png",
                   "traj_hist_vs_TI.png","traj_times.png"]:
            p=os.path.join(ti_dir,fn)
            if os.path.exists(p):
                fig=plt.figure(figsize=(8.27,11.69)); ax=fig.add_subplot(111); ax.axis('off')
                arr=plt.imread(p); ax.imshow(arr); pdf.savefig(); plt.close()

    pdf_main = os.path.join(out_dir, "TI_ST_FULL_v14_report.pdf")
    with PdfPages(pdf_main) as pdf:
        plt.figure(figsize=(8.27,11.69)); plt.axis('off')
        txt=(f"TI on ST — Report (FULL v14)\nArtifacts: {artifacts_dir}\n"
             f"Level: {summary.get('level')}, r_est={summary.get('r_estimate'):.6g}, "
             f"r_used={summary.get('r_used'):.6g}, n={summary.get('n_effective')}, bc={summary.get('bc')}\n"
             f"Viz: mode={viz_mode}, max_dim={viz_max_dim}, dpi={viz_dpi} (nur Darstellung). "
             f"Eigs: mode={eigs_mode}, k={eigs_k}. Traj(unobs)={traj_unobs_mode}")
        plt.text(0.07,0.92, txt, fontsize=11, va='top'); pdf.savefig(); plt.close()
    res={"pdf_main": pdf_main, "pdf_ti": pdf_ti, "ti_dir": ti_dir}
    with open(os.path.join(out_dir,"result_paths.json"),"w",encoding="utf-8") as f: json.dump(res,f,indent=2)
    progress_step("ANALYZE",12,14,"result paths written")

    # 12) (optional) weitere Kernel-Visuals bereits erledigt; Abschluss
    progress_done("ANALYZE")
    return res

# ===== CLI =====
def parse_levels(s: str) -> List[int]:
    if not s: return []
    out=[]
    for tok in s.split(","):
        tok=tok.strip()
        if "-" in tok:
            a,b=tok.split("-",1); a=int(a); b=int(b)
            if b<a: a,b=b,a
            out.extend(list(range(a, b+1)))
        else:
            out.append(int(tok))
    out = [x for x in out if 0<=x<=MAX_LEVEL]
    out = out[:MAX_SPEC_LVLS]
    return sorted(set(out))

def main():
    ap=argparse.ArgumentParser(description="TI-on-ST FULL v14: Full Lindblad + trajectories + rich exports.")
    ap.add_argument("--stage", default="all", choices=["build","analyze","all"])
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--out", default="./report")

    # Build
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--t", type=float, default=0.6)
    ap.add_argument("--s", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--bc", default="dirichlet", choices=["dirichlet","neumann"])
    ap.add_argument("--use_fixed_r", action="store_true")

    # Analyze (core)
    ap.add_argument("--pixels_json", default="")
    ap.add_argument("--pixels", type=int, default=4)
    ap.add_argument("--eff", type=float, default=1.0)
    ap.add_argument("--mixer", default="unitary", choices=["unitary","bs","none"])

    ap.add_argument("--schro", default="yes")
    ap.add_argument("--ham", default="laplacian", choices=["laplacian","adjacency"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--nt", type=int, default=21)

    # Sub-Gaussian
    ap.add_argument("--sg", default="no")
    ap.add_argument("--sg_tmin", type=float, default=0.1)
    ap.add_argument("--sg_tmax", type=float, default=1.0)
    ap.add_argument("--sg_nt", type=int, default=6)
    ap.add_argument("--dw", type=float, default=2.6, help="walk dimension for sub-Gaussian fit")

    ap.add_argument("--branch", default="no")
    ap.add_argument("--spec", default="no")
    ap.add_argument("--spec_levels", default="1,2,3")

    # Open-system & trajectories
    ap.add_argument("--open", default="yes")
    ap.add_argument("--dephase_site", type=float, default=0.0)
    ap.add_argument("--dephase_pixel", type=float, default=0.0)
    ap.add_argument("--det_gamma", type=float, default=1.0)
    ap.add_argument("--loss", type=float, default=0.0)  # für FULL derzeit nicht separat implementiert (spurtreu)

    ap.add_argument("--traj", default="no")
    ap.add_argument("--ntraj", type=int, default=1000)
    ap.add_argument("--dt", dest="dt_traj", type=float, default=0.01)
    ap.add_argument("--tmax_traj", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--traj_scheme", default="jumps", choices=["jumps"])
    ap.add_argument("--traj_unobs", default="nojump", choices=["nojump","jumps"],
                    help="(nur Info) FULL nutzt Heff für unbeobachtete in Traj.")

    # FAST VIZ (nur Darstellung)
    ap.add_argument("--viz_mode", default="auto", choices=["auto","stride","block","full","off"],
                    help="Downsampling nur für Heatmap-Darstellung; Rohdaten bleiben unverändert.")
    ap.add_argument("--viz_max_dim", type=int, default=1200, help="Max Pixel-Kante für Heatmap-Darstellung.")
    ap.add_argument("--viz_dpi", type=int, default=160, help="DPI beim Speichern der Heatmap.")

    # Eigs
    ap.add_argument("--eigs_mode", default="small", choices=["small","full","skip"],
                    help="Spektrum von L: kleines Teil, ganz oder überspringen.")
    ap.add_argument("--eigs_k", type=int, default=128, help="Anzahl Eigenwerte bei eigs_mode=small.")

    # Save-Flags (neue)
    ap.add_argument("--save_L", default="yes")
    ap.add_argument("--save_kernels", default="yes")
    ap.add_argument("--save_eigs_csv", default="yes")
    ap.add_argument("--save_disp_csv", default="yes")
    ap.add_argument("--save_open_rho", default="no")
    ap.add_argument("--open_series_stride", type=int, default=0)
    ap.add_argument("--save_modes", default="yes")
    ap.add_argument("--save_R_diag", default="no")
    ap.add_argument("--save_traj_events", default="no")
    
    # [PATCH A] optionale Test-Parameter (wirken nur, wenn gesetzt)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Signifikanzniveau für Tests (Default 0.05; nur wirksam, wenn gesetzt).")
    ap.add_argument("--eq_margin_abs", type=float, default=None,
                    help="Absolute Äquivalenzmarge je Pixel (Default auto = z_0.975 * SE; nur wirksam, wenn gesetzt).")



    args=ap.parse_args()
    args.artifacts = ensure_dir(args.artifacts); args.out = ensure_dir(args.out)

    # Validate
    assert_range("--level", args.level, 0, MAX_LEVEL)
    for nm in ["t","s","beta","gamma","sigma0","tmax","sg_tmin","sg_tmax","dt_traj","tmax_traj","dephase_site","dephase_pixel","det_gamma","loss","dw"]:
        val = float(getattr(args, nm))
        if val < 0: raise ValueError(f"{nm} must be >=0")
    if args.nt < 2 or args.nt > MAX_STEPS: raise ValueError("--nt out of bounds")
    if args.sg_nt < 2 or args.sg_nt > 1000: raise ValueError("--sg_nt out of bounds")
    if args.pixels < 1 or args.pixels > MAX_PIXELS: raise ValueError("--pixels out of bounds")
    if args.ntraj < 1 or args.ntraj > MAX_TRAJ: raise ValueError("--ntraj out of bounds")
    if args.viz_max_dim < 100 or args.viz_max_dim > 10000: raise ValueError("--viz_max_dim out of bounds")
    if args.viz_dpi < 72 or args.viz_dpi > 600: raise ValueError("--viz_dpi out of bounds")

    # normalize yes/no
    yn = lambda s: (str(s).lower() in ["1","yes","true","y"])

    if args.stage in ["build","all"]:
        build_artifacts(args.artifacts, args.level, args.t, args.s, args.beta, args.bc, args.use_fixed_r)
    if args.stage in ["analyze","all"]:
        res=analyze(
            artifacts_dir=args.artifacts, out_dir=args.out,
            pixels_json=args.pixels_json, auto_pixels=args.pixels, auto_eff=args.eff,
            mixer_mode=args.mixer,
            do_schro=yn(args.schro),
            ham=args.ham, gamma=args.gamma, sigma0=args.sigma0, tmax=args.tmax, nt=args.nt,
            do_sg=yn(args.sg),
            sg_tmin=args.sg_tmin, sg_tmax=args.sg_tmax, sg_nt=args.sg_nt, d_w=args.dw,
            do_branch=yn(args.branch),
            do_spec=yn(args.spec),
            spec_levels=parse_levels(args.spec_levels),
            do_open=yn(args.open),
            dephase_site=args.dephase_site, dephase_pixel=args.dephase_pixel,
            det_gamma=args.det_gamma, loss_gamma=args.loss,
            do_traj=yn(args.traj), n_traj=args.ntraj, dt_traj=args.dt_traj, tmax_traj=args.tmax_traj,
            seed=args.seed, traj_scheme=args.traj_scheme,
            viz_mode=args.viz_mode, viz_max_dim=args.viz_max_dim, viz_dpi=args.viz_dpi,
            eigs_mode=args.eigs_mode, eigs_k=args.eigs_k, traj_unobs_mode=args.traj_unobs,
            save_L=yn(args.save_L), save_kernels=yn(args.save_kernels), save_eigs_csv=yn(args.save_eigs_csv),
            save_disp_csv=yn(args.save_disp_csv), save_open_rho=yn(args.save_open_rho),
            open_series_stride=args.open_series_stride, save_modes=yn(args.save_modes),
            save_R_diag=yn(args.save_R_diag), save_traj_events=yn(args.save_traj_events),
    alpha=args.alpha, eq_margin_abs=args.eq_margin_abs)
        print(json.dumps(res, indent=2))

if __name__=="__main__":
    main()

