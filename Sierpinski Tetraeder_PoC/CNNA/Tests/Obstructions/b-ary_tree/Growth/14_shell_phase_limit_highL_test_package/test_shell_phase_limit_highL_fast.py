#!/usr/bin/env python3
"""
Fast high-level shell-normalized phase-limit test.

Incremental version:
- local operator data are added only when a parent has just completed its three children;
- summaries are compact;
- loop residuals are accumulated, not stored per loop.

This is intended to reach one level higher than the broad diagnostic scripts.
"""

from __future__ import annotations
import argparse, csv, importlib.util, math, sys, time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

EPS=1e-12
METHODS=["polar","eigen","G_weighted"]

def load_scaling_model():
    path=Path("/mnt/data/test_conductance_scaling_generalization.py")
    spec=importlib.util.spec_from_file_location("scaling",path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod.ScalingModel

def wrap_deg(x): return ((x+180.0)%360.0)-180.0
def mean(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")
def std(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.std(xs)) if xs else float("nan")
def perc(vals,q):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.percentile(xs,q)) if xs else float("nan")
def maxv(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.max(xs)) if xs else float("nan")

def column_stochastic(M):
    P=M.copy().astype(float)
    for j in range(3):
        s=P[:,j].sum()
        if s>EPS: P[:,j]/=s
        else: P[j,j]=1.0
    return P

def skew_axis(A): return np.array([A[2,1],A[0,2],A[1,0]],float)

def plane_basis(axis):
    a=axis/(np.linalg.norm(axis)+EPS)
    h=np.array([1.,0.,0.])
    if abs(float(np.dot(a,h)))>0.9: h=np.array([0.,1.,0.])
    u=h-np.dot(h,a)*a; u=u/(np.linalg.norm(u)+EPS)
    v=np.cross(a,u); v=v/(np.linalg.norm(v)+EPS)
    return np.vstack([u,v]).T

def polar_so2(A2):
    U,s,Vt=np.linalg.svd(A2)
    Q=U@Vt
    if np.linalg.det(Q)<0:
        U[:,-1]*=-1
        Q=U@Vt
    return Q,s

def angle_so2_deg(Q):
    return math.degrees(math.atan2(float((Q[1,0]-Q[0,1])/2.0), float((Q[0,0]+Q[1,1])/2.0)))

def local_data(model,parent:int)->Optional[dict]:
    M=model.local_matrix_for_parent(parent)
    if M is None: return None
    P=column_stochastic(M)
    A=0.5*(P-P.T)
    axis=skew_axis(A)
    if np.linalg.norm(axis)<=EPS: return None
    c=np.ones(3)/math.sqrt(3.0)
    if np.dot(axis,c)<0: axis=-axis
    axis=axis/(np.linalg.norm(axis)+EPS)
    B=plane_basis(axis)
    R2=B.T@P@B
    S2=0.5*(R2+R2.T)
    Q,sing=polar_so2(R2)
    theta_polar=angle_so2_deg(Q)
    ev=np.linalg.eigvals(P)
    ces=[z for z in ev if abs(z.imag)>1e-9]
    theta_eigen=float("nan")
    if ces:
        z=max(ces,key=lambda z:z.imag)
        theta_eigen=math.degrees(math.atan2(z.imag,z.real))
    theta_G=float("nan")
    for sign,cand in [(-1,-S2),(1,S2)]:
        eig=np.linalg.eigvalsh(cand)
        if np.all(eig>1e-10):
            try:
                L=np.linalg.cholesky(cand)
                C=L.T
                Rg=C@R2@np.linalg.inv(C)
                Qg,_=polar_so2(Rg)
                theta_G=angle_so2_deg(Qg)
                break
            except np.linalg.LinAlgError:
                pass
    return dict(parent=parent,parent_level=model.nodes[parent].level,
                theta_polar=theta_polar,theta_eigen=theta_eigen,theta_G=theta_G)

def method_theta(d,m):
    if m=="polar": return d["theta_polar"]
    if m=="eigen": return d["theta_eigen"]
    if m=="G_weighted": return d["theta_G"]
    raise ValueError(m)

def summarize_density(gl,data):
    row=dict(global_level=gl,parent_level=-1,count=len(data))
    ds=list(data.values())
    for m in METHODS:
        vals=[method_theta(d,m) for d in ds]
        row[f"{m}_mean_deg"]=mean(vals)
        row[f"{m}_std_deg"]=std(vals)
    return row

def loop_residual_summary(gl,model,data):
    bg_global={m:mean([method_theta(d,m) for d in data.values()]) for m in METHODS}
    by_level={}
    for d in data.values(): by_level.setdefault(d["parent_level"],[]).append(d)
    bg_level={(m,pl):mean([method_theta(d,m) for d in ds]) for pl,ds in by_level.items() for m in METHODS}
    accum={}
    def add(mode,loop):
        for m in METHODS:
            vals=[method_theta(data[u],m) for u in loop]
            if not all(np.isfinite(vals)): continue
            raw=wrap_deg(sum(vals))
            cl=wrap_deg(sum(vals)-sum(bg_level[(m,data[u]["parent_level"])] for u in loop))
            key=(mode,m)
            if key not in accum: accum[key]=dict(abs_raw=[],abs_cl=[],theta=[])
            accum[key]["abs_raw"].append(abs(raw))
            accum[key]["abs_cl"].append(abs(cl))
            accum[key]["theta"].append(sum(vals)/len(loop))
    for p in list(data.keys()):
        cs=[c for c in model.nodes[p].children if c in data]
        if len(cs)==3:
            c1,c2,c3=cs
            add("sibling_cycle",[c1,c2,c3])
            add("parent_child_ring",[p,c1,c2,c3])
            add("parent_fan_triangle",[p,c1,c2])
            add("parent_fan_triangle",[p,c2,c3])
            add("parent_fan_triangle",[p,c3,c1])
    rows=[]
    for (mode,m),a in sorted(accum.items()):
        rows.append(dict(global_level=gl,loop_mode=mode,method=m,count=len(a["abs_raw"]),
                         mean_abs_raw_wrapped_deg=mean(a["abs_raw"]),
                         mean_abs_centered_level_deg=mean(a["abs_cl"]),
                         p95_abs_centered_level_deg=perc(a["abs_cl"],95),
                         max_abs_centered_level_deg=maxv(a["abs_cl"]),
                         mean_theta_step_deg=mean(a["theta"])))
    return rows

def fit_theta(L,y):
    best=None
    for r in np.linspace(0.2,0.98,157):
        f=r**L; X=np.vstack([np.ones_like(f),f]).T
        beta,*_=np.linalg.lstsq(X,y,rcond=None)
        yhat=X@beta
        err=float(np.sqrt(np.mean((y-yhat)**2)))
        cand=dict(theta_inf=float(beta[0]),A=float(beta[1]),r=float(r),rmse=err,last_pred=float(yhat[-1]))
        if best is None or cand["rmse"]<best["rmse"]: best=cand
    return best

def fit_decay(L,y):
    mask=np.isfinite(y)&(y>0); L=L[mask]; y=y[mask]
    if len(y)<3: return {}
    logy=np.log(y); X=np.vstack([np.ones_like(L),L]).T
    beta,*_=np.linalg.lstsq(X,logy,rcond=None)
    yhat=np.exp(X@beta)
    return dict(C=float(math.exp(beta[0])),r=float(math.exp(beta[1])),
                rmse=float(np.sqrt(np.mean((y-yhat)**2))),last_pred=float(yhat[-1]))

def fit_results(density,resid,tail_start):
    rows=[]; dens=[r for r in density if r["global_level"]>=tail_start]
    L=np.array([r["global_level"] for r in dens],float)
    for m in METHODS:
        y=np.array([r[f"{m}_mean_deg"] for r in dens],float)
        th=fit_theta(L,y)
        rows.append(dict(kind="theta_inf",method=m,theta_inf=th["theta_inf"],r=th["r"],rmse=th["rmse"],last_observed=float(y[-1])))
        sy=np.array([r[f"{m}_std_deg"] for r in dens],float)
        dc=fit_decay(L,sy)
        rows.append(dict(kind="phase_std_decay",method=m,theta_inf=float("nan"),r=dc.get("r",float("nan")),rmse=dc.get("rmse",float("nan")),last_observed=float(sy[-1])))
    # Average residual summaries over loop modes per level/method.
    for m in METHODS:
        vals=[]
        for gl in sorted({r["global_level"] for r in resid if r["global_level"]>=tail_start}):
            xs=[r["mean_abs_centered_level_deg"] for r in resid if r["global_level"]==gl and r["method"]==m]
            vals.append((gl,mean(xs)))
        Lr=np.array([x[0] for x in vals],float); y=np.array([x[1] for x in vals],float)
        dc=fit_decay(Lr,y)
        rows.append(dict(kind="centered_residual_decay",method=m,theta_inf=float("nan"),r=dc.get("r",float("nan")),rmse=dc.get("rmse",float("nan")),last_observed=float(y[-1])))
    return rows

def write_csv(path,rows):
    if not rows: return
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def run(max_level,outdir,tail_start):
    outdir.mkdir(parents=True,exist_ok=True)
    ScalingModel=load_scaling_model()
    model=ScalingModel(kernel="shell_norm_inverse_square",max_level=max_level,mode="log")
    frontier=[model.root]
    data={}
    density=[]; resid=[]
    lines=["SHELL PHASE LIMIT HIGH-L FAST TEST",f"  max_level={max_level}",f"  tail_start={tail_start}",""]
    t0=time.time()
    for gl in range(1,max_level+1):
        old_frontier=frontier
        frontier=model.grow_one_level(frontier,gl)
        for p in old_frontier:
            d=local_data(model,p)
            if d is not None: data[p]=d
        dr=summarize_density(gl,data); density.append(dr)
        rr=loop_residual_summary(gl,model,data); resid.extend(rr)
        polar_centered=mean([r["mean_abs_centered_level_deg"] for r in rr if r["method"]=="polar"])
        lines.append(f"  L={gl}: nodes={len(model.nodes)}, triples={len(data)}, "
                     f"polar_mean={dr['polar_mean_deg']:.9f}, polar_std={dr['polar_std_deg']:.9f}, "
                     f"centered_loop_mean={polar_centered:.9f}")
    fits=fit_results(density,resid,tail_start)
    write_csv(outdir/"shell_highL_fast_phase_density.csv",density)
    write_csv(outdir/"shell_highL_fast_loop_residuals.csv",resid)
    write_csv(outdir/"shell_highL_fast_fits.csv",fits)
    lines+=["","FITS"]
    for f in fits:
        if f["kind"]=="theta_inf":
            lines.append(f"  {f['method']} theta_inf={f['theta_inf']:.9f}, last={f['last_observed']:.9f}, r={f['r']:.6f}, rmse={f['rmse']:.6e}")
    for f in fits:
        if f["kind"]!="theta_inf":
            lines.append(f"  {f['method']} {f['kind']}: last={f['last_observed']:.9f}, r={f['r']:.6f}, rmse={f['rmse']:.6e}")
    lines.append(f"\nelapsed_sec={time.time()-t0:.3f}")
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=11)
    ap.add_argument("--tail-start",type=int,default=6)
    ap.add_argument("--outdir",type=Path,default=Path("shell_phase_limit_highL_fast_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir,args.tail_start))
if __name__=="__main__": main()
