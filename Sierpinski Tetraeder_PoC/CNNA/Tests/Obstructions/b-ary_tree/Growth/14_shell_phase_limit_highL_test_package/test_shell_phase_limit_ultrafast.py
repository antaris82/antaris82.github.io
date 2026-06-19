#!/usr/bin/env python3
"""
Ultra-fast shell phase limit test.

Polar-only high-level run for shell_norm_inverse_square.

Rationale:
- polar and skew_iso agreed exactly in the robustness test;
- G_weighted was very close;
- eigenphase is cleaner but costlier;
- for pushing L higher, polar-only is the appropriate first limit probe.
"""

from __future__ import annotations
import argparse, csv, importlib.util, math, sys, time
from pathlib import Path
import numpy as np

EPS=1e-12

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

def polar_angle_from_R2(R2):
    # Fast 2x2 polar rotation angle. For nonsingular 2x2 matrix A, polar rotation angle
    # can be obtained by atan2(c-b, a+d) for A=[[a,b],[c,d]].
    a,b=R2[0,0],R2[0,1]
    c,d=R2[1,0],R2[1,1]
    return math.degrees(math.atan2(float(c-b), float(a+d)))

def local_polar_theta(model,parent:int):
    M=model.local_matrix_for_parent(parent)
    if M is None: return float("nan")
    P=column_stochastic(M)
    A=0.5*(P-P.T)
    axis=skew_axis(A)
    if np.linalg.norm(axis)<=EPS: return float("nan")
    cst=np.ones(3)/math.sqrt(3.0)
    if np.dot(axis,cst)<0: axis=-axis
    axis=axis/(np.linalg.norm(axis)+EPS)
    B=plane_basis(axis)
    R2=B.T@P@B
    return polar_angle_from_R2(R2)

def fit_theta(L,y):
    best=None
    for r in np.linspace(0.2,0.98,157):
        f=r**L
        X=np.vstack([np.ones_like(f),f]).T
        beta,*_=np.linalg.lstsq(X,y,rcond=None)
        yhat=X@beta
        err=float(np.sqrt(np.mean((y-yhat)**2)))
        cand=dict(theta_inf=float(beta[0]),A=float(beta[1]),r=float(r),rmse=err,last_pred=float(yhat[-1]))
        if best is None or cand["rmse"]<best["rmse"]: best=cand
    return best

def fit_decay(L,y):
    mask=np.isfinite(y)&(y>0); L=L[mask]; y=y[mask]
    if len(y)<3: return {}
    X=np.vstack([np.ones_like(L),L]).T
    beta,*_=np.linalg.lstsq(X,np.log(y),rcond=None)
    yhat=np.exp(X@beta)
    return dict(C=float(math.exp(beta[0])),r=float(math.exp(beta[1])),
                rmse=float(np.sqrt(np.mean((y-yhat)**2))),last_pred=float(yhat[-1]))

def loop_residual_summary(model,data,theta_by_level):
    # full local closure residuals, polar only
    vals=[]
    def add(loop):
        raw=sum(data[u] for u in loop)
        centered=sum(data[u]-theta_by_level[model.nodes[u].level] for u in loop)
        vals.append(abs(wrap_deg(centered)))
    for p in list(data.keys()):
        cs=[c for c in model.nodes[p].children if c in data]
        if len(cs)==3:
            c1,c2,c3=cs
            add([c1,c2,c3])
            add([p,c1,c2,c3])
            add([p,c1,c2]); add([p,c2,c3]); add([p,c3,c1])
    return dict(count=len(vals),mean_abs_centered_level_deg=mean(vals),
                p95_abs_centered_level_deg=perc(vals,95),
                max_abs_centered_level_deg=maxv(vals))

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
    density=[]; residuals=[]
    lines=["SHELL PHASE LIMIT ULTRAFAST POLAR TEST",f"  max_level={max_level}",f"  tail_start={tail_start}",""]
    t0=time.time()
    for gl in range(1,max_level+1):
        old_frontier=frontier
        frontier=model.grow_one_level(frontier,gl)
        for p in old_frontier:
            th=local_polar_theta(model,p)
            if np.isfinite(th): data[p]=th
        by_level={}
        for p,th in data.items():
            by_level.setdefault(model.nodes[p].level,[]).append(th)
        theta_by_level={pl:mean(vs) for pl,vs in by_level.items()}
        all_vals=list(data.values())
        dr=dict(global_level=gl,nodes=len(model.nodes),triples=len(data),
                polar_mean_deg=mean(all_vals),polar_std_deg=std(all_vals),
                polar_min_deg=float(np.min(all_vals)) if all_vals else float("nan"),
                polar_max_deg=float(np.max(all_vals)) if all_vals else float("nan"))
        density.append(dr)
        rr=loop_residual_summary(model,data,theta_by_level)
        rr["global_level"]=gl
        residuals.append(rr)
        lines.append(f"  L={gl}: nodes={len(model.nodes)}, triples={len(data)}, "
                     f"polar_mean={dr['polar_mean_deg']:.9f}, polar_std={dr['polar_std_deg']:.9f}, "
                     f"centered_loop_mean={rr['mean_abs_centered_level_deg']:.9f}, loops={rr['count']}")
    tail=[r for r in density if r["global_level"]>=tail_start]
    L=np.array([r["global_level"] for r in tail],float)
    y=np.array([r["polar_mean_deg"] for r in tail],float)
    sy=np.array([r["polar_std_deg"] for r in tail],float)
    rt=[r for r in residuals if r["global_level"]>=tail_start]
    Lr=np.array([r["global_level"] for r in rt],float)
    ry=np.array([r["mean_abs_centered_level_deg"] for r in rt],float)
    ftheta=fit_theta(L,y); fstd=fit_decay(L,sy); fres=fit_decay(Lr,ry)
    fits=[
        dict(kind="theta_inf",theta_inf=ftheta["theta_inf"],r=ftheta["r"],rmse=ftheta["rmse"],last_observed=float(y[-1]),last_pred=ftheta["last_pred"]),
        dict(kind="phase_std_decay",theta_inf=float("nan"),r=fstd.get("r",float("nan")),rmse=fstd.get("rmse",float("nan")),last_observed=float(sy[-1]),last_pred=fstd.get("last_pred",float("nan"))),
        dict(kind="centered_residual_decay",theta_inf=float("nan"),r=fres.get("r",float("nan")),rmse=fres.get("rmse",float("nan")),last_observed=float(ry[-1]),last_pred=fres.get("last_pred",float("nan"))),
    ]
    write_csv(outdir/"shell_ultrafast_density.csv",density)
    write_csv(outdir/"shell_ultrafast_residuals.csv",residuals)
    write_csv(outdir/"shell_ultrafast_fits.csv",fits)
    lines+=["","FITS",
            f"  polar theta_inf={ftheta['theta_inf']:.9f}, last={y[-1]:.9f}, r={ftheta['r']:.6f}, rmse={ftheta['rmse']:.6e}",
            f"  polar phase_std_decay: last={sy[-1]:.9f}, r={fstd.get('r',float('nan')):.6f}, rmse={fstd.get('rmse',float('nan')):.6e}",
            f"  polar centered_residual_decay: last={ry[-1]:.9f}, r={fres.get('r',float('nan')):.6f}, rmse={fres.get('rmse',float('nan')):.6e}",
            f"",
            f"elapsed_sec={time.time()-t0:.3f}"]
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=11)
    ap.add_argument("--tail-start",type=int,default=6)
    ap.add_argument("--outdir",type=Path,default=Path("shell_phase_limit_ultrafast_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir,args.tail_start))
if __name__=="__main__": main()
