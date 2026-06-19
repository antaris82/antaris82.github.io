#!/usr/bin/env python3
"""
Phase-density scaling test for shell/remote-kernel CNNA/NGF growth.

Question
--------
How do local response phase density and residual loop curvature scale with level?

Measured quantities
-------------------
For several remote ancestor kernels and for each global growth level:

1. local phase density by parent level:
   theta_polar mean/std
   theta_eigen mean/std
   theta_G mean/std
   variance proxies

2. loop residual curvature:
   for sibling_cycle, parent_fan_triangle, parent_child_ring
   subtract method-specific background phase and measure centered loop phase.

3. kernel dependence:
   inverse_square
   critical_exp_1over3
   exp_0p40
   exp_0p25
   shell_norm_inverse_square

Interpretation
--------------
A large raw loop phase can simply be accumulated uniform phase density.
The centered residual after subtracting the local phase background is the
curvature/frustration diagnostic.
"""

from __future__ import annotations
import argparse, csv, importlib.util, math, sys
from pathlib import Path
from typing import Dict, List, Optional
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

def wrap_deg(x):
    return ((x+180.0)%360.0)-180.0

def circ_mean_deg(vals):
    xs=[math.radians(float(v)) for v in vals if np.isfinite(float(v))]
    if not xs: return float("nan")
    z=sum(complex(math.cos(x),math.sin(x)) for x in xs)/len(xs)
    return math.degrees(math.atan2(z.imag,z.real))

def circ_std_deg(vals):
    xs=[math.radians(float(v)) for v in vals if np.isfinite(float(v))]
    if not xs: return float("nan")
    z=sum(complex(math.cos(x),math.sin(x)) for x in xs)/len(xs)
    R=abs(z)
    if R<=0: return 180.0
    return math.degrees(math.sqrt(max(0.0,-2.0*math.log(R))))

def mean(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")
def std(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.std(xs)) if xs else float("nan")
def maxv(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.max(xs)) if xs else float("nan")
def perc(vals,q):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.percentile(xs,q)) if xs else float("nan")

def column_stochastic(M):
    P=M.copy().astype(float)
    for j in range(3):
        s=P[:,j].sum()
        if s>EPS: P[:,j]/=s
        else: P[j,j]=1.0
    return P

def skew_axis(A):
    return np.array([A[2,1],A[0,2],A[1,0]],float)

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
    refl=0
    if np.linalg.det(Q)<0:
        refl=1
        U[:,-1]*=-1
        Q=U@Vt
    return Q,s,refl

def angle_so2_deg(Q):
    cosv=float((Q[0,0]+Q[1,1])/2.0)
    sinv=float((Q[1,0]-Q[0,1])/2.0)
    return math.degrees(math.atan2(sinv,cosv))

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

    Q,sing,refl=polar_so2(R2)
    theta_polar=angle_so2_deg(Q)

    # Eigenphase of the complex pair of P.
    ev=np.linalg.eigvals(P)
    ces=[z for z in ev if abs(z.imag)>1e-9]
    theta_eigen=float("nan")
    if ces:
        z=max(ces,key=lambda z:z.imag)
        theta_eigen=math.degrees(math.atan2(z.imag,z.real))

    theta_G=float("nan")
    G_cond=float("nan")
    G_sign=0
    for sign,cand in [(-1,-S2),(1,S2)]:
        eig=np.linalg.eigvalsh(cand)
        if np.all(eig>1e-10):
            try:
                L=np.linalg.cholesky(cand)
                C=L.T
                Rg=C@R2@np.linalg.inv(C)
                Qg,_,_=polar_so2(Rg)
                theta_G=angle_so2_deg(Qg)
                G_cond=float(max(eig)/(min(eig)+EPS))
                G_sign=sign
                break
            except np.linalg.LinAlgError:
                pass

    return dict(parent=parent,parent_level=model.nodes[parent].level,
                theta_polar=theta_polar,theta_eigen=theta_eigen,theta_G=theta_G,
                polar_reflection=refl,polar_cond=float(max(sing)/(min(sing)+EPS)),
                G_cond=G_cond,G_sign=G_sign,
                axis_x=float(axis[0]),axis_y=float(axis[1]),axis_z=float(axis[2]))

def build_loops(model,data):
    loops=[]
    for p in data:
        cs=[c for c in model.nodes[p].children if c in data]
        if len(cs)==3:
            c1,c2,c3=cs
            loops.append(dict(mode="sibling_cycle",level=model.nodes[p].level+1,base=p,loop=[c1,c2,c3]))
            loops.append(dict(mode="parent_child_ring",level=model.nodes[p].level,base=p,loop=[p,c1,c2,c3]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c1,c2]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c2,c3]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c3,c1]))
    return loops

def method_theta(d,method):
    if method=="polar": return d["theta_polar"]
    if method=="eigen": return d["theta_eigen"]
    if method=="G_weighted": return d["theta_G"]
    raise ValueError(method)

def phase_density_rows(kernel,global_level,data):
    rows=[]
    by_pl={}
    for p,d in data.items():
        by_pl.setdefault(d["parent_level"],[]).append(d)
    for pl,ds in sorted(by_pl.items()):
        row=dict(kernel=kernel,global_level=global_level,parent_level=pl,count=len(ds))
        for method in METHODS:
            vals=[method_theta(d,method) for d in ds]
            row[f"{method}_mean_linear_deg"]=mean(vals)
            row[f"{method}_std_linear_deg"]=std(vals)
            row[f"{method}_circ_mean_deg"]=circ_mean_deg(vals)
            row[f"{method}_circ_std_deg"]=circ_std_deg(vals)
            row[f"{method}_min_deg"]=float(np.min([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
            row[f"{method}_max_deg"]=float(np.max([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
        rows.append(row)

    # Global over all completed triples.
    ds=list(data.values())
    row=dict(kernel=kernel,global_level=global_level,parent_level=-1,count=len(ds))
    for method in METHODS:
        vals=[method_theta(d,method) for d in ds]
        row[f"{method}_mean_linear_deg"]=mean(vals)
        row[f"{method}_std_linear_deg"]=std(vals)
        row[f"{method}_circ_mean_deg"]=circ_mean_deg(vals)
        row[f"{method}_circ_std_deg"]=circ_std_deg(vals)
        row[f"{method}_min_deg"]=float(np.min([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
        row[f"{method}_max_deg"]=float(np.max([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
    rows.append(row)
    return rows

def residual_loop_rows(kernel,global_level,data):
    # Background by parent_level and method. For a loop, subtract source-node level background.
    bg={}
    for method in METHODS:
        for pl in sorted({d["parent_level"] for d in data.values()}):
            vals=[method_theta(d,method) for d in data.values() if d["parent_level"]==pl]
            bg[(method,pl)]=mean(vals)
        bg[(method,-1)]=mean([method_theta(d,method) for d in data.values()])

    loops=build_loops_dummy(data) # replaced below
    return []

def loop_residuals_for_model(kernel,global_level,model,data):
    loops=build_loops(model,data)
    bg_level={}
    bg_global={}
    for method in METHODS:
        bg_global[method]=mean([method_theta(d,method) for d in data.values()])
        levels=sorted({d["parent_level"] for d in data.values()})
        for pl in levels:
            bg_level[(method,pl)]=mean([method_theta(d,method) for d in data.values() if d["parent_level"]==pl])
    rows=[]
    for item in loops:
        loop=item["loop"]
        for method in METHODS:
            vals=[method_theta(data[u],method) for u in loop]
            if not all(np.isfinite(vals)): continue
            raw=wrap_deg(sum(vals))
            centered_global=wrap_deg(sum(vals)-bg_global[method]*len(loop))
            centered_source_level=wrap_deg(sum(vals)-sum(bg_level[(method,data[u]["parent_level"])] for u in loop))
            rows.append(dict(kernel=kernel,global_level=global_level,loop_mode=item["mode"],loop_level=item["level"],
                             method=method,loop_len=len(loop),raw_wrapped_deg=raw,
                             abs_raw_wrapped_deg=abs(raw),
                             centered_global_deg=centered_global,
                             abs_centered_global_deg=abs(centered_global),
                             centered_level_deg=centered_source_level,
                             abs_centered_level_deg=abs(centered_source_level),
                             mean_theta_step_deg=sum(vals)/len(loop)))
    return rows

def summarize_residual(rows,keys):
    groups={}
    for r in rows:
        k=tuple(r[x] for x in keys)
        groups.setdefault(k,[]).append(r)
    out=[]
    for k,rs in sorted(groups.items()):
        d={keys[i]:k[i] for i in range(len(keys))}
        d.update(count=len(rs),
                 mean_abs_raw_wrapped_deg=mean([r["abs_raw_wrapped_deg"] for r in rs]),
                 mean_abs_centered_global_deg=mean([r["abs_centered_global_deg"] for r in rs]),
                 p95_abs_centered_global_deg=perc([r["abs_centered_global_deg"] for r in rs],95),
                 max_abs_centered_global_deg=maxv([r["abs_centered_global_deg"] for r in rs]),
                 mean_abs_centered_level_deg=mean([r["abs_centered_level_deg"] for r in rs]),
                 p95_abs_centered_level_deg=perc([r["abs_centered_level_deg"] for r in rs],95),
                 max_abs_centered_level_deg=maxv([r["abs_centered_level_deg"] for r in rs]),
                 mean_theta_step_deg=mean([r["mean_theta_step_deg"] for r in rs]),
                 mean_loop_len=mean([r["loop_len"] for r in rs]))
        out.append(d)
    return out

def write_csv(path,rows):
    if not rows: return
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def run(max_level,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    ScalingModel=load_scaling_model()
    kernels=["inverse_square","critical_exp_1over3","exp_0p40","exp_0p25","shell_norm_inverse_square"]
    all_density=[]; all_resid=[]
    lines=["PHASE DENSITY SCALING TEST",f"  max_level={max_level}",""]
    for kernel in kernels:
        model=ScalingModel(kernel=kernel,max_level=max_level,mode="log")
        frontier=[model.root]
        lines.append(f"KERNEL {kernel}")
        for gl in range(1,max_level+1):
            frontier=model.grow_one_level(frontier,gl)
            data={}
            for n in model.nodes.values():
                if len(n.children)==3:
                    d=local_data(model,n.id)
                    if d is not None: data[n.id]=d
            all_density.extend(phase_density_rows(kernel,gl,data))
            res=loop_residuals_for_model(kernel,gl,model,data)
            all_resid.extend(res)
            global_density=[r for r in all_density if r["kernel"]==kernel and r["global_level"]==gl and r["parent_level"]==-1][0]
            polar_mean=global_density["polar_mean_linear_deg"]
            polar_std=global_density["polar_std_linear_deg"]
            polar_loops=[r for r in res if r["method"]=="polar"]
            centered=mean([r["abs_centered_level_deg"] for r in polar_loops])
            lines.append(f"  L={gl}: triples={len(data)}, polar_mean={polar_mean:.9f}, polar_std={polar_std:.9f}, centered_loop_mean={centered:.9f}")
        lines.append("")
    summary_resid_method=summarize_residual(all_resid,["kernel","method","loop_mode","global_level"])
    summary_resid_global=summarize_residual(all_resid,["kernel","method","global_level"])
    write_csv(outdir/"phase_density_by_parent_level.csv",all_density)
    write_csv(outdir/"phase_loop_residuals.csv",all_resid)
    write_csv(outdir/"phase_loop_residuals_by_method_mode_level.csv",summary_resid_method)
    write_csv(outdir/"phase_loop_residuals_by_method_level.csv",summary_resid_global)
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=8)
    ap.add_argument("--outdir",type=Path,default=Path("phase_density_scaling_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir))
if __name__=="__main__": main()
