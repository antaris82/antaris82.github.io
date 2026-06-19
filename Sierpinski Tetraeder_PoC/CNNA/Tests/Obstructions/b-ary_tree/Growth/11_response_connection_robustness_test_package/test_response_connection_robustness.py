#!/usr/bin/env python3
"""
Response connection robustness test.

Compares several ways of extracting a local response phase from the same
shell-controlled CNNA/NGF local Markov operator:

1. polar: SO(2) polar rotation of R2 = B^T P B
2. eigen: argument of the complex eigenvalue of P
3. skew_iso: atan2(skew coefficient, isotropic trace coefficient) of R2
4. G_weighted: polar phase after similarity transform by candidate G = -S2
   (or +S2 if that is the positive definite sign)
5. centered versions: loop phase after subtracting global mean local phase

The goal is to check whether response holonomy is robust or just an artifact
of the polar extraction convention.

This is a numerical diagnostic, not a Lean theorem.
"""

from __future__ import annotations
import argparse, csv, importlib.util, math, sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

EPS = 1e-12

def load_scaling_model():
    path = Path("/mnt/data/test_conductance_scaling_generalization.py")
    spec = importlib.util.spec_from_file_location("scaling", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ScalingModel

def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0

def mean(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")
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

def cross_matrix(a):
    x,y,z=a
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]],float)

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

def minimal_rotation(a,b):
    a=a/(np.linalg.norm(a)+EPS); b=b/(np.linalg.norm(b)+EPS)
    v=np.cross(a,b); s=float(np.linalg.norm(v)); c=float(np.dot(a,b))
    if s<1e-12:
        if c>0: return np.eye(3)
        h=np.array([1.,0.,0.])
        if abs(float(np.dot(a,h)))>0.9: h=np.array([0.,1.,0.])
        u=h-np.dot(h,a)*a; u=u/(np.linalg.norm(u)+EPS)
        K=cross_matrix(u)
        return np.eye(3)+2*(K@K)
    K=cross_matrix(v)
    return np.eye(3)+K+K@K*((1-c)/(s*s))

def transition_gauge(du,dv):
    R3=minimal_rotation(du["axis"],dv["axis"])
    Graw=dv["B"].T@R3@du["B"]
    G,_,refl=polar_so2(Graw)
    return G,refl

def local_data(model,parent:int)->Optional[dict]:
    M=model.local_matrix_for_parent(parent)
    if M is None: return None
    P=column_stochastic(M)
    A=0.5*(P-P.T)
    axis=skew_axis(A)
    if np.linalg.norm(axis)<=EPS: return None
    c=np.ones(3)/math.sqrt(3)
    if np.dot(axis,c)<0: axis=-axis
    axis=axis/(np.linalg.norm(axis)+EPS)
    B=plane_basis(axis)
    R2=B.T@P@B
    S2=0.5*(R2+R2.T)
    A2=0.5*(R2-R2.T)

    Qpol, sing, refl = polar_so2(R2)
    theta_polar=angle_so2_deg(Qpol)

    ev=np.linalg.eigvals(P)
    complex_eigs=[z for z in ev if abs(z.imag)>1e-9]
    if complex_eigs:
        z=max(complex_eigs,key=lambda z:z.imag)
        theta_eigen=math.degrees(math.atan2(z.imag,z.real))
    else:
        theta_eigen=float("nan")

    q=float((R2[0,0]+R2[1,1])/2.0)
    k=float((R2[1,0]-R2[0,1])/2.0)
    theta_skew_iso=math.degrees(math.atan2(k,q))

    theta_G=float("nan")
    G_sign=0
    G_cond=float("nan")
    # Candidate metric: choose sign of S2 that is positive definite.
    for sign,cand in [(-1,-S2),(1,S2)]:
        eig=np.linalg.eigvalsh(cand)
        if np.all(eig>1e-10):
            try:
                L=np.linalg.cholesky(cand)  # cand=L L^T
                C=L.T
                Rg=C@R2@np.linalg.inv(C)
                Qg,_,_=polar_so2(Rg)
                theta_G=angle_so2_deg(Qg)
                G_sign=sign
                G_cond=float(max(eig)/(min(eig)+EPS))
                break
            except np.linalg.LinAlgError:
                pass

    return dict(parent=parent,level=model.nodes[parent].level,axis=axis,B=B,P=P,R2=R2,
                theta_polar=theta_polar,theta_eigen=theta_eigen,
                theta_skew_iso=theta_skew_iso,theta_G=theta_G,
                polar_reflection=refl,polar_cond=float(max(sing)/(min(sing)+EPS)),
                G_sign=G_sign,G_cond=G_cond,
                complex_pair=float(len(complex_eigs)>0))

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
    by_level={}
    for p in data:
        by_level.setdefault(model.nodes[p].level,[]).append(p)
    for level,ps in sorted(by_level.items()):
        ps=sorted(ps,key=lambda x:model.nodes[x].birth_time)
        if len(ps)>=3:
            loops.append(dict(mode="level_birth_ring",level=level,base=-1,loop=ps))
            chunk=9
            for i in range(0,len(ps)-chunk+1,chunk):
                loops.append(dict(mode="level_birth_ring_chunk9",level=level,base=-1,loop=ps[i:i+chunk]))
    return loops

METHODS=["polar","eigen","skew_iso","G_weighted"]

def method_theta(d,method):
    if method=="polar": return d["theta_polar"]
    if method=="eigen": return d["theta_eigen"]
    if method=="skew_iso": return d["theta_skew_iso"]
    if method=="G_weighted": return d["theta_G"]
    raise ValueError(method)

def gauge_phase_for_loop(loop,data):
    W=np.eye(2); refl=0
    for u,v in zip(loop,loop[1:]+[loop[0]]):
        G,r=transition_gauge(data[u],data[v])
        W=G@W; refl+=r
    Q,_,_=polar_so2(W)
    return angle_so2_deg(Q),refl

def loop_rows_for_item(item,data,global_means):
    loop=item["loop"]
    gauge_phase,gauge_refl=gauge_phase_for_loop(loop,data)
    rows=[]
    for method in METHODS:
        vals=[method_theta(data[u],method) for u in loop]
        if not all(np.isfinite(vals)):
            continue
        raw=sum(vals)
        wrapped=wrap_deg(raw)
        excess=wrap_deg(wrapped-gauge_phase)
        centered=wrap_deg(raw - global_means[method]*len(loop))
        rows.append(dict(mode=item["mode"],level=item["level"],base=item["base"],
                         method=method,loop_len=len(loop),
                         gauge_phase_deg=gauge_phase,
                         raw_phase_sum_deg=raw,
                         raw_phase_wrapped_deg=wrapped,
                         excess_phase_deg=excess,
                         abs_excess_phase_deg=abs(excess),
                         centered_phase_deg=centered,
                         abs_centered_phase_deg=abs(centered),
                         mean_theta_step_deg=raw/len(loop),
                         gauge_reflections=gauge_refl,
                         loop_nodes=" ".join(map(str,loop[:30]))+(" ..." if len(loop)>30 else "")))
    return rows

def summarize(rows,keys):
    groups={}
    for r in rows:
        k=tuple(r[x] for x in keys)
        groups.setdefault(k,[]).append(r)
    out=[]
    for k,rs in sorted(groups.items()):
        d={keys[i]:k[i] for i in range(len(keys))}
        d.update(count=len(rs),
                 mean_abs_excess_phase_deg=mean([r["abs_excess_phase_deg"] for r in rs]),
                 p95_abs_excess_phase_deg=perc([r["abs_excess_phase_deg"] for r in rs],95),
                 max_abs_excess_phase_deg=maxv([r["abs_excess_phase_deg"] for r in rs]),
                 mean_abs_centered_phase_deg=mean([r["abs_centered_phase_deg"] for r in rs]),
                 p95_abs_centered_phase_deg=perc([r["abs_centered_phase_deg"] for r in rs],95),
                 max_abs_centered_phase_deg=maxv([r["abs_centered_phase_deg"] for r in rs]),
                 mean_theta_step_deg=mean([r["mean_theta_step_deg"] for r in rs]),
                 mean_gauge_phase_abs_deg=mean([abs(r["gauge_phase_deg"]) for r in rs]),
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
    model=ScalingModel(kernel="shell_norm_inverse_square",max_level=max_level,mode="log")
    frontier=[model.root]
    for level in range(1,max_level+1):
        frontier=model.grow_one_level(frontier,level)

    data={}; local_rows=[]
    for n in model.nodes.values():
        if len(n.children)==3:
            d=local_data(model,n.id)
            if d is not None:
                data[n.id]=d
                local_rows.append(dict(parent=n.id,level=n.level,
                                       theta_polar=d["theta_polar"],theta_eigen=d["theta_eigen"],
                                       theta_skew_iso=d["theta_skew_iso"],theta_G=d["theta_G"],
                                       polar_reflection=d["polar_reflection"],polar_cond=d["polar_cond"],
                                       G_sign=d["G_sign"],G_cond=d["G_cond"],
                                       complex_pair=d["complex_pair"]))
    global_means={m:mean([r[f"theta_{m if m!='G_weighted' else 'G'}"] if m!="skew_iso" else r["theta_skew_iso"] for r in local_rows]) for m in METHODS}
    # Easier explicit:
    global_means["polar"]=mean([r["theta_polar"] for r in local_rows])
    global_means["eigen"]=mean([r["theta_eigen"] for r in local_rows])
    global_means["skew_iso"]=mean([r["theta_skew_iso"] for r in local_rows])
    global_means["G_weighted"]=mean([r["theta_G"] for r in local_rows])

    loops=build_loops(model,data)
    rows=[]
    for item in loops:
        rows.extend(loop_rows_for_item(item,data,global_means))
    by_method_mode=summarize(rows,["method","mode"])
    by_method=summarize(rows,["method"])
    by_level=summarize(rows,["method","mode","level"])
    write_csv(outdir/"response_connection_local_phases.csv",local_rows)
    write_csv(outdir/"response_connection_loop_phases.csv",rows)
    write_csv(outdir/"response_connection_by_method_mode.csv",by_method_mode)
    write_csv(outdir/"response_connection_by_method.csv",by_method)
    write_csv(outdir/"response_connection_by_level.csv",by_level)

    lines=["RESPONSE CONNECTION ROBUSTNESS TEST",
           f"  final level={max_level}, nodes={len(model.nodes)}, local operators={len(data)}, loops={len(loops)}",
           "  global mean local phase:"]
    for m in METHODS:
        lines.append(f"    {m}: {global_means[m]:.9f} deg")
    lines+=["","BY METHOD"]
    for r in by_method:
        lines.append(f"  {r['method']}: count={r['count']}, mean|excess|={r['mean_abs_excess_phase_deg']:.9f}, "
                     f"p95|excess|={r['p95_abs_excess_phase_deg']:.9f}, "
                     f"mean|centered|={r['mean_abs_centered_phase_deg']:.9f}, "
                     f"p95|centered|={r['p95_abs_centered_phase_deg']:.9f}, "
                     f"mean theta step={r['mean_theta_step_deg']:.9f}")
    lines+=["","SELECTED METHOD × MODE"]
    for r in by_method_mode:
        if r["mode"] in ("sibling_cycle","parent_fan_triangle","parent_child_ring"):
            lines.append(f"  {r['method']} / {r['mode']}: count={r['count']}, "
                         f"mean|excess|={r['mean_abs_excess_phase_deg']:.9f}, "
                         f"mean|centered|={r['mean_abs_centered_phase_deg']:.9f}, "
                         f"mean theta={r['mean_theta_step_deg']:.9f}")
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=9)
    ap.add_argument("--outdir",type=Path,default=Path("response_connection_robustness_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir))
if __name__=="__main__": main()
