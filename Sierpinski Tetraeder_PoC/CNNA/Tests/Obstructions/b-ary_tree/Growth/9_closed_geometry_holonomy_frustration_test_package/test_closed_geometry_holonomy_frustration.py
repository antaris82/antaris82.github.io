#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, importlib.util, math, sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
EPS=1e-12

def load_scaling_model():
    path=Path("/mnt/data/test_conductance_scaling_generalization.py")
    spec=importlib.util.spec_from_file_location("scaling", path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod.ScalingModel

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

def local_axis(model,parent:int)->Optional[np.ndarray]:
    M=model.local_matrix_for_parent(parent)
    if M is None: return None
    P=column_stochastic(M)
    A=0.5*(P-P.T)
    a=skew_axis(A)
    n=float(np.linalg.norm(a))
    if n<=EPS: return None
    c=np.ones(3)/math.sqrt(3)
    if np.dot(a,c)<0: a=-a
    return a/(np.linalg.norm(a)+EPS)

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

def plane_basis(axis):
    a=axis/(np.linalg.norm(axis)+EPS)
    h=np.array([1.,0.,0.])
    if abs(float(np.dot(a,h)))>0.9: h=np.array([0.,1.,0.])
    u=h-np.dot(h,a)*a; u=u/(np.linalg.norm(u)+EPS)
    v=np.cross(a,u); v=v/(np.linalg.norm(v)+EPS)
    return np.vstack([u,v]).T

def rotation_angle_deg(R):
    x=max(-1.,min(1.,(float(np.trace(R))-1.)/2.))
    return math.degrees(math.acos(x))

def signed_plane_phase_deg(R,axis):
    B=plane_basis(axis); U=B.T@R@B
    cosv=float((U[0,0]+U[1,1])/2.)
    sinv=float((U[1,0]-U[0,1])/2.)
    return math.degrees(math.atan2(sinv,cosv))

def loop_holonomy(loop,axes):
    R=np.eye(3)
    for u,v in zip(loop,loop[1:]+[loop[0]]):
        R=minimal_rotation(axes[u],axes[v])@R
    a0=axes[loop[0]]
    phase=signed_plane_phase_deg(R,a0)
    angle=rotation_angle_deg(R)
    axis_resid=float(np.linalg.norm(R@a0-a0))
    J0=cross_matrix(a0)
    Jmis=float(np.linalg.norm(R@J0@R.T-J0,ord="fro"))
    B=plane_basis(a0); U=B.T@R@B
    oerr=float(np.linalg.norm(U.T@U-np.eye(2)))
    return dict(loop_len=len(loop),holonomy_angle_deg=angle,
                signed_plane_phase_deg=phase,abs_plane_phase_deg=abs(phase),
                axis_closure_residual=axis_resid,J_loop_mismatch=Jmis,
                plane_U_orthogonality_error=oerr)

def build_loops(model,axes):
    loops=[]
    for p in axes:
        cs=[c for c in model.nodes[p].children if c in axes]
        if len(cs)==3:
            c1,c2,c3=cs
            loops.append(dict(mode="sibling_cycle",level=model.nodes[p].level+1,base=p,loop=[c1,c2,c3]))
            loops.append(dict(mode="parent_child_ring",level=model.nodes[p].level,base=p,loop=[p,c1,c2,c3]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c1,c2]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c2,c3]))
            loops.append(dict(mode="parent_fan_triangle",level=model.nodes[p].level,base=p,loop=[p,c3,c1]))
    by_level={}
    for p in axes:
        by_level.setdefault(model.nodes[p].level,[]).append(p)
    for level,ps in sorted(by_level.items()):
        ps=sorted(ps,key=lambda x:model.nodes[x].birth_time)
        if len(ps)>=3:
            loops.append(dict(mode="level_birth_ring",level=level,base=-1,loop=ps))
            chunk=9
            for i in range(0,len(ps)-chunk+1,chunk):
                loops.append(dict(mode="level_birth_ring_chunk9",level=level,base=-1,loop=ps[i:i+chunk]))
    return loops

def mean(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")
def maxv(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.max(xs)) if xs else float("nan")
def perc(vals,q):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.percentile(xs,q)) if xs else float("nan")

def summarize(rows, keys):
    groups={}
    for r in rows:
        k=tuple(r[x] for x in keys)
        groups.setdefault(k,[]).append(r)
    out=[]
    for k,rs in sorted(groups.items()):
        d={keys[i]:k[i] for i in range(len(keys))}
        d.update(count=len(rs),
                 mean_abs_phase_deg=mean([r["abs_plane_phase_deg"] for r in rs]),
                 p95_abs_phase_deg=perc([r["abs_plane_phase_deg"] for r in rs],95),
                 max_abs_phase_deg=maxv([r["abs_plane_phase_deg"] for r in rs]),
                 mean_holonomy_angle_deg=mean([r["holonomy_angle_deg"] for r in rs]),
                 max_holonomy_angle_deg=maxv([r["holonomy_angle_deg"] for r in rs]),
                 mean_axis_closure_residual=mean([r["axis_closure_residual"] for r in rs]),
                 max_axis_closure_residual=maxv([r["axis_closure_residual"] for r in rs]),
                 mean_J_loop_mismatch=mean([r["J_loop_mismatch"] for r in rs]),
                 max_J_loop_mismatch=maxv([r["J_loop_mismatch"] for r in rs]),
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
    axes={}
    for n in model.nodes.values():
        if len(n.children)==3:
            a=local_axis(model,n.id)
            if a is not None: axes[n.id]=a
    loops=build_loops(model,axes)
    rows=[]
    for item in loops:
        h=loop_holonomy(item["loop"],axes)
        rows.append(dict(mode=item["mode"],level=item["level"],base=item["base"],
                         loop_nodes=" ".join(map(str,item["loop"][:30]))+(" ..." if len(item["loop"])>30 else ""),
                         **h))
    by_level=summarize(rows,["mode","level"])
    by_mode=summarize(rows,["mode"])
    write_csv(outdir/"closed_holonomy_loops.csv",rows)
    write_csv(outdir/"closed_holonomy_by_level.csv",by_level)
    write_csv(outdir/"closed_holonomy_by_mode.csv",by_mode)
    lines=["CLOSED GEOMETRY HOLONOMY / FRUSTRATION TEST",
           f"  final level={max_level}, nodes={len(model.nodes)}, local axes={len(axes)}, loops={len(rows)}",
           "",
           "BY MODE"]
    for r in by_mode:
        lines.append(f"  {r['mode']}: count={r['count']}, mean|phase|={r['mean_abs_phase_deg']:.9f} deg, "
                     f"p95|phase|={r['p95_abs_phase_deg']:.9f}, max|phase|={r['max_abs_phase_deg']:.9f}, "
                     f"mean axis residual={r['mean_axis_closure_residual']:.3e}, mean J mismatch={r['mean_J_loop_mismatch']:.3e}, "
                     f"mean len={r['mean_loop_len']:.2f}")
    lines+=["","SELECTED BY LEVEL"]
    for r in by_level:
        if r["mode"] in ("sibling_cycle","parent_fan_triangle","parent_child_ring"):
            lines.append(f"  {r['mode']} L={r['level']}: count={r['count']}, "
                         f"mean|phase|={r['mean_abs_phase_deg']:.9f}, max|phase|={r['max_abs_phase_deg']:.9f}, "
                         f"mean Jmis={r['mean_J_loop_mismatch']:.3e}")
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=9)
    ap.add_argument("--outdir",type=Path,default=Path("closed_geometry_holonomy_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir))
if __name__=="__main__": main()
