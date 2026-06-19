#!/usr/bin/env python3
"""
Response-weighted holonomy test for shell-controlled CNNA/NGF growth.

Previous result:
- Minimal SO(3) transport between local J-axes is flat around closure loops.
- Therefore nontrivial phase/curvature, if present, must come from the
  response/operator connection itself.

This test:
1. Builds the shell-controlled growth tower.
2. For every completed local sibling triple, computes:
   - local Markov response P
   - derived skew axis a
   - derived plane B = a^perp
   - 2D response block R2 = B^T P B
   - SO(2) polar rotation U2 of R2
   - local response angle theta
3. For closure loops, transports a 2D frame by:
   x_{i+1} = G_{i->i+1} U_i x_i
   where G is the minimal axis/frame gauge transport and U_i is the local
   response rotation at the source.
4. Compares:
   - gauge-only loop phase
   - response-weighted loop phase
   - response phase minus gauge phase

Interpretation:
- gauge phase ~ 0 reproduces the flat axis-connection result.
- nonzero response-weighted phase means the response connection carries
  U(1)-like holonomy even though the axes glue flatly.
- This is still numerical and not a global J theorem.
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

def column_stochastic(M):
    P = M.copy().astype(float)
    for j in range(3):
        s = P[:, j].sum()
        if s > EPS:
            P[:, j] /= s
        else:
            P[j, j] = 1.0
    return P

def cross_matrix(a):
    x,y,z = a
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]], float)

def skew_axis(A):
    return np.array([A[2,1], A[0,2], A[1,0]], float)

def minimal_rotation(a,b):
    a = a/(np.linalg.norm(a)+EPS)
    b = b/(np.linalg.norm(b)+EPS)
    v = np.cross(a,b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a,b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        h = np.array([1.,0.,0.])
        if abs(float(np.dot(a,h))) > 0.9:
            h = np.array([0.,1.,0.])
        u = h - np.dot(h,a)*a
        u = u/(np.linalg.norm(u)+EPS)
        K = cross_matrix(u)
        return np.eye(3) + 2*(K@K)
    K = cross_matrix(v)
    return np.eye(3) + K + K@K*((1-c)/(s*s))

def plane_basis(axis):
    a = axis/(np.linalg.norm(axis)+EPS)
    h = np.array([1.,0.,0.])
    if abs(float(np.dot(a,h))) > 0.9:
        h = np.array([0.,1.,0.])
    u = h - np.dot(h,a)*a
    u = u/(np.linalg.norm(u)+EPS)
    v = np.cross(a,u)
    v = v/(np.linalg.norm(v)+EPS)
    return np.vstack([u,v]).T

def polar_so2(A2):
    # Nearest O(2) matrix via SVD, then force SO(2) for the rotation component.
    U,s,Vt = np.linalg.svd(A2)
    Q = U @ Vt
    detQ = float(np.linalg.det(Q))
    reflection = 0
    if detQ < 0:
        reflection = 1
        U[:, -1] *= -1
        Q = U @ Vt
    return Q, s, reflection

def angle_so2_deg(Q):
    # For rotation [[cos,-sin],[sin,cos]]
    cosv = float((Q[0,0] + Q[1,1]) / 2.0)
    sinv = float((Q[1,0] - Q[0,1]) / 2.0)
    return math.degrees(math.atan2(sinv, cosv))

def wrap_deg(x):
    return ((x + 180.0) % 360.0) - 180.0

def local_data(model, parent:int) -> Optional[dict]:
    M = model.local_matrix_for_parent(parent)
    if M is None:
        return None
    P = column_stochastic(M)
    A = 0.5*(P-P.T)
    axis = skew_axis(A)
    n = float(np.linalg.norm(axis))
    if n <= EPS:
        return None
    c = np.ones(3)/math.sqrt(3)
    if np.dot(axis,c) < 0:
        axis = -axis
    axis = axis/(np.linalg.norm(axis)+EPS)
    B = plane_basis(axis)
    R2 = B.T @ P @ B
    U2, sing, refl = polar_so2(R2)
    theta = angle_so2_deg(U2)
    contraction = float(np.sqrt(max(0.0, float(np.prod(sing)))))
    anisotropy = float(max(sing)/(min(sing)+EPS))
    ev = np.linalg.eigvals(P)
    return dict(
        parent=parent,
        level=model.nodes[parent].level,
        axis=axis,
        B=B,
        P=P,
        R2=R2,
        U2=U2,
        theta_deg=theta,
        theta_abs_deg=abs(theta),
        polar_reflection=refl,
        singular_0=float(sing[0]),
        singular_1=float(sing[1]),
        contraction=contraction,
        anisotropy=anisotropy,
        complex_pair=float(np.max(np.abs(np.imag(ev))) > 1e-9),
    )

def transition_gauge(data_u, data_v):
    R3 = minimal_rotation(data_u["axis"], data_v["axis"])
    Graw = data_v["B"].T @ R3 @ data_u["B"]
    G, sing, refl = polar_so2(Graw)
    return G, refl

def build_loops(model, data):
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

def loop_response_holonomy(item, data):
    loop=item["loop"]
    Wg=np.eye(2)
    Wr=np.eye(2)
    raw_theta_sum=0.0
    refl_g=0
    refl_u=0
    for u,v in zip(loop, loop[1:]+[loop[0]]):
        du=data[u]; dv=data[v]
        G, rg = transition_gauge(du,dv)
        refl_g += rg
        U = du["U2"]
        raw_theta_sum += du["theta_deg"]
        Wr = G @ U @ Wr
        Wg = G @ Wg
        refl_u += du["polar_reflection"]
    Ug, sg, rgloop = polar_so2(Wg)
    Ur, sr, rrloop = polar_so2(Wr)
    gauge_phase = angle_so2_deg(Ug)
    response_phase = angle_so2_deg(Ur)
    excess = wrap_deg(response_phase - gauge_phase)
    raw_wrapped = wrap_deg(raw_theta_sum)
    return dict(
        mode=item["mode"],
        level=item["level"],
        base=item["base"],
        loop_len=len(loop),
        loop_nodes=" ".join(map(str,loop[:30]))+(" ..." if len(loop)>30 else ""),
        gauge_phase_deg=gauge_phase,
        response_phase_deg=response_phase,
        excess_response_phase_deg=excess,
        abs_excess_response_phase_deg=abs(excess),
        raw_theta_sum_deg=raw_theta_sum,
        raw_theta_sum_wrapped_deg=raw_wrapped,
        abs_raw_theta_sum_wrapped_deg=abs(raw_wrapped),
        mean_local_theta_deg=raw_theta_sum/len(loop),
        gauge_reflections=refl_g,
        local_polar_reflections=refl_u,
        loop_polar_reflection=rrloop,
        response_singular_0=float(sr[0]),
        response_singular_1=float(sr[1]),
        response_condition=float(sr[0]/(sr[1]+EPS)),
    )

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
        phases=[r["abs_excess_response_phase_deg"] for r in rs]
        raw=[r["abs_raw_theta_sum_wrapped_deg"] for r in rs]
        d.update(count=len(rs),
                 mean_abs_excess_phase_deg=mean(phases),
                 p95_abs_excess_phase_deg=perc(phases,95),
                 max_abs_excess_phase_deg=maxv(phases),
                 mean_abs_raw_wrapped_phase_deg=mean(raw),
                 p95_abs_raw_wrapped_phase_deg=perc(raw,95),
                 max_abs_raw_wrapped_phase_deg=maxv(raw),
                 mean_gauge_phase_deg=mean([abs(r["gauge_phase_deg"]) for r in rs]),
                 mean_local_theta_deg=mean([r["mean_local_theta_deg"] for r in rs]),
                 mean_response_condition=mean([r["response_condition"] for r in rs]),
                 max_response_condition=maxv([r["response_condition"] for r in rs]),
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

    data={}
    local_rows=[]
    for n in model.nodes.values():
        if len(n.children)==3:
            d=local_data(model,n.id)
            if d is not None:
                data[n.id]=d
                local_rows.append(dict(parent=n.id,level=n.level,theta_deg=d["theta_deg"],
                                       theta_abs_deg=d["theta_abs_deg"],
                                       polar_reflection=d["polar_reflection"],
                                       contraction=d["contraction"],
                                       anisotropy=d["anisotropy"],
                                       complex_pair=d["complex_pair"],
                                       axis_x=float(d["axis"][0]),axis_y=float(d["axis"][1]),axis_z=float(d["axis"][2])))
    loops=build_loops(model,data)
    rows=[loop_response_holonomy(item,data) for item in loops]
    by_mode=summarize(rows,["mode"])
    by_level=summarize(rows,["mode","level"])
    write_csv(outdir/"response_holonomy_local_data.csv",local_rows)
    write_csv(outdir/"response_holonomy_loops.csv",rows)
    write_csv(outdir/"response_holonomy_by_mode.csv",by_mode)
    write_csv(outdir/"response_holonomy_by_level.csv",by_level)
    lines=["RESPONSE-WEIGHTED HOLONOMY TEST",
           f"  final level={max_level}, nodes={len(model.nodes)}, local operators={len(data)}, loops={len(rows)}",
           f"  mean local theta={mean([r['theta_deg'] for r in local_rows]):.9f} deg",
           f"  mean |local theta|={mean([r['theta_abs_deg'] for r in local_rows]):.9f} deg",
           f"  local polar reflection fraction={mean([r['polar_reflection'] for r in local_rows]):.6f}",
           "",
           "BY MODE"]
    for r in by_mode:
        lines.append(f"  {r['mode']}: count={r['count']}, mean|excess|={r['mean_abs_excess_phase_deg']:.9f} deg, "
                     f"p95|excess|={r['p95_abs_excess_phase_deg']:.9f}, max|excess|={r['max_abs_excess_phase_deg']:.9f}, "
                     f"mean gauge phase={r['mean_gauge_phase_deg']:.3e}, mean local theta per step={r['mean_local_theta_deg']:.9f}, "
                     f"mean cond={r['mean_response_condition']:.6f}, mean len={r['mean_loop_len']:.2f}")
    lines+=["","SELECTED BY LEVEL"]
    for r in by_level:
        if r["mode"] in ("sibling_cycle","parent_fan_triangle","parent_child_ring"):
            lines.append(f"  {r['mode']} L={r['level']}: count={r['count']}, "
                         f"mean|excess|={r['mean_abs_excess_phase_deg']:.9f}, max|excess|={r['max_abs_excess_phase_deg']:.9f}, "
                         f"mean theta/step={r['mean_local_theta_deg']:.9f}")
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=9)
    ap.add_argument("--outdir",type=Path,default=Path("response_weighted_holonomy_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir))
if __name__=="__main__": main()
