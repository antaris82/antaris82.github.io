#!/usr/bin/env python3
"""
Focused high-L phase limit test for the shell-normalized kernel.

This is optimized compared to the broad five-kernel test:
- only kernel = shell_norm_inverse_square
- only log dynamic mode
- aggregate summaries only
- computes local phase density and centered loop residuals by level
- fits theta_inf and decay models from the produced levels

It is still a numerical diagnostic, not a theorem.
"""

from __future__ import annotations

import argparse, csv, importlib.util, math, sys, time
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12

@dataclass
class Node:
    id: int
    parent: Optional[int]
    level: int
    birth_order: int
    birth_time: int
    birth_g: float
    g: float
    children: List[int] = field(default_factory=list)

class ShellGrowth:
    def __init__(
        self,
        branching: int = 3,
        base: float = 1.0,
        alpha_env: float = 0.22,
        ancestor_env_decay: float = 0.55,
        br_ancestor: float = 0.045,
        br_sibling: float = 0.035,
    ):
        self.b=branching
        self.base=base
        self.alpha_env=alpha_env
        self.ancestor_env_decay=ancestor_env_decay
        self.br_ancestor=br_ancestor
        self.br_sibling=br_sibling
        self.nodes: Dict[int,Node]={}
        self.local_w: Dict[int,Dict[Tuple[int,int],float]]=defaultdict(lambda: defaultdict(float))
        self.next_id=0
        self.t=0
        root=self._new_node(None,0,0,1.0)
        self.root=root.id
    def _new_node(self,parent,level,birth_order,birth_g):
        n=Node(self.next_id,parent,level,birth_order,self.t,birth_g,birth_g)
        self.nodes[n.id]=n; self.next_id+=1
        if parent is not None:
            self.nodes[parent].children.append(n.id)
        return n
    def kernel(self,d:int)->float:
        return 1.0/((self.b**(d-1))*(d*d))
    def parent_line(self,parent:int)->List[int]:
        out=[]; cur=parent
        while cur is not None:
            out.append(cur); cur=self.nodes[cur].parent
        return out
    def birth_env(self,parent:int,older:List[int])->float:
        env=0.0
        for d,a in enumerate(self.parent_line(parent),start=1):
            env+=self.nodes[a].g*(self.ancestor_env_decay**(d-1))
        for s in older:
            env+=self.nodes[s].g
        return env
    def child_g(self,env:float)->float:
        return self.base+self.alpha_env*math.log1p(env)
    def add_child(self,parent:int,order:int)->int:
        self.t+=1
        older=list(self.nodes[parent].children)
        env=self.birth_env(parent,older)
        bg=self.child_g(env)
        child=self._new_node(parent,self.nodes[parent].level+1,order,bg)
        c=child.id
        for s in older:
            i=self.nodes[s].birth_order; j=order
            self.local_w[parent][(i,j)] += self.alpha_env*self.nodes[s].g/(env+EPS)*bg
        for d,a in enumerate(self.parent_line(parent),start=1):
            self.nodes[a].g += self.br_ancestor*bg*self.kernel(d)
        for s in older:
            i=order; j=self.nodes[s].birth_order
            delta=self.br_sibling*bg
            self.nodes[s].g += delta
            self.local_w[parent][(i,j)] += delta
        return c
    def grow_one_level(self,frontier:List[int])->List[int]:
        out=[]
        for p in frontier:
            for k in range(1,self.b+1):
                out.append(self.add_child(p,k))
        return out
    def local_matrix(self,parent:int)->Optional[np.ndarray]:
        if len(self.nodes[parent].children)!=3:
            return None
        M=np.zeros((3,3),float); w=self.local_w[parent]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    M[j-1,i-1]=w.get((i,j),0.0)
        return M

def wrap_deg(x):
    return ((x+180.0)%360.0)-180.0

def mean(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")
def std(vals):
    xs=[float(v) for v in vals if np.isfinite(float(v))]
    return float(np.std(xs)) if xs else float("nan")
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
    if abs(float(np.dot(a,h)))>0.9:
        h=np.array([0.,1.,0.])
    u=h-np.dot(h,a)*a; u=u/(np.linalg.norm(u)+EPS)
    v=np.cross(a,u); v=v/(np.linalg.norm(v)+EPS)
    return np.vstack([u,v]).T

def polar_so2(A2):
    U,s,Vt=np.linalg.svd(A2)
    Q=U@Vt
    refl=0
    if np.linalg.det(Q)<0:
        refl=1; U[:,-1]*=-1; Q=U@Vt
    return Q,s,refl

def angle_so2_deg(Q):
    cosv=float((Q[0,0]+Q[1,1])/2.0)
    sinv=float((Q[1,0]-Q[0,1])/2.0)
    return math.degrees(math.atan2(sinv,cosv))

def local_phase(model:ShellGrowth,parent:int)->Optional[dict]:
    M=model.local_matrix(parent)
    if M is None:
        return None
    P=column_stochastic(M)
    A=0.5*(P-P.T)
    axis=skew_axis(A)
    if np.linalg.norm(axis)<=EPS:
        return None
    c=np.ones(3)/math.sqrt(3)
    if np.dot(axis,c)<0:
        axis=-axis
    axis=axis/(np.linalg.norm(axis)+EPS)
    B=plane_basis(axis)
    R2=B.T@P@B
    Q,sing,_=polar_so2(R2)
    theta_polar=angle_so2_deg(Q)
    ev=np.linalg.eigvals(P)
    ces=[z for z in ev if abs(z.imag)>1e-9]
    theta_eigen=float("nan")
    if ces:
        z=max(ces,key=lambda z:z.imag)
        theta_eigen=math.degrees(math.atan2(z.imag,z.real))
    S2=0.5*(R2+R2.T)
    theta_G=float("nan")
    for sign,cand in [(-1,-S2),(1,S2)]:
        eig=np.linalg.eigvalsh(cand)
        if np.all(eig>1e-10):
            try:
                L=np.linalg.cholesky(cand); C=L.T
                Rg=C@R2@np.linalg.inv(C)
                Qg,_,_=polar_so2(Rg)
                theta_G=angle_so2_deg(Qg)
                break
            except np.linalg.LinAlgError:
                pass
    return dict(parent=parent,level=model.nodes[parent].level,
                polar=theta_polar,eigen=theta_eigen,G_weighted=theta_G)

def completed_parent_ids(model:ShellGrowth)->List[int]:
    return [n.id for n in model.nodes.values() if len(n.children)==3]

def build_main_loops(model:ShellGrowth, data:Dict[int,dict]):
    loops=[]
    for p in data:
        cs=[c for c in model.nodes[p].children if c in data]
        if len(cs)==3:
            c1,c2,c3=cs
            loops.append(("sibling_cycle",model.nodes[p].level+1,[c1,c2,c3]))
            loops.append(("parent_child_ring",model.nodes[p].level,[p,c1,c2,c3]))
            loops.append(("parent_fan_triangle",model.nodes[p].level,[p,c1,c2]))
            loops.append(("parent_fan_triangle",model.nodes[p].level,[p,c2,c3]))
            loops.append(("parent_fan_triangle",model.nodes[p].level,[p,c3,c1]))
    return loops

METHODS=["polar","eigen","G_weighted"]

def summarize_level(model:ShellGrowth, gl:int):
    data={}
    for p in completed_parent_ids(model):
        d=local_phase(model,p)
        if d is not None:
            data[p]=d
    rows=[]
    row=dict(global_level=gl,nodes=len(model.nodes),completed=len(data))
    for method in METHODS:
        vals=[d[method] for d in data.values()]
        row[f"{method}_mean"]=mean(vals)
        row[f"{method}_std"]=std(vals)
        row[f"{method}_min"]=float(np.min([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
        row[f"{method}_max"]=float(np.max([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("nan")
    loops=build_main_loops(model,data)
    for method in METHODS:
        # Background by source parent level
        by_level=defaultdict(list)
        for d in data.values():
            by_level[d["level"]].append(d[method])
        bg={k:mean(v) for k,v in by_level.items()}
        residuals=[]; raw=[]
        by_mode=defaultdict(list)
        for mode,lvl,loop in loops:
            vals=[data[u][method] for u in loop]
            if not all(np.isfinite(vals)): continue
            r=wrap_deg(sum(vals))
            c=wrap_deg(sum(vals)-sum(bg[data[u]["level"]] for u in loop))
            raw.append(abs(r)); residuals.append(abs(c)); by_mode[mode].append(abs(c))
        row[f"{method}_loop_count"]=len(residuals)
        row[f"{method}_raw_loop_mean"]=mean(raw)
        row[f"{method}_centered_mean"]=mean(residuals)
        row[f"{method}_centered_p95"]=perc(residuals,95)
        for mode in ["sibling_cycle","parent_fan_triangle","parent_child_ring"]:
            row[f"{method}_{mode}_centered_mean"]=mean(by_mode[mode])
    return row

def rmse(y,yhat):
    return float(np.sqrt(np.mean((y-yhat)**2)))

def fit_theta(L,y):
    best=None
    for r in np.linspace(0.2,0.98,157):
        f=r**L
        X=np.vstack([np.ones_like(f),f]).T
        beta,*_=np.linalg.lstsq(X,y,rcond=None)
        yhat=X@beta
        obj=dict(model="theta_inf_plus_A_r_pow_L",theta_inf=float(beta[0]),amp=float(beta[1]),r=float(r),rmse=rmse(y,yhat))
        if best is None or obj["rmse"]<best["rmse"]:
            best=obj
    return best

def fit_decay(L,y):
    mask=np.isfinite(y)&(y>0)
    L=L[mask]; y=y[mask]
    if len(y)<4: return {}
    logy=np.log(y)
    X=np.vstack([np.ones_like(L),L]).T
    beta,*_=np.linalg.lstsq(X,logy,rcond=None)
    yhat=np.exp(X@beta)
    expfit=dict(model="C_r_pow_L",C=float(math.exp(beta[0])),r=float(math.exp(beta[1])),rmse=rmse(y,yhat))
    N=(3.0**(L+1)-1.0)/2.0
    X=np.vstack([np.ones_like(N),np.log(N)]).T
    beta,*_=np.linalg.lstsq(X,logy,rcond=None)
    yhat=np.exp(X@beta)
    powfit=dict(model="C_N_minus_alpha",C=float(math.exp(beta[0])),alpha=float(-beta[1]),rmse=rmse(y,yhat))
    return expfit if expfit["rmse"]<=powfit["rmse"] else powfit

def write_csv(path,rows):
    if not rows: return
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def run(max_level,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    model=ShellGrowth()
    frontier=[model.root]
    level_rows=[]
    t0=time.time()
    for gl in range(1,max_level+1):
        frontier=model.grow_one_level(frontier)
        row=summarize_level(model,gl)
        row["elapsed_sec"]=time.time()-t0
        level_rows.append(row)
    write_csv(outdir/"shell_phase_highL_levels.csv",level_rows)
    L=np.array([r["global_level"] for r in level_rows if r["global_level"]>=4],float)
    fits=[]
    for method in METHODS:
        theta=np.array([r[f"{method}_mean"] for r in level_rows if r["global_level"]>=4],float)
        ft=fit_theta(L,theta)
        ft.update(kind="theta_mean",method=method,last=float(theta[-1]))
        fits.append(ft)
        stdv=np.array([r[f"{method}_std"] for r in level_rows if r["global_level"]>=4],float)
        fd=fit_decay(L,stdv); fd.update(kind="phase_std",method=method,last=float(stdv[-1])); fits.append(fd)
        res=np.array([r[f"{method}_centered_mean"] for r in level_rows if r["global_level"]>=4],float)
        fr=fit_decay(L,res); fr.update(kind="centered_residual",method=method,last=float(res[-1])); fits.append(fr)
    write_csv(outdir/"shell_phase_highL_fits.csv",fits)
    lines=["SHELL PHASE LIMIT HIGH-L TEST",
           f"  max_level={max_level}",
           f"  final nodes={level_rows[-1]['nodes']}",
           f"  final completed={level_rows[-1]['completed']}",
           ""]
    for r in level_rows:
        lines.append(f"  L={r['global_level']}: nodes={r['nodes']}, completed={r['completed']}, polar_mean={r['polar_mean']:.9f}, polar_std={r['polar_std']:.9f}, polar_centered={r['polar_centered_mean']:.9f}, elapsed={r['elapsed_sec']:.2f}s")
    lines+=["","FITS"]
    for f in fits:
        if f["kind"]=="theta_mean":
            lines.append(f"  {f['method']} theta_inf={f['theta_inf']:.9f}, last={f['last']:.9f}, r={f['r']:.6f}, rmse={f['rmse']:.6e}")
        elif f["model"]=="C_r_pow_L":
            lines.append(f"  {f['method']} {f['kind']} decay: last={f['last']:.9f}, r={f['r']:.6f}, rmse={f['rmse']:.6e}")
        else:
            lines.append(f"  {f['method']} {f['kind']} decay: last={f['last']:.9f}, alpha={f['alpha']:.6f}, rmse={f['rmse']:.6e}")
    summary="\n".join(lines)
    (outdir/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-level",type=int,default=10)
    ap.add_argument("--outdir",type=Path,default=Path("shell_phase_limit_highL_out"))
    args=ap.parse_args()
    print(run(args.max_level,args.outdir))
if __name__=="__main__": main()
