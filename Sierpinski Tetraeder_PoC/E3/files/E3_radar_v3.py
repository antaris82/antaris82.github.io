
import math, json
import numpy as np
import pandas as pd

def eigh_H(H):
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

def psi_from_source(evals, evecs, src_index, tgrid, J=1.0):
    phi0 = evecs.T[:, src_index]
    phases = np.exp(-1j * (J*evals[:,None]) * tgrid[None,:])
    return evecs @ (phases * phi0[:,None])

def chain_H(L, J=1.0):
    H = np.zeros((L,L), float)
    for i in range(L-1):
        H[i,i+1] = H[i+1,i] = -J
    return H

def simulate_radar_chain(L=801, J=1.0, u=0.6, xA=None, xB0=None,
                         eps_out=0.02, tau_emit=8.0, tmax=300.0, T=6001,
                         return_mode="ballistic"):
    if xA is None: xA = L//4
    if xB0 is None: xB0 = 3*L//4
    H = chain_H(L,J); evals, evecs = eigh_H(H)
    tgrid = np.linspace(0.0, tmax, int(T)); dt=tgrid[1]-tgrid[0]
    vstar = 2.0*J
    psiA = psi_from_source(evals, evecs, xA, tgrid, J=J); ampA=np.abs(psiA)
    speed = u*vstar
    xB_t = np.clip(np.round(xB0 + speed*tgrid).astype(int), 0, L-1)
    def first_arrival_to_B(s_emit):
        start = int(np.searchsorted(tgrid, s_emit, 'left'))
        for k in range(start, len(tgrid)):
            tau = tgrid[k]-s_emit; tau_idx=int(round(tau/dt))
            if 1 <= tau_idx < len(tgrid):
                j = int(xB_t[k])
                if ampA[j, tau_idx] >= eps_out:
                    return float(tgrid[k]), j
        return None, None
    def t_echo_ballistic(t_out, j_out):
        return float(t_out + abs(j_out - xA) / vstar)
    outs=[]
    for s in (0.0, float(tau_emit)):
        t_out, j_out = first_arrival_to_B(s)
        if t_out is None: outs.append({"s_emit":s,"t_out":None,"t_echo":None}); continue
        outs.append({"s_emit":s,"t_out":t_out,"t_echo":t_echo_ballistic(t_out,j_out),"j_out":int(j_out)})
    valid=[r for r in outs if r["t_echo"] is not None]
    if len(valid)==2:
        ratio = (valid[1]["t_echo"] - valid[0]["t_echo"]) / (valid[1]["s_emit"] - valid[0]["s_emit"])
        k_est = float(math.sqrt(ratio)); gamma_est = float(0.5*(k_est + 1.0/k_est))
        u_est = float((k_est*k_est - 1.0)/(k_est*k_est + 1.0))
    else:
        k_est=gamma_est=u_est=float('nan')
    return {"L":L,"u_in":u,"xA":xA,"xB0":xB0,"k_est":k_est,"gamma_est":gamma_est,"u_est":u_est,"events":outs}

def main():
    rows=[]
    for u in (0.2,0.4,0.6):
        res = simulate_radar_chain(L=801, u=u, tau_emit=8.0, tmax=360.0, T=7201, eps_out=0.01)
        k=res["k_est"]; g=res["gamma_est"]
        k_SR = math.sqrt((1.0+u)/(1.0-u)); g_SR=1.0/math.sqrt(1.0-u*u)
        rows.append({
            "u_in":u,"k_est":k,"k_SR":k_SR,"rel_err_k":(k/k_SR-1.0) if (k_SR>0 and np.isfinite(k)) else float('nan'),
            "gamma_est":g,"gamma_SR":g_SR,"rel_err_gamma":(g/g_SR-1.0) if (g_SR>0 and np.isfinite(g)) else float('nan'),
            "u_est_from_k":res["u_est"],"abs_err_u":(res["u_est"]-u) if np.isfinite(res["u_est"]) else float('nan')
        })
    df=pd.DataFrame(rows); df.to_csv("E3_v3_chain_summary.csv", index=False); print(df.to_string(index=False))

if __name__=="__main__":
    main()
