
"""
E3_v2 – Radar-/Echo-Protokoll mit wählbarem Rückweg:
- return_mode="ctqw": Rücklauf via CTQW-Amplitude (wie v1; empfindlich gegenüber Schwellen/Interferenzen).
- return_mode="ballistic": Rücklaufzeit als Distanz/v_* (robust, LR/ballistische Front).

Standard: 1D-Kette (J=1 -> v_*=2), u in {0.2, 0.4, 0.6}.
"""
import math, json
import numpy as np
import pandas as pd
from collections import deque

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

def simulate_radar_chain(L=301, J=1.0, u=0.4, eps_out=0.02, eps_back=1e-4,
                         tau_emit=6.0, tmax=240.0, T=4801, return_mode="ballistic"):
    H = chain_H(L,J)
    evals, evecs = eigh_H(H)
    tgrid = np.linspace(0.0, tmax, int(T))
    dt = tgrid[1] - tgrid[0]
    xA, xB0 = L//3, 2*L//3
    vstar = 2.0*J
    # outgoing amplitude from A
    psiA = psi_from_source(evals, evecs, xA, tgrid, J=J); ampA=np.abs(psiA)
    # worldline
    speed = u*vstar
    xB_t = np.clip(np.round(xB0 + speed*tgrid).astype(int), 0, L-1)

    def first_arrival_to_B(s_emit: float):
        start_idx = int(np.searchsorted(tgrid, s_emit, side='left'))
        for k in range(start_idx, len(tgrid)):
            tau = tgrid[k] - s_emit
            tau_idx = int(round(tau/dt))
            if tau_idx < 1 or tau_idx >= len(tgrid):
                continue
            j = int(xB_t[k])
            if ampA[j, tau_idx] >= eps_out:  # outgoing threshold
                return float(tgrid[k]), j
        return None, None

    def first_return_to_A_ctqw(t_out: float, j_out: int):
        psiB = psi_from_source(evals, evecs, j_out, tgrid, J=J)
        ampB_A = np.abs(psiB[xA,:])
        start_idx = int(np.searchsorted(tgrid, t_out, side='left'))
        for k in range(start_idx, len(tgrid)):
            tau = tgrid[k] - t_out
            tau_idx = int(round(tau/dt))
            if tau_idx < 1 or tau_idx >= len(tgrid):
                continue
            if ampB_A[tau_idx] >= eps_back:
                return float(tgrid[k])
        return None

    def first_return_to_A_ballistic(t_out: float, j_out: int):
        dist = abs(j_out - xA)
        return float(t_out + dist / vstar)

    outs=[]
    for s in (0.0, float(tau_emit)):
        t_out, j_out = first_arrival_to_B(s)
        if t_out is None:
            outs.append({"s_emit": s, "t_out": None, "t_echo": None})
            continue
        if return_mode == "ctqw":
            t_echo = first_return_to_A_ctqw(t_out, j_out)
        else:
            t_echo = first_return_to_A_ballistic(t_out, j_out)
        outs.append({"s_emit": s, "t_out": t_out, "t_echo": t_echo, "j_out": int(j_out) if j_out is not None else None})

    valid=[r for r in outs if r["t_echo"] is not None]
    if len(valid)==2:
        ratio = (valid[1]["t_echo"] - valid[0]["t_echo"]) / (valid[1]["s_emit"] - valid[0]["s_emit"])
        k_est = float(math.sqrt(ratio))
        gamma_est = float(0.5*(k_est + 1.0/k_est))
        u_est = float((k_est*k_est - 1.0)/(k_est*k_est + 1.0))
    else:
        ratio=k_est=gamma_est=u_est=float('nan')

    return {
        "model":"chain","L":int(L),"J":float(J),"vstar":float(vstar),
        "u_in":float(u),"eps_out":float(eps_out),"eps_back":float(eps_back),
        "tau_emit":float(tau_emit),"ratio_k2":ratio,"k_est":k_est,"gamma_est":gamma_est,"u_est":u_est,
        "events":outs,"return_mode":return_mode
    }

def main():
    rows=[]
    results=[]
    for u in (0.2, 0.4, 0.6):
        r = simulate_radar_chain(u=u, return_mode="ballistic", eps_out=0.02, eps_back=1e-4, tau_emit=6.0)
        results.append(r)
        k=r["k_est"]; g=r["gamma_est"]
        k_SR = math.sqrt((1.0+u)/(1.0-u)); g_SR=1.0/math.sqrt(1.0-u*u)
        rows.append({
            "u_in":u,"k_est":k,"k_SR":k_SR,"rel_err_k":(k/k_SR-1.0),
            "gamma_est":g,"gamma_SR":g_SR,"rel_err_gamma":(g/g_SR-1.0),
            "u_est_from_k":r["u_est"],"abs_err_u":r["u_est"]-u
        })
    pd.DataFrame(results).to_json("E3_v2_chain_results.json", orient="records", indent=2)
    df = pd.DataFrame(rows)
    df.to_csv("E3_v2_chain_summary.csv", index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
