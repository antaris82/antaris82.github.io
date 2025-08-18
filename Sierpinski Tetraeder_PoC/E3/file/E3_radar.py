
"""
E3 – Radar-/Echo-Protokoll (Bondi k-Kalkül) und Lorentz-Kinematik aus v_* (ohne Maxwell).
- Teil A: 1D-Kette (CTQW / Tight-Binding), Echo-Messung -> k, gamma, u
- Teil B: Analytischer Radar-Check (Bondi) für Kette und ST-Pfad (mit invariantem c=v_*)
- (Optional) Teil C: ST-CTQW-Echo (PoC-Funktionen vorhanden, standardmäßig nicht ausgeführt)

Hinweise:
- Kette: v_* = 2J (Gruppengeschwindigkeit), wir wählen J=1 und L=301.
- Echo-Erkennung: schwacher Rücklauf braucht kleinere eps_back als eps_out.
- Quellen (Theorie): Pal (Relativität ohne Lichtpostulat), Bondi k-Kalkül, Nachtergaele–Ogata–Sims (Lieb–Robinson).
"""

import math, json
import numpy as np
import pandas as pd

# -------------------- Utilities --------------------
def eigh_H(H):
    """Eigenzerlegung eines symmetrischen H."""
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

def psi_from_source(evals, evecs, src_index, tgrid, J=1.0):
    """Zeitentwicklung |psi(t)> = exp(-i J H t) |src> im Eigenbasis-Frame."""
    phi0 = evecs.T[:, src_index]
    phases = np.exp(-1j * (J*evals[:,None]) * tgrid[None,:])
    return evecs @ (phases * phi0[:,None])

# -------------------- 1D-Kette (CTQW) --------------------
def chain_H(L, J=1.0):
    H = np.zeros((L,L), float)
    for i in range(L-1):
        H[i,i+1] = H[i+1,i] = -J
    return H

def simulate_radar_chain_single(L=301, J=1.0, u=0.4, eps_out=0.02, eps_back=1e-4,
                                tau_emit=6.0, tmax=240.0, T=4801):
    """
    Echo-Protokoll auf der 1D-Kette:
    - A sendet bei t=0 und t=tau_emit eine Front (Quelle = Delta am xA)
    - B bewegt sich mit u * v_* (v_* = 2J) nach rechts (Spiegel). Ankunft -> sofortige Reflexion
    - A misst Echozeiten; aus Ratio der Echo-Abstände folgt k^2 = (ΔT_echo)/(ΔT_emit).
    """
    H = chain_H(L,J)
    evals, evecs = eigh_H(H)
    tgrid = np.linspace(0.0, tmax, int(T))
    dt = tgrid[1] - tgrid[0]
    xA, xB0 = L//3, 2*L//3
    vstar = 2.0*J
    # Vorwärtsamplitude von A
    psiA = psi_from_source(evals, evecs, xA, tgrid, J=J)
    ampA = np.abs(psiA)
    # B-Weltlinie (diskret)
    speed = u * vstar
    xB_t = np.clip(np.round(xB0 + speed*tgrid).astype(int), 0, L-1)

    def first_arrival_to_B(s_emit: float):
        start_idx = int(np.searchsorted(tgrid, s_emit, side='left'))
        for k in range(start_idx, len(tgrid)):
            tau = tgrid[k] - s_emit
            tau_idx = int(round(tau/dt))
            if tau_idx < 1 or tau_idx >= len(tgrid):  # verbiete tau=0 (Trivialecho)
                continue
            j = int(xB_t[k])
            if ampA[j, tau_idx] >= eps_out:
                return float(tgrid[k]), j
        return None, None

    def first_return_to_A(t_out: float, j_out: int):
        # Von j_out aus propagieren und Amp am A beobachten
        psiB = psi_from_source(evals, evecs, j_out, tgrid, J=J)
        ampB_A = np.abs(psiB[xA, :])
        start_idx = int(np.searchsorted(tgrid, t_out, side='left'))
        for k in range(start_idx, len(tgrid)):
            tau = tgrid[k] - t_out
            tau_idx = int(round(tau/dt))
            if tau_idx < 1 or tau_idx >= len(tgrid):
                continue
            if ampB_A[tau_idx] >= eps_back:
                return float(tgrid[k])
        return None

    outs = []
    for s in (0.0, float(tau_emit)):
        t_out, j_out = first_arrival_to_B(s)
        if t_out is None:
            outs.append({"s_emit": s, "t_out": None, "t_echo": None})
            continue
        t_echo = first_return_to_A(t_out, j_out)
        outs.append({"s_emit": s, "t_out": t_out, "t_echo": t_echo, "j_out": int(j_out) if j_out is not None else None})

    valid = [r for r in outs if r["t_echo"] is not None]
    if len(valid) == 2:
        ratio = (valid[1]["t_echo"] - valid[0]["t_echo"]) / (valid[1]["s_emit"] - valid[0]["s_emit"])
        k_est = float(math.sqrt(ratio))
        gamma_est = float(0.5 * (k_est + 1.0/k_est))
        u_est = float((k_est*k_est - 1.0) / (k_est*k_est + 1.0))
    else:
        ratio = k_est = gamma_est = u_est = float('nan')

    return {
        "model": "chain",
        "L": int(L),
        "J": float(J),
        "vstar": float(vstar),
        "u_in": float(u),
        "eps_out": float(eps_out),
        "eps_back": float(eps_back),
        "tau_emit": float(tau_emit),
        "ratio_k2": ratio,
        "k_est": k_est,
        "gamma_est": gamma_est,
        "u_est": u_est,
        "events": outs
    }

# -------------------- Analytischer Bondi-Radar (allgemein) --------------------
def radar_echo_times(s0: float, c: float, u: float, tau_emit: float, n: int = 2):
    """
    Bondi-Radar (analytisch): A sendet n Pulse mit Abstand tau_emit; B entfernt sich mit v=u*c.
    Echozeiten T_i bei A: geschl. Formeln -> Ratio = k^2 = (1+u)/(1-u).
    """
    s_em = [i * tau_emit for i in range(n)]
    T_echo = []
    for s in s_em:
        t_out = (s0 + c*s) / (c*(1.0 - u))      # Meet-Zeit (hin)
        sB = s0 + u*c*t_out                     # Position von B beim Treffen
        T = t_out + sB / c                      # Rückkehrzeit
        T_echo.append(T)
    return s_em, T_echo

def run_analytic_csv(out_csv="E3_radar_analytic.csv"):
    rows = []
    for kind, c, s0 in [("chain", 2.0, 100.0), ("ST_L6_path", 1.1, 20.0)]:
        for u in (0.2, 0.4, 0.6):
            s_em, T = radar_echo_times(s0, c, u, tau_emit=6.0, n=2)
            ratio = (T[1]-T[0])/(s_em[1]-s_em[0])
            k_est = math.sqrt(ratio)
            gamma_est = 0.5*(k_est + 1.0/k_est)
            rows.append({
                "kind": kind, "u_in": u, "c_used": c, "ratio_k2_est": ratio,
                "k_est": k_est, "k_SR": math.sqrt((1.0+u)/(1.0-u)),
                "gamma_est": gamma_est, "gamma_SR": 1.0/math.sqrt(1.0-u*u)
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# -------------------- Main: Experimente ausführen --------------------
def main():
    # Teil A: 1D-Kette – Echo-Messung für u in {0.2, 0.4, 0.6}
    chain_results = []
    for u in (0.2, 0.4, 0.6):
        res = simulate_radar_chain_single(L=301, J=1.0, u=u,
                                          eps_out=0.02, eps_back=1e-4,
                                          tau_emit=6.0, tmax=240.0, T=4801)
        chain_results.append(res)

    # Artefakte sichern
    Path("E3_radar_chain_results.json").write_text(json.dumps(chain_results, indent=2))

    # Zusammenfassung (inkl. Soll-/Ist-Werte)
    rows = []
    for r in chain_results:
        u = r["u_in"]
        k = r["k_est"]; gamma = r["gamma_est"]; u_est = r["u_est"]
        k_SR = math.sqrt((1.0+u)/(1.0-u))
        gamma_SR = 1.0/math.sqrt(1.0-u*u)
        rows.append({
            "kind": "chain",
            "u_in": u,
            "k_est": k,
            "k_SR": k_SR,
            "rel_err_k": (k/k_SR - 1.0) if (k_SR>0 and math.isfinite(k)) else float('nan'),
            "gamma_est": gamma,
            "gamma_SR": gamma_SR,
            "rel_err_gamma": (gamma/gamma_SR - 1.0) if (gamma_SR>0 and math.isfinite(gamma)) else float('nan'),
            "u_est_from_k": u_est,
            "abs_err_u": (u_est - u) if math.isfinite(u_est) else float('nan')
        })
    df_chain = pd.DataFrame(rows)
    df_chain.to_csv("E3_radar_chain_summary.csv", index=False)

    # Teil B: Analytischer Radar (Bondi) – Kette & ST-Pfad
    df_analytic = run_analytic_csv(out_csv="E3_radar_analytic.csv")

    print("== E3: Kette (CTQW) – Echo-Messung ==")
    print(df_chain.to_string(index=False))
    print("\n== E3: Analytischer Radar (Bondi) ==")
    print(df_analytic.to_string(index=False))

if __name__ == "__main__":
    from pathlib import Path
    main()
