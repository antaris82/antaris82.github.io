
<!-- Math uses $$ ... $$ delimiters throughout -->

# README - GKSL Measurement App (Repeated Interactions → GKSL)

**§Short version.** The app shows **Micro → Macro§** in one image: *Shock model* (Repeated Interactions, partial exchange unitary + Trace-out) is **§directly§** mirrored against the **GKSL/Lindblad dynamics**. Tab 3 provides a **verification suite** (CPTP/Choi, GADC match, semigroup, Spohn monotonicity, MCWF≈GKSL). Tab 4 exports everything as Markdown + plots.

---

## 1. Physical idea at a glance

- **Bath as collisions.** Fresh thermal ancillas hit the system qubit one after the other. One collision: $$\rho'=\operatorname{Tr}_B\!\big[U(\theta)\,(\rho\otimes\tau_B)\,U^\dagger(\theta)\big].$$
- **A step is a channel.** For qubits, a collision (except for base conjugation) corresponds to **GADC** with $$\eta=\sin^2\!\theta,\qquad q=p_{\mathrm{exc}}=\frac{1}{1+e^{\beta\omega}}.$$
- **Continuous limit.** For small steps and constant $$K=\frac{\sin^2\!\theta}{\Delta t}$$ the **GKSL master equation** with $$\dot\rho=-i[H_S,\rho]+\gamma_\downarrow\mathcal D[\sigma_-](\rho)+\gamma_\uparrow\mathcal D[\sigma_+](\rho),$$ $$\gamma_\downarrow=K\,p_{\mathrm{gnd}},\qquad \gamma_\uparrow=K\,p_{\mathrm{exc}},\qquad \frac{\gamma_\uparrow}{\gamma_\downarrow}=e^{-\beta\omega}.$$
- **Target state.**Fixed point is the **Gibbs state** $$\rho_\beta=\frac{e^{-\beta H_S}}{\operatorname{Tr}(e^{-\beta H_S})},\qquad H_S=\frac{\omega}{2}\sigma_z.$$
- **Relaxation times.**Without pure dephasing $$T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\qquad T_2=2T_1.$$

---

## §2. tabs & functions

### Tab 1 - Trace-out & Rates
- Simulates **§collisions**, plots populations/coherences, calculates ** rates** \( \gamma_{\downarrow,\uparrow} \) and checks **KMS**.
- Core controller: \( \omega,\beta,g,\tau_{\mathrm{int}},\Delta t,\theta_0,\phi_0 \).

### Tab 2 - GKSL & Fits
- Integrates GKSL and compares **Trace-out vs. GKSL** (Pop., \(|\rho_{01}|\), Phase, Trace-Distance).
- **Integrator selection:** *Strand (CPTP, 2nd order)* | *RK4 (+/- PSD)*.
- **Substeps per \(\Delta t\)**: N (recommended: 10-20).
- **Fits** for \(T_1/T_2\) (±95 % bands) + **Δt-convergence** \(\sim\mathcal O(\Delta t^2)\) at constant \(K\).

### Tab 3 - Verification Suite (A-E)
- **§A§** CPTP/Choi, **§B§** GADC-Match, **§C§** Semigroup, **§D§** Gibbs/Spohn, **§E§** MCWF≈GKSL.

### Tab 4 - Export
- Generates `settings.md`, `results.md`, `verification.md` + selected plots (HTML/PNG).

---

## 3. verification suite - OK thresholds (recommended)

| Check | Metric | Green | Yellow | Red | Note |
|---|---|---:|---:|---:|---|
| **§A1§** CPTP | $$\lambda_{\min}\big(\tfrac{C+C^\dagger}{2}\big)$$ | $$\ge -10^{-12}$$ | $$[-10^{-10},-10^{-12})$$ | $$§§X20§§10^{-9}$$ | By rounding/steps. |
| **§B§** GADC distance | $$\|C_{\text{micro}}-C_{\text{GADC}}\|_F$$ | $$\le 10^{-3}$$ | $$\le 5\cdot 10^{-3}$$ | $$>5\cdot10^{-3}$$ | Additionally: \(|\eta-\sin^2\theta|\le 10^{-2},\ |q-p_{\rm exc}|\le 10^{-2}\). |
| **§C§** Semigroup | $$\|T(t+s)-T(t)T(s)\|_F$$ | $$\le 10^{-3}$$ | $$\le 5\cdot10^{-3}$$ | $$>5\cdot10^{-3}$$ | \(T(\cdot)\): affine Pauli transfer matrix; coincides with substeps \(N\). |
| **D§** Spohn | violations $$\#\{\Delta D> \varepsilon\}$$ | $$0$$ | $$\le 2$$ | $$>2$$ | \(\varepsilon=10^{-6}\) relative; \(D(\rho_t\Vert\rho_\beta)\) monotonic↓. |
| **E§** MCWF≈GKSL | $$\max_t T(\rho_{\rm traj},\rho_{\rm GKSL})$$ | $$\le 5\cdot10^{-3}$$ | $$\le 2\cdot10^{-2}$$ | $$>2\cdot10^{-2}$$ | trajectories ≥ 1000, fine \(dt_{\rm traj}\). |

> **Rules of thumb:** Choose **Strand§** + **N ≥ 10**, keep $$K=\sin^2\theta/\Delta t$$ for Δt sweeps **§constant§** and use stable logistics for \(p_{\rm exc}=1/(1+e^{\beta\omega})\).

---

## 4. Quickstart

1. **§Tab 1**: Defaults → *Run micro trace-out* → Check rates/KMS.  
2. **Tab 2**: Integrator **§Strand§**, **Substeps§** = 10-20 → Compare GKSL vs. Compare trace-out; **\(T_1,T_2\)** fit.  
§3. **Tab 3**: View A-E (green).  
4. **Tab 4§**: Create export (ZIP).

---

## 5. primary sources (short list)

- Lindblad/GKSL (1976), GKS (1976); Davies (1974, Weak Coupling/KMS); Choi (1975, CPTP); Spohn (1978, Entropy Production); Dalibard-Castin-Mølmer (1992) & Plenio-Knight (1998, MCWF); Ciccarello et al. (2022, collision models).

---

**Status:** 2025-09-18 17:29
