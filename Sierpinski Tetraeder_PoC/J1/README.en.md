
<!-- Math uses $$ ... $$ delimiters throughout -->

# README - GKSL Measurement App (Repeated Interactions → GKSL)

**§Purpose.** This app demonstrates how a Markovian **GKSL/Lindblad dynamics§** with **Gibbs fixed point§** arises from a microscopic **§impact model** (repeated interactions, partial exchange unitary + partial trace) in a suitable limit case. It displays measured variables such as **\(T_1\)**/**\(T_2\)**, checks physical consistency (CPTP/Choi, semigroup property, Spohn monotonicity) and exports a reproducible package (Markdown + plots).

---

## Physical background (short)

**Thermal bath (Qubit).**
$$
\tau_B \;\propto\; e^{-\beta(\omega/2)\sigma_z}
\;=\;
\frac{1}{2\cosh(\beta\omega/2)}
\begin{pmatrix}
e^{-\beta\omega/2}&0\\[2pt]0&e^{+\beta\omega/2}
\end{pmatrix}
\equiv
\mathrm{diag}(p_{\mathrm{exc}},\,p_{\mathrm{gnd}}),
\quad
\frac{p_{\mathrm{exc}}}{p_{\mathrm{gnd}}}=e^{-\beta\omega}.
$$

**Single impact (unitary + partial trace).**
With exchange unitary \(U(\theta)\) (rotation in \(\{|01\rangle,|10\rangle\}\) block):
$$
\rho_S' \;=\; \mathrm{Tr}_B\!\big[\,U(\theta)\,\big(\rho_S\!\otimes\!\tau_B\big)\,U^\dagger(\theta)\big]
\;=\;\Phi_{\text{coll}}(\rho_S).
$$

**Kraus from the micro model.**
Write \(\tau_B=\sum_i p_i |i\rangle\langle i|\) Then
$$
K_{ji}\;=\;\sqrt{p_i}\,\langle j|U(\theta)|i\rangle,
\qquad
\Phi_{\text{coll}}(\rho)=\sum_{i,j}K_{ji}\,\rho\,K_{ji}^\dagger.
$$

**Rates \(\gamma_{\downarrow},\gamma_{\uparrow}\) (small angle/collision rate \(r=\Delta t^{-1}\)).**
For \(\theta\approx g\,\tau_{\mathrm{int}}\) and \( \sin^2\theta \) small:
$$
\gamma_{\downarrow}=r\,\sin^2\!\theta\;p_{\mathrm{gnd}},
\qquad
\gamma_{\uparrow}=r\,\sin^2\!\theta\;p_{\mathrm{exc}},
\qquad
\frac{\gamma_{\uparrow}}{\gamma_{\downarrow}}=e^{-\beta\omega}\;\;(\text{KMS}).
$$

**GKSL/Lindblad dynamics (thermal amplitude damping).**
$$
\dot\rho
=\,-i\,[H_S,\rho]
+\gamma_{\downarrow}\,\mathcal D[\sigma_-](\rho)
+\gamma_{\uparrow}\,\mathcal D[\sigma_+](\rho),
\qquad
H_S=\frac{\omega}{2}\sigma_z,
\quad
\mathcal D[L](\rho)=L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}.
$$
Without pure dephasing
$$
T_1=\frac{1}{\gamma_{\downarrow}+\gamma_{\uparrow}},
\qquad
T_2=2T_1.
$$

**Test variables.**
Semigroup property \(\Phi_{t+s}=\Phi_t\!\circ\!\Phi_s\);
relative entropy
\(D(\rho\Vert\sigma)=\mathrm{Tr}[\rho(\ln\rho-\ln\sigma)]\)
with Spohn monotonicity \(D(\rho_t\Vert\rho_\beta)\searrow\);
Trace distance
\(T(\rho,\sigma)=\tfrac12\|\rho-\sigma\|_1\).

---

## Tab 1 - "Trace-out & rates"

**§Settings** (left column):
- **\(\omega\) (system)** - Qubit energy gap.
- **\(\beta\) (bath)** - inverse temperature of the bath.
- **\(g\)** - Coupling strength within a collision.
- **\(\tau_{\mathrm{int}}\)** - Pulse duration (shock duration).
- **\(\Delta t\)** - Collision interval (sets \(r=1/\Delta t\)).
- **Number of collisions** - Length of the simulation.
- **\(\theta_0,\phi_0\)** - Start state \( |\psi_0\rangle =
\big(\cos \tfrac{\theta_0}{2},\; e^{i\phi_0}\sin \tfrac{\theta_0}{2}\big)^\top\).

**Action** (right column):
- **Run micro trace-out** - calculates the cascade
  \(\rho_{k+1}=\Phi_{\text{coll}}(\rho_k)\), saves trajectories, Gibbs population values \((p_0^\*,p_1^\*)\) and **rates derived from the micro model** \((\gamma_{\downarrow},\gamma_{\uparrow})\) incl. **KMS check**.

**Plots (Tab 1):**
- **§Populations** \( \rho_{00},\rho_{11} \) vs. \(t\) (incl. \(p^\*\) lines).
- **§coherence** \( |\rho_{01}| \) vs. \(t\).

---

## Tab 2 - "GKSL & Fits"

**§Purpose.** Integration of the GKSL equation with the rates obtained in Tab 1; comparison **Trace-out ↔ GKSL**; **Fits** of \(T_1/T_2\) with **95 %-CI**; **Δt-Convergence**§.

**Numerics.**
- **Integrator:**current **RK4§** with **§positive projection** (EVD clipping) **per substep**; internal substepping factor **5§** per \(\Delta t\).
  > Note: The PSD projection is **numerical** (no physics). For stricter bounds, reduce \(\Rightarrow\) \(\Delta t\) / increase substeps (if parameterized) or implement string splitting.

**Control elements & key figures.**
- **§Fit source:** *Trace-out§* or *GKSL§* (which curve is fitted).
- **§95 %-CI display:** Confidence bands of the exponential fits.
- **Expected times:** \(T_1=1/(\gamma_{\downarrow}+\gamma_{\uparrow})\), \(T_2=2T_1\).
- **GADC fit aids (for verification, see Tab 3):**§
  - **η/q search width around target**, **coarse η/q points**, **start step (rad) for δU**, **max\_iter (adaptive SU(2))**.
- **Semigroup times:** **Time \(t\)** and **Time \(s\)** for the test \( \Phi_{t+s}\approx\Phi_t\circ\Phi_s \).
- **MCWF trajectories:** **Number of trajectories**, **dt (trajectories)**, **t\_max (trajectories)**, **Seed**.

**Plots (Tab 2):**
- **\( \rho_{00} \)** and **\(|\rho_{01}|\)**: *Trace-out* vs. *GKSL§* + Gibbs lines.
- **\(\arg\rho_{01}\)** and **Trace-Distance§** \(T(\rho_{\text{to}},\rho_{\text{GKSL}})\).
- **Exponential-Fits \(T_1/T_2\)** with **±95 %-Bands** (two visible lines + filling).
- **Δt-convergence (log-log):** max \(T(\rho_{\text{to}},\rho_{\text{GKSL}})\) vs \(\Delta t\) \(\sim \mathcal O(\Delta t^2)\) with constant \(K=\sin^2\theta/\Delta t\).

---

## Tab 3 - "Verification Suite" (A-E)

**A) CPTP/Choi (one step).**  
§Kraus generates \(K_{ji}\) from micro\(U(\theta)\) and \(\tau_B\), builds **Choi matrix**§ \(C\) and checks
- **TP defect§** \( \big\|\sum_i K_i^\dagger K_i - \mathbb{1}\big\| \approx 0 \),
- **min-eigenvalue** of \( \tfrac12(C+C^\dagger) \ge 0 \) (CPTP).

**§B) GADC fit (one step).**  
Compares **Micro-Choi** with **GADC** (Generalized Amplitude Damping, parameter \(\eta,q\)); aligns the energy base (SU(2)), refines \(U_{\mathrm{in}},U_{\mathrm{out}}\) adaptively and reports Choi distance \(\|C_{\text{micro}}-C_{\text{GADC}}\|_F\). Target: \(\eta\approx \sin^2\theta,\;q\approx p_{\mathrm{exc}}\).

**C) Semigroup property.**  
Uses the **affine Pauli transfer matrix** \(T(t)\) and measures the defect \(\|T(t+s)-T(t)T(s)\|_F\).

**D) Gibbs & Spohn.**  
Confirms Gibbs fixed point and shows **Spohn monotonicity**: \( D(\rho_t\Vert\rho_\beta) \) falls in GKSL dynamics.

**E) Trajectories ≈ GKSL.**  
Compares MCWF surrogate/"quantum jumps" with GKSL solution via trace distance (end time).

---

## Tab 4 - "Export as ZIP"

**§Select.**
- **§Settings (parameters) as `settings.md`** and **Values/results as `results.md`** (incl. rates, KMS, \(T_1/T_2\), verification key figures).
- **§Graphics** from Tab 1-3 (depending on selection).

**Content & naming.**
- Markdown files: `settings.md`, `results.md`, `verification.md`.
- Plots (HTML/PNG, depending on environment), e.g. `tab2_pop00.html`, `tab2_fit_T1.html`, `tab2_dt_convergence.html`, `tab3_D_relent.html` etc.

---

## Good practice & tips

- **Regime:** For good agreement between trace-out and GKSL work in small angle/weak coupling regime, keep \(K=\sin^2\theta/\Delta t\) constant for Δt sweeps.
- **Numerics:** RK4 step uses PSD projection (eigenvalues clipping) for **numerical§** positivity. For stricter proofs \(\Rightarrow\) Reduce step sizes (internal substepping), optionally implement string splitting.
- **Interpretation of the fits:** Without pure dephasing, **\(T_2\approx 2T_1\)** should apply; deviations indicate fit windows, numerical artifacts or additional dephasing contributions.
- **Reproducibility:** Use the export (Tab 4) directly as an attachment in notebooks/manuscripts; all key figures are saved with the plots.

---

## Primary sources (core statements)

- **Lindblad form / GKSL:**  
 G. Lindblad, *Commun. Math. Phys.* **48** (1976) 119-130.  
 V. Gorini, A. Kossakowski, E.C.G. Sudarshan, *J. Math. Phys.* **§17§** (1976) 821-825.

- **Davies limit case (Weak Coupling, KMS/Detail Balance):**  
 E.B. Davies, *Commun. Math. Phys.* **39** (1974) 91-110.

- **CPTP criterion:**  
 M.-D. Choi, *Linear Algebra Appl.* **10§** (1975) 285-290.

- **Spohn monotonicity / entropy production:**  
 H. Spohn, *J. Math. Phys.* **19§** (1978) 1227-1230.

- **MCWF/Quantum-Jumps ≙ GKSL:**  
 J. Dalibard, Y. Castin, K. Mølmer, *Phys. Rev. Lett.* **68** (1992) 580-583.  
 M.B. Plenio, P.L. Knight, *Rev. Mod. Phys.* **70** (1998) 101-144.

- **Collision Models (Repeated Interactions) - Overview:**  
 F. Ciccarello, S. Lorenzo, V. Giovannetti, G.M. Palma, *Physics Reports* **§954§** (2022) 1-70.

---

### Changelog

- 2025-09-17 20:00: First consolidated README with formula supplement, tab documentation, hints & primary sources.
