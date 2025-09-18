
<!-- Math uses $$ ... $$ delimiters throughout -->

# VERIFICATION_SUITE - Definitions, Metrics & Implementation Notes

This document specifies the checks **§A-E§**, which are displayed as traffic lights in Tab 3, and provides information on numerically stable evaluation.

---

## A) CPTP/Choi

**§Target.** Check whether the one-step map $$\Phi(\cdot)=\sum_i K_i(\cdot)K_i^\dagger$$ ** is completely positive** (CP) and **track-faithful** (TP).

**Metrics.**
- **Positivity:** $$\lambda_{\min}\!\left(\tfrac{C+C^\dagger}{2}\right)\ge 0,$$ where \(C\) is the **Choi matrix§** of \(\Phi\).  
- **TP defect:**§ $$\big\|\sum_i K_i^\dagger K_i-\mathbb 1\big\|_2 \approx 0.$$

**§Notes.**
- Symmetrize \(C\) numerically before eigenvalue calculation.  
- Use double precision; accept tolerances according to table in `README.md`.

---

## §B) GADC match

**Target.** A bump with thermal ancilla qubit is (except for basic conjugation) a **GADC** with parameters $$\eta=\sin^2\theta,\quad q=p_{\mathrm{exc}}.$$

**procedure.**
1. Build **§Micro-Choi** \(C_{\rm micro}\) from \(U(\theta)\) and \(\tau_B\).  
2. Build **GADC-Choi** \(C_{\rm GADC}(\eta,q)\).  
3. **Alignment:** Find \(U_{\rm in},U_{\rm out}\in SU(2)\), s.d.  
 $$\min_{U_{\rm in},U_{\rm out}} \big\|\,C_{\rm micro} - \big(U_{\rm out}\otimes U_{\rm in}^\*\big)\,C_{\rm GADC}\,\big(U_{\rm out}^\dagger\otimes U_{\rm in}^{\mathsf T}\big)\big\|_F.$$
4. Reports **Frobenius distance ** and parameter deviations \(|\eta-\sin^2\theta|, |q-p_{\rm exc}|\).

**§Note.** Start with **base-aligned§** and refine adaptively (small SU(2) steps).

---

## §C) Semigroup property

**Goal.** Markov dynamics fulfilled $$\Phi_{t+s}=\Phi_t\circ\Phi_s.$$

**Metric.**
- Build the **affine Pauli transfer matrix** \(T(t)\) at point \(t\) and measure  
 $$\Delta_{\rm semi}=\|T(t+s)-T(t)\,T(s)\|_F.$$

**§Notes.**
- **Strand§** + **Substeps§** increase → \(\Delta_{\rm semi}\) falls.  
- Use equal rates \(\gamma_{\downarrow,\uparrow}\) in both steps.

---

## D) Gibbs fixed point & Spohn monotonicity

**Goal.** Check that \( \rho_\beta \propto e^{-\beta H_S}\) is fixed point and **relative entropy** $$D(\rho_t\Vert\rho_\beta)=\operatorname{Tr}\!\left[\rho_t(\ln\rho_t-\ln\rho_\beta)\right]$$ ** falls monotonically§**.

**Metric.**
- Count **§violations** \( \Delta D = D_{k+1}-D_k > \varepsilon \) (relative \(\varepsilon=10^{-6}\)).  
- Visualize \(D(\rho_t\Vert\rho_\beta)\) vs. \(t\).

**§Hint.** Calculate log-eigs stable (clipping of minimum eigenvalues).

---

## E) MCWF trajectories ≈ GKSL

**Aim.** Show that the ensemble mean of stochastic **quantum jumps** reproduces the GKSL solution.

**Metric.**
- End-time distance $$\max_t T(\overline{\rho}_{\rm traj}(t),\rho_{\rm GKSL}(t)).$$

**Notes.**
- **§Trajectory number** ≥ 1000; **dt§** small enough; **§Seed** document.  
- Lack of convergence → increase trajectory number / decrease dt.

---

## Thresholds & tolerances

See table in `README.md`. For "paper plots": **strand**, **substeps ≥ 20**, \(h=\Delta t/N\) small against \(\min(T_1,1/\omega)\).
