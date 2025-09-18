
<!-- Math uses $$ ... $$ delimiters throughout -->

# VERIFICATION_SUITE — Definitionen, Metriken & Implementation‑Hinweise

Dieses Dokument präzisiert die Checks **A–E**, die in Tab 3 als Ampeln angezeigt werden, und gibt Hinweise zur numerisch stabilen Auswertung.

---

## A) CPTP/Choi

**Ziel.** Prüfe, ob die Ein‑Schritt‑Map $$\Phi(\cdot)=\sum_i K_i(\cdot)K_i^\dagger$$ **komplett positiv** (CP) und **spurtreu** (TP) ist.

**Metriken.**
- **Positivität:** $$\lambda_{\min}\!\left(\tfrac{C+C^\dagger}{2}\right)\ge 0,$$ dabei ist \(C\) die **Choi‑Matrix** von \(\Phi\).  
- **TP‑Defekt:** $$\big\|\sum_i K_i^\dagger K_i-\mathbb 1\big\|_2 \approx 0.$$

**Hinweise.**
- Symmetriere \(C\) numerisch vor der Eigenwertberechnung.  
- Nutze doppelte Genauigkeit; akzeptiere Toleranzen gemäß Tabelle in `README.md`.

---

## B) GADC‑Match

**Ziel.** Ein Stoß mit thermalem Ancilla‑Qubit ist (bis auf Basiskonjugation) eine **GADC** mit Parametern $$\eta=\sin^2\theta,\quad q=p_{\mathrm{exc}}.$$

**Vorgehen.**
1. Baue **Mikro‑Choi** \(C_{\rm micro}\) aus \(U(\theta)\) und \(\tau_B\).  
2. Baue **GADC‑Choi** \(C_{\rm GADC}(\eta,q)\).  
3. **Ausrichtung:** Finde \(U_{\rm in},U_{\rm out}\in SU(2)\), s.d.  
   $$\min_{U_{\rm in},U_{\rm out}} \big\|\,C_{\rm micro} - \big(U_{\rm out}\otimes U_{\rm in}^\*\big)\,C_{\rm GADC}\,\big(U_{\rm out}^\dagger\otimes U_{\rm in}^{\mathsf T}\big)\big\|_F.$$
4. Berichte **Frobenius‑Distanz** und Parameter‑Abweichungen \(|\eta-\sin^2\theta|, |q-p_{\rm exc}|\).

**Hinweis.** Beginne mit **basis‑aligned** und verfeinere adaptiv (kleine SU(2)‑Schritte).

---

## C) Semigruppeneigenschaft

**Ziel.** Markov‑Dynamik erfüllt $$\Phi_{t+s}=\Phi_t\circ\Phi_s.$$

**Metrik.**
- Baue die **affine Pauli‑Transfer‑Matrix** \(T(t)\) am Punkt \(t\) und messe  
  $$\Delta_{\rm semi}=\|T(t+s)-T(t)\,T(s)\|_F.$$

**Hinweise.**
- **Strang** + **Substeps** erhöhen → \(\Delta_{\rm semi}\) fällt.  
- Nutze gleiche Raten \(\gamma_{\downarrow,\uparrow}\) in beiden Schritten.

---

## D) Gibbs‑Fixpunkt & Spohn‑Monotonie

**Ziel.** Prüfe, dass \( \rho_\beta \propto e^{-\beta H_S}\) Fixpunkt ist und die **relative Entropie** $$D(\rho_t\Vert\rho_\beta)=\operatorname{Tr}\!\left[\rho_t(\ln\rho_t-\ln\rho_\beta)\right]$$ **monoton fällt**.

**Metrik.**
- Zähle **Verletzungen** \( \Delta D = D_{k+1}-D_k > \varepsilon \) (relatives \(\varepsilon=10^{-6}\)).  
- Visualisiere \(D(\rho_t\Vert\rho_\beta)\) vs. \(t\).

**Hinweis.** Log‑Eigs stabil berechnen (Clipping minimaler Eigenwerte).

---

## E) MCWF‑Trajektorien ≈ GKSL

**Ziel.** Zeige, dass das Ensemble‑Mittel stochastischer **Quantum‑Jumps** die GKSL‑Lösung reproduziert.

**Metrik.**
- Endzeit‑Abstand $$\max_t T(\overline{\rho}_{\rm traj}(t),\rho_{\rm GKSL}(t)).$$

**Hinweise.**
- **Trajektorienzahl** ≥ 1000; **dt** klein genug; **Seed** dokumentieren.  
- Fehlende Konvergenz → Trajektorienzahl erhöhen / dt verkleinern.

---

## Schwellen & Toleranzen

Siehe Tabelle in `README.md`. Für „Paper‑Plots“: **Strang**, **Substeps ≥ 20**, \(h=\Delta t/N\) klein gegen \(\min(T_1,1/\omega)\).
