
<!-- Math uses $$ ... $$ delimiters throughout -->

# README — GKSL Measurement App (Repeated Interactions → GKSL)

**Kurzfassung.** Die App zeigt **Mikro → Makro** in einem Bild: *Stoßmodell* (Repeated Interactions, partielle Austausch‑Unitary + Trace‑out) wird **direkt** gegen die **GKSL/Lindblad‑Dynamik** gespiegelt. Tab 3 liefert eine **Verification‑Suite** (CPTP/Choi, GADC‑Match, Semigruppe, Spohn‑Monotonie, MCWF≈GKSL). Tab 4 exportiert alles als Markdown + Plots.

---

## 1. Physikalische Idee auf einen Blick

- **Bad als Kollisionen.** Frische thermische Ancillas treffen nacheinander auf das System‑Qubit. Ein Stoß: $$\rho'=\operatorname{Tr}_B\!\big[U(\theta)\,(\rho\otimes\tau_B)\,U^\dagger(\theta)\big].$$
- **Ein Schritt ist ein Kanal.** Für Qubits entspricht ein Stoß (bis auf Basiskonjugation) der **GADC** mit $$\eta=\sin^2\!\theta,\qquad q=p_{\mathrm{exc}}=\frac{1}{1+e^{\beta\omega}}.$$
- **Kontinuierliche Grenze.** Für kleine Schritte und konstantes $$K=\frac{\sin^2\!\theta}{\Delta t}$$ entsteht die **GKSL‑Mastergleichung** mit $$\dot\rho=-i[H_S,\rho]+\gamma_\downarrow\mathcal D[\sigma_-](\rho)+\gamma_\uparrow\mathcal D[\sigma_+](\rho),$$ $$\gamma_\downarrow=K\,p_{\mathrm{gnd}},\qquad \gamma_\uparrow=K\,p_{\mathrm{exc}},\qquad \frac{\gamma_\uparrow}{\gamma_\downarrow}=e^{-\beta\omega}.$$
- **Zielzustand.** Fixpunkt ist der **Gibbs‑Zustand** $$\rho_\beta=\frac{e^{-\beta H_S}}{\operatorname{Tr}(e^{-\beta H_S})},\qquad H_S=\frac{\omega}{2}\sigma_z.$$
- **Relaxationszeiten.** Ohne reine Dephasierung $$T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\qquad T_2=2T_1.$$

---

## 2. Tabs & Funktionen

### Tab 1 — Trace‑out & Raten
- Simuliert **Kollisionen**, plottet Populationen/Kohärenzen, berechnet **Raten** \( \gamma_{\downarrow,\uparrow} \) und prüft **KMS**.
- Kern‑Regler: \( \omega,\beta,g,\tau_{\mathrm{int}},\Delta t,\theta_0,\phi_0 \).

### Tab 2 — GKSL & Fits
- Integriert GKSL und vergleicht **Trace‑out vs. GKSL** (Pop., \(|\rho_{01}|\), Phase, Trace‑Distance).
- **Integratorwahl:** *Strang (CPTP, 2. Ordnung)* | *RK4 (+/− PSD)*.
- **Substeps pro \(\Delta t\)**: N (empfohlen: 10–20).
- **Fits** für \(T_1/T_2\) (±95 %-Bänder) + **Δt‑Konvergenz** \(\sim\mathcal O(\Delta t^2)\) bei konstantem \(K\).

### Tab 3 — Verification Suite (A–E)
- **A** CPTP/Choi, **B** GADC‑Match, **C** Semigroup, **D** Gibbs/Spohn, **E** MCWF≈GKSL.

### Tab 4 — Export
- Erzeugt `settings.md`, `results.md`, `verification.md` + ausgewählte Plots (HTML/PNG).

---

## 3. Verification‑Suite — OK‑Schwellen (empfohlen)

| Check | Metrik | Grün | Gelb | Rot | Hinweis |
|---|---|---:|---:|---:|---|
| **A1** CPTP | $$\lambda_{\min}\big(\tfrac{C+C^\dagger}{2}\big)$$ | $$\ge -10^{-12}$$ | $$[-10^{-10},-10^{-12})$$ | $$< -10^{-10}$$ | Choi‑Matrix \(C\) aus Kraus; numerische Toleranzen einkalkulieren. |
| **A2** TP‑Defekt | $$\big\|\sum K_i^\dagger K_i-\mathbb 1\big\|_2$$ | $$\le 10^{-12}$$ | $$\le 10^{-9}$$ | $$>10^{-9}$$ | Durch Rundung/Schrittweiten. |
| **B** GADC‑Distanz | $$\|C_{\text{micro}}-C_{\text{GADC}}\|_F$$ | $$\le 10^{-3}$$ | $$\le 5\cdot 10^{-3}$$ | $$>5\cdot10^{-3}$$ | Zusätzlich: \(|\eta-\sin^2\theta|\le 10^{-2},\ |q-p_{\rm exc}|\le 10^{-2}\). |
| **C** Semigroup | $$\|T(t+s)-T(t)T(s)\|_F$$ | $$\le 10^{-3}$$ | $$\le 5\cdot10^{-3}$$ | $$>5\cdot10^{-3}$$ | \(T(\cdot)\): affine Pauli‑Transfer‑Matrix; fällt mit Substeps \(N\). |
| **D** Spohn | Verletzungen $$\#\{\Delta D> \varepsilon\}$$ | $$0$$ | $$\le 2$$ | $$>2$$ | \(\varepsilon=10^{-6}\) relativ; \(D(\rho_t\Vert\rho_\beta)\) monoton↓. |
| **E** MCWF≈GKSL | $$\max_t T(\rho_{\rm traj},\rho_{\rm GKSL})$$ | $$\le 5\cdot10^{-3}$$ | $$\le 2\cdot10^{-2}$$ | $$>2\cdot10^{-2}$$ | Trajektorien ≥ 1000, feines \(dt_{\rm traj}\). |

> **Faustregeln:** Wähle **Strang** + **N ≥ 10**, halte $$K=\sin^2\theta/\Delta t$$ bei Δt‑Sweeps **konstant** und nutze stabile Logistik für \(p_{\rm exc}=1/(1+e^{\beta\omega})\).

---

## 4. Quickstart

1. **Tab 1**: Defaults → *Run micro trace‑out* → Raten/KMS prüfen.  
2. **Tab 2**: Integrator **Strang**, **Substeps** = 10–20 → GKSL vs. Trace‑out vergleichen; **\(T_1,T_2\)** fitten.  
3. **Tab 3**: A–E ansehen (grün).  
4. **Tab 4**: Export erzeugen (ZIP).

---

## 5. Primärquellen (Kurzliste)

- Lindblad/GKSL (1976), GKS (1976); Davies (1974, Weak Coupling/KMS); Choi (1975, CPTP); Spohn (1978, Entropieproduktion); Dalibard–Castin–Mølmer (1992) & Plenio–Knight (1998, MCWF); Ciccarello et al. (2022, Collision‑Models).

---

**Stand:** 2025-09-18 17:29
