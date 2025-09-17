
<!-- Math uses $$ ... $$ delimiters throughout -->

# README — GKSL Measurement App (Repeated Interactions → GKSL)

**Zweck.** Diese App demonstriert, wie aus einem mikroskopischen **Stoßmodell** (Repeated Interactions, partielle Austausch‑Unitary + partieller Trace) im geeigneten Grenzfall eine markovsche **GKSL/Lindblad‑Dynamik** mit **Gibbs‑Fixpunkt** entsteht. Sie stellt Messgrößen wie **\(T_1\)**/**\(T_2\)** dar, prüft physikalische Konsistenz (CPTP/Choi, Semigruppeneigenschaft, Spohn‑Monotonie) und exportiert ein reproduzierbares Paket (Markdown + Plots).

---

## Physikalischer Hintergrund (kurz)

**Thermisches Bad (Qubit).**
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

**Einzelstoß (Unitary + partielle Spur).**
Mit Austausch‑Unitary \(U(\theta)\) (Rotation im \(\{|01\rangle,|10\rangle\}\)-Block):
$$
\rho_S' \;=\; \mathrm{Tr}_B\!\big[\,U(\theta)\,\big(\rho_S\!\otimes\!\tau_B\big)\,U^\dagger(\theta)\big]
\;=\;\Phi_{\text{coll}}(\rho_S).
$$

**Kraus aus dem Mikromodell.**
Schreibe \(\tau_B=\sum_i p_i |i\rangle\langle i|\). Dann
$$
K_{ji}\;=\;\sqrt{p_i}\,\langle j|U(\theta)|i\rangle,
\qquad
\Phi_{\text{coll}}(\rho)=\sum_{i,j}K_{ji}\,\rho\,K_{ji}^\dagger.
$$

**Raten \(\gamma_{\downarrow},\gamma_{\uparrow}\) (Kleinwinkel/Collision‑Rate \(r=\Delta t^{-1}\)).**
Für \(\theta\approx g\,\tau_{\mathrm{int}}\) und \( \sin^2\theta \) klein:
$$
\gamma_{\downarrow}=r\,\sin^2\!\theta\;p_{\mathrm{gnd}},
\qquad
\gamma_{\uparrow}=r\,\sin^2\!\theta\;p_{\mathrm{exc}},
\qquad
\frac{\gamma_{\uparrow}}{\gamma_{\downarrow}}=e^{-\beta\omega}\;\;(\text{KMS}).
$$

**GKSL/Lindblad‑Dynamik (thermische Amplitudendämpfung).**
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
Ohne reine Dephasierung gilt
$$
T_1=\frac{1}{\gamma_{\downarrow}+\gamma_{\uparrow}},
\qquad
T_2=2T_1.
$$

**Prüfgrößen.**
Semigruppeneigenschaft \(\Phi_{t+s}=\Phi_t\!\circ\!\Phi_s\);
relative Entropie
\(D(\rho\Vert\sigma)=\mathrm{Tr}[\rho(\ln\rho-\ln\sigma)]\)
mit Spohn‑Monotonie \(D(\rho_t\Vert\rho_\beta)\searrow\);
Trace‑Distance
\(T(\rho,\sigma)=\tfrac12\|\rho-\sigma\|_1\).

---

## Tab 1 — „Trace‑out & Raten“

**Einstellungen** (linke Spalte):
- **\(\omega\) (System)** — Energielücke des Qubits.
- **\(\beta\) (Bad)** — inverse Temperatur des Bades.
- **\(g\)** — Kopplungsstärke innerhalb eines Stoßes.
- **\(\tau_{\mathrm{int}}\)** — Pulsdauer (Stoßdauer).
- **\(\Delta t\)** — Kollisionsintervall (setzt \(r=1/\Delta t\)).
- **Anzahl Kollisionen** — Länge der Simulation.
- **\(\theta_0,\phi_0\)** — Startzustand \( |\psi_0\rangle =
\big(\cos \tfrac{\theta_0}{2},\; e^{i\phi_0}\sin \tfrac{\theta_0}{2}\big)^\top\).

**Aktion** (rechte Spalte):
- **Run micro trace‑out** — berechnet die Kaskade
  \(\rho_{k+1}=\Phi_{\text{coll}}(\rho_k)\), speichert Trajektorien, Gibbs‑Populationswerte \((p_0^\*,p_1^\*)\) und aus dem Mikromodell abgeleitete **Raten** \((\gamma_{\downarrow},\gamma_{\uparrow})\) inkl. **KMS‑Check**.

**Plots (Tab 1):**
- **Populationen** \( \rho_{00},\rho_{11} \) vs. \(t\) (inkl. \(p^\*\)‑Linien).
- **Kohärenz** \( |\rho_{01}| \) vs. \(t\).

---

## Tab 2 — „GKSL & Fits“

**Zweck.** Integration der GKSL‑Gleichung mit den in Tab 1 gewonnenen Raten; Vergleich **Trace‑out ↔ GKSL**; **Fits** von \(T_1/T_2\) mit **95 %‑CI**; **Δt‑Konvergenz**.

**Numerik.**
- **Integrator:** aktuell **RK4** mit **positiver Projektion** (EVD‑Clipping) **pro Substep**; interne Substepping‑Faktor **5** pro \(\Delta t\).
  > Hinweis: Die PSD‑Projektion ist **numerisch** (keine Physik). Für strengere Bounds \(\Rightarrow\) \(\Delta t\) verkleinern / Substeps erhöhen (falls parametrisiert) oder Strang‑Splitting implementieren.

**Bedienelemente & Kennzahlen.**
- **Fit‑Quelle:** *Trace‑out* oder *GKSL* (welche Kurve gefittet wird).
- **95 %‑CI anzeigen:** Konfidenzbänder der Exponential‑Fits.
- **Erwartete Zeiten:** \(T_1=1/(\gamma_{\downarrow}+\gamma_{\uparrow})\), \(T_2=2T_1\).
- **GADC‑Fit‑Hilfen (für Verifikation, siehe Tab 3):**
  - **η/q‑Suchbreite um Ziel**, **koarse η/q‑Punkte**, **Startschritt (rad) für δU**, **max\_iter (adaptive SU(2))**.
- **Semigroup‑Zeiten:** **Zeit \(t\)** und **Zeit \(s\)** für den Test \( \Phi_{t+s}\approx\Phi_t\circ\Phi_s \).
- **MCWF‑Trajektorien:** **Anzahl Trajektorien**, **dt (Trajektorien)**, **t\_max (Trajektorien)**, **Seed**.

**Plots (Tab 2):**
- **\( \rho_{00} \)** und **\(|\rho_{01}|\)**: *Trace‑out* vs. *GKSL* + Gibbs‑Linien.
- **\(\arg\rho_{01}\)** und **Trace‑Distance** \(T(\rho_{\text{to}},\rho_{\text{GKSL}})\).
- **Exponential‑Fits \(T_1/T_2\)** mit **±95 %‑Bändern** (zwei sichtbare Linien + Füllung).
- **Δt‑Konvergenz (Log‑Log):** max \(T(\rho_{\text{to}},\rho_{\text{GKSL}})\) vs. \(\Delta t\) \(\sim \mathcal O(\Delta t^2)\) bei konstantem \(K=\sin^2\theta/\Delta t\).

---

## Tab 3 — „Verification Suite“ (A–E)

**A) CPTP/Choi (ein Schritt).**  
Erzeugt Kraus \(K_{ji}\) aus Mikro‑\(U(\theta)\) und \(\tau_B\), baut **Choi‑Matrix** \(C\) und prüft
- **TP‑Defekt** \( \big\|\sum_i K_i^\dagger K_i - \mathbb{1}\big\| \approx 0 \),
- **min‑Eigenwert** von \( \tfrac12(C+C^\dagger) \ge 0 \) (CPTP).

**B) GADC‑Fit (ein Schritt).**  
Vergleicht **Mikro‑Choi** mit **GADC** (Generalized Amplitude Damping, Parameter \(\eta,q\)); richtet die Energiebasis aus (SU(2)), verfeinert \(U_{\mathrm{in}},U_{\mathrm{out}}\) adaptiv und meldet Choi‑Distanz \(\|C_{\text{micro}}-C_{\text{GADC}}\|_F\). Ziel: \(\eta\approx \sin^2\theta,\;q\approx p_{\mathrm{exc}}\).

**C) Semigruppeneigenschaft.**  
Verwendet die **affine Pauli‑Transfer‑Matrix** \(T(t)\) und misst den Defekt \(\|T(t+s)-T(t)T(s)\|_F\).

**D) Gibbs & Spohn.**  
Bestätigt Gibbs‑Fixpunkt und zeigt **Spohn‑Monotonie**: \( D(\rho_t\Vert\rho_\beta) \) fällt in der GKSL‑Dynamik.

**E) Trajektorien ≈ GKSL.**  
Vergleicht MCWF‑Surrogat/„Quantum‑Jumps“ mit GKSL‑Lösung via Trace‑Abstand (Endzeit).

---

## Tab 4 — „Export als ZIP“

**Auswahl.**
- **Einstellungen (Parameter) als `settings.md`** und **Werte/Ergebnisse als `results.md`** (inkl. Raten, KMS, \(T_1/T_2\), Verifikations‑Kennzahlen).
- **Grafiken** aus Tab 1–3 (je nach Auswahl).

**Inhalt & Benennung.**
- Markdown‑Dateien: `settings.md`, `results.md`, `verification.md`.
- Plots (HTML/PNG, je nach Umgebung), z. B. `tab2_pop00.html`, `tab2_fit_T1.html`, `tab2_dt_convergence.html`, `tab3_D_relent.html` etc.

---

## Good Practice & Hinweise

- **Regime:** Für gute Übereinstimmung zwischen Trace‑out und GKSL arbeite im Kleinwinkel/Weak‑Coupling‑Regime, halte \(K=\sin^2\theta/\Delta t\) bei Δt‑Sweeps konstant.
- **Numerik:** RK4‑Schritt nutzt PSD‑Projektion (Eigenwerte clippen) zur **numerischen** Positivität. Für strengere Nachweise \(\Rightarrow\) Schrittweiten verringern (internes Substepping), optional Strang‑Splitting implementieren.
- **Interpretation der Fits:** Ohne reine Dephasierung sollte **\(T_2\approx 2T_1\)** gelten; Abweichungen deuten auf Fit‑Fenster, numerische Artefakte oder zusätzliche dephasierende Beiträge hin.
- **Reproduzierbarkeit:** Nutze den Export (Tab 4) direkt als Anhang in Notebooks/Manuskripten; alle Kennzahlen werden mit den Plots gesichert.

---

## Primärquellen (Kernaussagen)

- **Lindblad‑Form / GKSL:**  
  G. Lindblad, *Commun. Math. Phys.* **48** (1976) 119–130.  
  V. Gorini, A. Kossakowski, E.C.G. Sudarshan, *J. Math. Phys.* **17** (1976) 821–825.

- **Davies‑Grenzfall (Weak Coupling, KMS/Detail Balance):**  
  E.B. Davies, *Commun. Math. Phys.* **39** (1974) 91–110.

- **CPTP‑Kriterium:**  
  M.-D. Choi, *Linear Algebra Appl.* **10** (1975) 285–290.

- **Spohn‑Monotonie / Entropieproduktion:**  
  H. Spohn, *J. Math. Phys.* **19** (1978) 1227–1230.

- **MCWF/Quantum‑Jumps ≙ GKSL:**  
  J. Dalibard, Y. Castin, K. Mølmer, *Phys. Rev. Lett.* **68** (1992) 580–583.  
  M.B. Plenio, P.L. Knight, *Rev. Mod. Phys.* **70** (1998) 101–144.

- **Collision‑Models (Repeated Interactions) – Überblick:**  
  F. Ciccarello, S. Lorenzo, V. Giovannetti, G.M. Palma, *Physics Reports* **954** (2022) 1–70.

---

### Changelog

- 2025-09-17 20:00: Erste konsolidierte README mit Formel‑Supplement, Tab‑Dokumentation, Hinweisen & Primärquellen.
