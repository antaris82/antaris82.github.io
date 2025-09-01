# ARGUMENTS.md

Vollständige Referenz der Kommandozeilen-Argumente für
`run_pipeline_full_open_systems_v14.6.py`

> **Kurzüberblick.** Die Pipeline hat drei Stufen:
>
> * `--stage build` erzeugt **Artefakte** (ST-Graph, Randbedingungen, Kerne, ρ, POVM/Pixel, Renorm etc.).
> * `--stage analyze` führt **Analysen/Simulationen** (Schrödinger, GKSL/MCWF) + **TI-Tests** auf Basis der Artefakte durch.
> * `--stage all` führt beides in einem Durchlauf aus.
>
> Die Argumente sind in **Module** gruppiert. Die meisten Analyse-Parameter werden bei `build` ignoriert, greifen aber bei `analyze`/`all`. Das CLI-Verhalten orientiert sich an Pythons `argparse` (automatische Hilfe/Validierung). ([Python documentation][1])

---

## 1) Globale Steuerung

### `--stage {build|analyze|all}`  *(Pflicht)*

* **Funktion:** Welche Pipeline-Stufe ausgeführt wird.
* **Hinweise:**

  * `build` schreibt in `--artifacts`, `analyze` liest aus `--artifacts` und schreibt nach `--out`.
  * `all` baut neu und analysiert direkt im Anschluss (konsistent, aber weniger flexibel für Replikate).

### `--artifacts PATH`  *(Pflicht)*

* **Funktion:** Ordner für Artefakte (bei `build`: Ziel; bei `analyze`: Quelle).
* **Best practice:** Pro **Level** eigener Artefakt-Ordner (`out/ART_L4`, `out/ART_L5`, …), damit Analysen level-rein laufen.

### `--out PATH`

* **Funktion:** Zielordner für Analyse-Ergebnisse (Plots, CSV/PDF/JSON der Refutation-Suite, optional Zustandsserien).
* **Empfehlung:** Sinnvolle Namen mit Seed/Profil (`out/ANA_L4_TI_seed13`).

---

## 2) Geometrie, Diskretisierung, Rand (Build-Modul)

### `--level INT`

* **Funktion:** ST-Approximationsgrad (feinere Diskretisierung bei höherem Level).
* **Wirkung:** Rand/Volumen-Effekte ↓, Isotropie ↑, Spektraleigenschaften konvergieren (Heat-Kernel/„Spectral Dimension“ stabilisiert). (Theorie: Analysis auf p.c.f. Fraktalen.) ([projecteuclid.org][2], [AIP Publishing][3])
* **Praxis:** L4 ist ein guter Default für „normalen PC“; L5+ profitiert von mehr Rechenleistung.

### `--t FLOAT`, `--s FLOAT`, `--beta FLOAT`

* **Funktion:** Feinsteuerung der Graph-/Kernel-Generierung (Topologie/Skalierung).
* **Praxis:** Bewährtes Trio: `--t 0.6 --s 0.4 --beta 1.2` (entspricht den Beispiel-Runs).

### `--bc {dirichlet[, …]}`

* **Funktion:** Randbedingungen für die Diskretisierung (z. B. Dirichlet).
* **Hinweis:** Dirichlet verstärkt bei kleinen Leveln Randtransienten; höhere Level relativieren das.

### `--use_fixed_r`

* **Funktion:** Fixiert die Renormierung $r$ auf eine vorgegebene interne Konstante (statt Schätzung).
* **Nutzung:** Für reproduzierbare Vergleiche zwischen Builds nützlich (gleiche Skala über Runs hinweg).

---

## 3) Detektoren / POVM

### `--pixels INT`

* **Funktion:** Anzahl gleichmäßiger Detektor-Pixel (wenn **keine** JSON-Definition genutzt wird).
* **Alternative:** siehe `--pixels_json`.

### `--pixels_json FILE.json`

* **Funktion:** Exakte Pixel-Definition (Regionen) als JSON.
* **Priorität:** Überschreibt `--pixels`.
* **Praxis:** Für konsistente TI-Vergleiche zwischen Leveln (gleiche physikalische Regionen) empfohlen.

### `--eff FLOAT`

* **Funktion:** Gesamteffizienz der POVM (Skalierung der Effekt-Operatoren).
* **Hinweis:** In TI-Tests spielt die **relative** Verteilung die zentrale Rolle; Effizienz wirkt v. a. auf Normierung.

### `--mixer {unitary|none}`

* **Funktion:** Optionale „Mischung“ (z. B. unitäre Glättung) vor der Detektion.
* **Praxis:** `unitary` ist ein robuster Default; `none` für reine Baseline-Vergleiche.

---

## 4) Geschlossene Dynamik (Schrödinger-Modul)

### `--schro {yes|no}`

* **Funktion:** Schrödinger-Schicht aktivieren.
* **Hintergrund:** $\mathrm{i}\,\dot\psi = H\psi$ mit hermiteschem $H$. ([projecteuclid.org][2])

### `--ham {laplacian}`

* **Funktion:** Auswahl des Hamiltonoperators (z. B. diskreter Laplace-Operator).
* **Hinweis:** Der diskrete Laplacian ist mit Heat-Kernel-Skalierung/Spektraldimension konsistent. ([projecteuclid.org][2])

### `--gamma FLOAT`

* **Funktion:** Kopplungs-/Skalierungsparameter in der geschlossenen Dynamik.
* **Praxis:** `1.0` als stabiler Startwert.

### `--sigma0 FLOAT`

* **Funktion:** Breite/Präparation des Anfangszustands (z. B. Gauss-ähnlich).
* **Praxis:** `0.12–0.15` in den Beispiel-Runs.

### `--tmax FLOAT`, `--nt INT`

* **Funktion:** Maximale Simulationszeit und Anzahl der Zeitschritte (für Outputs/Visuals der geschlossenen Schicht).
* **Trade-off:** Höheres `--nt` → feinere Ausgaben, aber mehr Rechenzeit/IO.

### `--save_disp_csv {yes|no}`

* **Funktion:** Export summarischer Größen (z. B. Dispersion) als CSV.

---

## 5) Offene Systeme (GKSL) & Quanten-Trajektorien (MCWF)

### `--open {yes|no}`

* **Funktion:** Lindblad/GKSL-Schicht aktivieren:
  $\dot\rho=-\mathrm{i}[H,\rho]+\sum_j\gamma_j(L_j\rho L_j^\dagger-\tfrac12\{L_j^\dagger L_j,\rho\})$. ([projecteuclid.org][2], [AIP Publishing][3])

### `--dephase_site FLOAT`, `--dephase_pixel FLOAT`, `--loss FLOAT`

* **Funktion:** Stärken von Dephasierung/Verlusten (je nach Implementierung site-/pixel-basiert).
* **Praxis:** Für TI-Validierung **0.00** setzen (vermeidet künstliche frühe Asymmetrien).

### `--det_gamma FLOAT`

* **Funktion:** Detektor-Ausschlagrate in der offenen Dynamik.
* **Trade-off:** Zu groß → erste Klicks „zu früh“ (Transitenten-Bias); zu klein → Laufzeit ↑.
  **Richtwerte (L4):** `1.6–1.8` (gute Ausmischung vor dem Klick).

---

## 6) MCWF-Trajektorien (Quantum-Jumps)

### `--traj {yes|no}`

* **Funktion:** Monte-Carlo-Wellenfunktions-Entfaltung (Quantum-Jump-Unraveling) aktivieren. ([Physical Review][4])

### `--traj_scheme {jumps}`

* **Funktion:** Wahl der Unraveling-Variante (hier: Sprung-Schema).

### `--traj_unobs {nojump[, …]}`

* **Funktion:** Behandlung unbeobachteter Kanäle (z. B. „kein Sprung“ für unobserved).

### `--ntraj INT`

* **Funktion:** Anzahl Trajektorien (Replikate).
* **Statistik:** Standardfehler $\mathrm{SE}\sim\sqrt{p(1-p)/N}$ ↓ mit $N$; aber Achtung: **Auto-ε** wird strenger, wenn $N↑$. (Intervall/SE: Wilson/Brown–Cai–DasGupta.) ([projecteuclid.org][5], [www-stat.wharton.upenn.edu][6])

### `--dt FLOAT`, `--tmax_traj FLOAT`

* **Funktion:** Zeitschritt und maximales Zeitfenster bis zum „ersten Klick“.
* **Praxis:** `--dt 0.01`, `--tmax_traj 3.0–3.3` erzeugen **> 90 %** Klickquote (Frühzeit-Bias ↓).

### `--seed INT`

* **Funktion:** RNG-Seed (Reproduzierbarkeit).
* **Empfehlung:** Mehrere Seeds (z. B. 11/13/19) für Replikate.

---

## 7) Sub-Gaussian / Heat-Kernel-Auswertung (optional)

### `--sg {yes|no}`

* **Funktion:** Sub-Gaussian/Heat-Kernel-Auswertung aktivieren (z. B. Spektraldimension aus $K_t$).

### `--dw FLOAT`, `--sg_tmin FLOAT`, `--sg_tmax FLOAT`, `--sg_nt INT`

* **Funktion:** Parameter des Zeitfensters und eventueller Fensterung/Glättung.
* **Hinweis:** Spektraldimension $d_s$ aus $\log K_t(ii)\sim a-(d_s/2)\log t$. (Fraktal-Analysis / Heat-Kernel.) ([projecteuclid.org][2])

---

## 8) Spektrum/Eigenmoden

### `--eigs_mode {small[, …]}`, `--eigs_k INT`

* **Funktion:** Spektrale Optionen (z. B. kleine Eigenwerte) und Anzahl $k$.
* **Praxis:** Für TI-Validierung reichen oft `small` und `--eigs_k 64` (schnell, informativ).

---

## 9) Output/Protokollierung (IO-Hebel)

### `--save_L {yes|no}`

* **Funktion:** Laplace-/Operator-Schnappschüsse sichern.

### `--save_kernels {yes|no}`, `--save_eigs_csv {yes|no}`

* **Funktion:** Kernel/Eigenspektrum exportieren (CSV).

### `--save_open_rho {yes|no}`, `--open_series_stride INT`

* **Funktion:** Dichteoperator-Zeitscheiben sichern; Stride steuert Ausdünnung.
* **Praxis:** Für schnelle TI-Runs: `--save_open_rho no`, `--open_series_stride 20`.

### `--save_traj_events {yes|no}`, `--save_modes {yes|no}`, `--save_R_diag {yes|no}`, `--save_disp_csv {yes|no}`

* **Funktion:** Feingranulare Ereignisse/Moden/Diagonalen/Dispersions-CSV speichern.
* **Praxis:** Für TI-Checks meist **aus** lassen (IO spart spürbar Zeit).

---

## 10) TI-Validierung (Tests & Schwellen)

### `--alpha FLOAT`

* **Funktion:** Signifikanzniveau (Tests/Intervalle). Wenn gesetzt, überschreibt es den Suite-Standard (üblich: 0.05).
* **Bezug:** GoF (Pearson-χ², LRT-G) basieren auf klassischer Asymptotik (Pearson/Wilks). ([JSTOR][7], [CiteSeerX][8])

### `--eq_margin_abs FLOAT`

* **Funktion:** **Äquivalenzmarge** $\varepsilon$ (absolut, z. B. **0.02** = 2 pp) für TOST je Pixel.
* **Hinweis:** `0` ⇒ **Auto-ε** (streng), typ. $1.96\cdot\mathrm{SE}$.

  * TOST testet „$|\hat p - p_0|<\varepsilon$?“, nicht „Gleichheit“. Für Theorie/Design siehe Wellek. ([Taylor & Francis][9])
* **Praxis:** Für „praktische TI“ empfehlen wir **0.02**; für Härtetests `0`.

---

## 11) Performance & Skalierung (einordnen)

* **Skalierbarkeit:** Trajektorien sind **embarrassingly parallel** (nahezu linear mit Kernzahl). GKSL/MCWF sind Standard in offenen Quanten-Systemen; die Rechenlast verteilt sich auf Operatoranwendungen und Random-Events. ([Physical Review][10])
* **Hebel:**
  `--level↑`, `--ntraj↑`, `--tmax_traj↑`, `--nt↑` ⇒ Aussagekraft ↑, Laufzeit/Memory ↑.
  `--det_gamma` moderat (z. B. 1.6–1.8) ⇒ mehr **Durchmischung vor erstem Klick** ⇒ Bias ↓.
  Dephasing für TI-Checks **0.00** setzen.
* **BLAS/OpenMP:** Thread-Zahl sinnvoll begrenzen (2–8), um Oversubscription zu vermeiden.
* **IO:** `--save_*` sparsam aktivieren; `--open_series_stride` > 1.

---

## 12) Typische Presets (kontextbezogen)

* **Build L4 (einmalig):**

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage build --artifacts out/ART_L4 \
    --level 4 --t 0.6 --s 0.4 --beta 1.2 --bc dirichlet --use_fixed_r
  ```
* **Analyze L4 (ε = 2 pp, Seed 11/13/19):**

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage analyze --artifacts out/ART_L4 --out out/ANA_L4_TI_seed11 \
    --pixels_json detector_pixels.json --mixer unitary \
    --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
    --tmax 3.5 --nt 40 \
    --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0 \
    --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --seed 11 \
    --traj_scheme jumps --traj_unobs nojump \
    --sg no --eigs_mode small --eigs_k 64 \
    --save_L yes --save_kernels no --save_eigs_csv yes \
    --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no \
    --alpha 0.05 --eq_margin_abs 0.02
  ```
* **Auto-ε Härtetest (ε=0, Seed 13; mehr Durchmischung):**

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage analyze --artifacts out/ART_L4 --out out/ANA_L4_TI_AUTO \
    --pixels_json detector_pixels.json --mixer unitary \
    --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
    --tmax 3.5 --nt 40 \
    --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.6 --loss 0.0 \
    --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.2 --seed 13 \
    --traj_scheme jumps --traj_unobs nojump \
    --sg no --eigs_mode small --eigs_k 64 \
    --save_L yes --save_kernels no --save_eigs_csv yes \
    --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no \
    --alpha 0.05 --eq_margin_abs 0
  ```

---

## 13) Methodische Referenzen (für dieses Argument-Dokument)

* **CLI/Argparse (allg. Verhalten/Hilfe):** Python-Doku. ([Python documentation][1])
* **GKSL/Lindblad (Markov. offene Systeme):** Gorini–Kossakowski–Sudarshan; Lindblad. ([AIP Publishing][3], [projecteuclid.org][2])
* **Quantum Trajectories/MCWF (Sprung-Unraveling):** Dalibard–Castin–Mølmer; Plenio–Knight (RMP). ([Physical Review][4])
* **Wilson-Intervalle / Binomial-SE:** Brown–Cai–DasGupta. ([projecteuclid.org][5])
* **TOST/Äquivalenztests (allg. Statistik):** Wellek (Monografie). ([Taylor & Francis][9])
* **GoF-Tests (χ²/LRT):** Pearson (historisch); Wilks (LRT-Asymptotik). ([JSTOR][7])
* **Heat-Kernel/Spektraldimension auf p.c.f.:** Kigami (Buch). ([projecteuclid.org][2])

---

### Glossar (Kurz)

* **TI:** Translationsinvarianz; hier als Übereinstimmung der **empirischen Pixel-Häufigkeiten** mit **POVM-Vorhersagen** verstanden.
* **MCWF/Quantum Jumps:** Stochastische Wellenfunktions-Methode, äquivalent zur GKSL-Mastergleichung auf Ensemble-Ebene. ([Physical Review][10])
* **Auto-ε:** Datengetriebene, strenge Äquivalenzmarge $\varepsilon\approx 1.96\cdot\mathrm{SE}$; bei großem $N$ **enger**. (Intervall-Logik: Brown–Cai–DasGupta; TOST-Design: Wellek.) ([projecteuclid.org][5], [Taylor & Francis][9])

> **Hinweis:** Diese Datei beschreibt die **Bedeutung und Wechselwirkungen** der Argumente. Konkrete **Defaults**/**Validierungen** liegen im Code (Argparse-Definition). Für die exakten Standardwerte bei deiner Version siehe die automatisch generierte `--help`-Ausgabe des Skripts. ([Python documentation][1])

[1]: https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com "argparse — Parser for command-line options, arguments ..."
[2]: https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-48/issue-2/On-the-generators-of-quantum-dynamical-semigroups/cmp/1103899849.full?utm_source=chatgpt.com "On the generators of quantum dynamical semigroups"
[3]: https://pubs.aip.org/aip/jmp/article/17/5/821/225427/Completely-positive-dynamical-semigroups-of-N?utm_source=chatgpt.com "Completely positive dynamical semigroups of N‐level systems"
[4]: https://link.aps.org/doi/10.1103/PhysRevLett.68.580?utm_source=chatgpt.com "Wave-function approach to dissipative processes in quantum ..."
[5]: https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full?utm_source=chatgpt.com "Interval Estimation for a Binomial Proportion"
[6]: https://www-stat.wharton.upenn.edu/~lbrown/Papers/2001a%20Interval%20estimation%20for%20a%20binomial%20proportion%20%28with%20T.%20T.%20Cai%20and%20A.%20DasGupta%29.pdf?utm_source=chatgpt.com "Interval Estimation for a Binomial Proportion"
[7]: https://www.jstor.org/stable/1402731?utm_source=chatgpt.com "Karl Pearson and the Chi-Squared Test"
[8]: https://citeseerx.ist.psu.edu/document?doi=8d94835a2f49607c22a081741a59a502a24d4e43&repid=rep1&type=pdf&utm_source=chatgpt.com "Karl Pearson and the Chi-Squared Test"
[9]: https://www.taylorfrancis.com/books/mono/10.1201/EBK1439808184/testing-statistical-hypotheses-equivalence-noninferiority-stefan-wellek?utm_source=chatgpt.com "Testing Statistical Hypotheses of Equivalence and Noninferiority"
[10]: https://link.aps.org/doi/10.1103/RevModPhys.70.101?utm_source=chatgpt.com "The quantum-jump approach to dissipative dynamics in ..."
