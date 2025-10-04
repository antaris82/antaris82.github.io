
# CHANGELOG — I8\_v0.2

**Release Date:** 2025-10-04

> Diese Version fokussiert auf robuste **RAM-/CPU-Optimierungen** ohne Funktionsverlust, klarere **Geometrie-Deformation des REAL-Graphen** bei **fixem Rand**, eine **exakt CPTP**-Kanalableitung aus der globalen Unitarität, sowie **stabile** Evidenz- und Diagnoseroutinen.

---

## ✨ Neu (Features)

- **Harmonische Einbettung des REAL-Graphen (Rand fix)**  
  Der REAL-Graph wird aus der Dirichlet-harmonischen Einbettung des REAL-Laplacians mit unverändertem Rand konstruiert.  
  $$ L\_{II}\,x\_I \;=\; -\,L\_{IB}\,x\_B \quad\Rightarrow\quad x\_I \;=\; -\,L\_{II}^{-1} L\_{IB}\,x\_B $$
  → Sichtbare **Innen-Deformation** mit **unverändertem äußerem Rand** (Anforderung erfüllt).

- **Auswahl des Initialzustands \(\rho\_0\) in der Sidebar**  
  Optionen: „Lokal rein (tiefster Level)“ (Default), „Zufällig rein“, „Zufällig gemischt“, „Maximale Mischung“.  
  Hintergrund: \(\rho\_0=\mathbb{1}/n\) bleibt unter LvN und unitaler GKSL invariant; die neuen Optionen liefern **nichttriviale** Dynamik.

- **Start-Button & Auto-Save**  
  *Start (Run & Save)* in der Sidebar. Alle Plots, Matrizen und Metadaten werden automatisch in einen **Run-Ordner** geschrieben und als **ZIP** exportiert.

- **Exakt CPTP-Kanal \(\Phi\_{\Delta t}\) aus globaler Unitarität (Kraus-Blöcke)**  
  Statt einer teuren Basis- oder Choi-Konstruktion wird \(U=\exp(-i H\_{\text{full}} \Delta t)\) blockweise in \((S,E)\) zerlegt,  
  $$ \Phi\_{\Delta t}(\cdot) \;=\; \sum\_{\alpha,\beta} K\_{\alpha\beta}(\cdot) K\_{\alpha\beta}^\dagger, \qquad
     K\_{\alpha\beta}\;=\;\sqrt{p\_\beta}\, U\_{\alpha\beta}, \quad \rho\_E=\sum\_\beta p\_\beta \lvert \beta\rangle\langle\beta\rvert $$
  → **CPTP per Konstruktion**, mit Trace-Preservation-Check \(\big\|\sum K^\dagger K - \mathbb{1}\big\|\).

---

## 🛠️ Geändert (Behaviour & Pipeline)

- **REAL-Gewichte (R\_eff):** Nutzung **effektiver Widerstände** auf Kanten, aber **ohne Pseudoinversen**. Einmalige LU-Faktorisierung des „geerdeten“ Laplacians und schnelle Solves:  
  $$ R\_{ij} \;=\; (e\_i - e\_j)^\top L^{\sim -1} (e\_i - e\_j),\quad \text{mit Erdung und }L^{\sim} y = b $$
  Innenkanten werden levelabhängig verstärkt, Randkanten bleiben Gewicht \(1\).

- **DtN-Kalibrierung \(s^\*\) & DtN-Fehler:** Schur-Komplement per **Sparse-Solves**  
  $$ \Lambda \;=\; L\_{BB} - L\_{BI}\,L\_{II}^{-1} L\_{IB},\qquad 
     \varepsilon\_{\text{DtN}} \;=\; \frac{\|\Lambda\_{\text{REAL}} - \Lambda\_{\text{IDEAL}}\|\_F}{\max(1,\|\Lambda\_{\text{IDEAL}}\|\_F)} $$

- **Cheeger/Fiedler:** zweites Eigenpaar via `eigsh` (sparse), **kein** volles `eigh`.  
  Cheeger-Grenze: \( \lambda\_1 \ge \tfrac{1}{2} h^2 \).

- **Varadhan-Test (Kurzzeit-Heat-Kernel):** Heat-Kernel mit `expm_multiply`, Widerstände pro Paar via einer LU — **ohne** dichte \(L^{+}\).

- **Dreiecksungleichung (Widerstandsmetrik):** stichprobenartige Prüfung mit **Sampling ohne Wiederholung**, Tripel-Deduplikation und adaptiver Obergrenze der Versuche.

- **REAL-Kanal auf Subsystem \(S\):** Partitionen „Level-Cut“ oder „Boundary-Cluster“ mit **Obergrenze** \(|S|\le S\_\text{cap}\) zur kontrollierten Superoperatorskalierung.

---

## 🐞 Gefixt (Bugs)

- **Out-of-Memory (32 GiB) bei Memory-Kernel-Rekonstruktion**  
  Entfernt: Riesige Kronecker-Produkte \((V\times V)\) mit \(V=n\_S^2\).  
  Neu: **Skizzierte Least Squares** auf einer orthonormierten, kleinen Testsuite \(S\) (q Spalten), gleiche Operatorgleichungen, **OOM-frei**.  
  $$ \min\_{K\_m}\sum\_t \big\| \Delta T\_t S \;-\; \sum\_{m=0}^{M} K\_m \, (T\_{t-m} S)\big\|\_F^2 $$
  Rückgabewert: **Rekonstruktionsfehler** als Qualitätsmaß.

- **NameError \(\Delta t\)** in der Statuszeile  
  Anzeige nutzt jetzt konsistent `dt`.

- **Sampling-Fehler** in der Dreiecksprüfung  
  Sichere Logik: *ohne Wiederholung* und mit Set-basiertem Tripel-Cache.

- **REAL/IDEAL-Dichte** „identisch“ trotz unterschiedlicher Modelle  
  Ursache: \(\rho\_0=\mathbb{1}/n\). Behoben durch **Sidebar-Auswahl**; Default ist **lokal rein** (tiefster Level).

---

## ⚡ Performance (RAM & CPU)

- **Kanal \(\Phi\_{\Delta t}\)**:  
  - vorher: vollständige Basis-/Choi-Konstruktion, \(O(n\_S^4)\) Operationen & Speicher.  
  - jetzt: **Kraus aus U-Blöcken**, exakt CPTP, signifikant weniger Multiplikationen, **keine** vierfach verschachtelten Schleifen.

- **Effektive Widerstände & Varadhan:**  
  - eine **einzige** Sparse-LU für den „geerdeten“ Laplacian → viele Paar-Anfragen **ohne** neue Faktorisierung.

- **Fiedler/Cheeger:**  
  - `eigsh(k=2, which='SM')` auf **CSR**, keine dichte Vollspektral-Analyse.

- **Harmonische Einbettung & DtN:**  
  - beides mit **Sparse-Solves** statt dichten Pseudoinversen.

- **Memory-Kern:**  
  - **skizzierte** LS, \(O(V q)\) statt \(O(V^2)\) mit \(q\ll V\).

---

## 🔬 Evidenz & Diagnostik (unverändert im Sinn, effizienter in der Umsetzung)

- **Markov-Diagnostik:**  
  - \( T\_{\text{err}} = \frac{\|\Phi\_{\Delta t} - e^{\Delta t \mathcal{L}\_{\text{REAL}}}\|\_F}{\max(1,\|e^{\Delta t \mathcal{L}\_{\text{REAL}}}\|\_F)} \)  
  - **RHP** (CP-Divisibilität): Choi-Minimalwert der Zwischenkarte \( \Phi\_{2\Delta t}\Phi\_{\Delta t}^{-1} \).  
  - **BLP**-Backflow: Zunahmen der Trace-Distanz \(D(t)\).  
  - **Entropy-Monotonie** unter GKSL (CPTP/PSD/TP Checks).

- **„Higgs“-Surrogat \(m\_A\):** aus \(\rho\_{\text{REAL}}\),  
  $$ m\_A \;\approx\; \bigg(\frac{1}{n}\sum\_{i} \sqrt{\rho\_{\text{REAL}}(i,i)}^{\,2}\bigg)^{1/2} $$

- **Cluster & Ordnungsparameter:** Fiedler-Modus → binäre Clusterung; Ordnungsparameter \(M\), **Binder-Kumulant** \(U\_4\).

---

## 📦 Export & Artefakte

Automatische Speicherung in `runs/{run_id}`:  
- `X_ideal.npy`, `X_real_harmonic.npy`  
- `L_ideal.npy`, `L_real.npy`  
- `rho_LvN.npy`, `rho_GKSL.npy`  
- `Phi_dt_super.npy`, `exp_dt_L_REAL_super.npy` (falls \(|S|\ge 1\))  
- `edges.json`, `meta.json`  
- Interaktive HTML-Plots: `ideal_graph.html`, `real_graph_harmonic.html`, `rho_ideal_heatmap.html`, `rho_real_heatmap.html`  
- Gesamtpaket als **`bundle.zip`**

---

## 🔁 Kompatibilität & Migration

- Keine API-Änderungen an den Hauptparametern.  
- **Empfehlung:** Bei großen Graphen `S_cap` konservativ halten (z. B. 8–16); die skizzierte Memory-LS passt sich automatisch an.  
- **REAL-Layout:** Für alte Runs ohne `X_real_harmonic.npy` wird die Deformation nicht angezeigt; neue Runs erzeugen diese Datei automatisch.

---

## 🧪 Validierung (Primärformeln, keine Vereinfachungen)

- **Dirichlet-Form & DtN**: Schur-Komplement exakt wie in der Literatur.  
- **Effektiver Widerstand**: Lösung des geerdeten Netzes liefert exakt dieselbe Größe wie über \(L^+\), jedoch numerisch stabiler.  
- **GKSL**: Standardform mit \( \mathcal{L}(\rho)= -i[H,\rho]+\sum\_k L\_k \rho L\_k^\dagger - \tfrac{1}{2}\{L\_k^\dagger L\_k,\rho\} \).  
- **Kanal-Konstruktion**: Kraus-Zerlegung aus globaler Unitarität mit gemischter \(\rho_E\) sichert **CPTP** ohne nachträgliche Projektion.

---

## 📚 Zusammenfassung

I8\_v0.2 liefert **identische wissenschaftliche Aussagen** wie zuvor, führt sie aber **numerisch deutlich effizienter** aus, macht die **REAL-Deformation** bei **fixem Rand** **sichtbar**, und stellt die **Kanal-Ableitung** \(\Phi\_{\Delta t}\) **exakt CPTP** sicher — bei zugleich **stabiler** Evidenzprüfung (Cheeger/Varadhan/RHP/BLP/Entropy/Binder).
