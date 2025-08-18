## E2 — Proof-of-Concept (PoC) für den ST-Graph

Dieser Ordner bündelt den **Schritt E2** des ST-Graph‑PoC. Er enthält (i) das Beweis‑PDF `E2_proof.pdf` und (ii) den Unterordner `files/` mit Skripten, Daten, Plots und einer eigenen README.

- 📄 **Beweis:** [`E2_proof.pdf`](./E2_proof.pdf)
- 📦 **Artefakte & Skripte:** [`files/`](./files/) → Detailübersicht in [`files/README.md`](./files/README.md)

---

## Kurzüberblick

- **Thema:** ST‑Graph PoC — Schritt *E2*  
- **Ziel:** Numerische Experimente (Level 4/5/6) + formaler Beweis zu Ausbreitungsgrenzen (Lieb–Robinson → Frontzeit, maximale Geschwindigkeit)  
- **Outputs:** CSV/JSON (Ziel‑/Messgrößen, Fits, Hard‑Checks), PNG‑Plots, TXT‑Notizen  
- **Reproduzierbarkeit:** siehe Abschnitt „Nutzung & Reproduzierbarkeit“

---

## Kernaussagen aus `E2_proof.pdf` (Beweis‑Zusammenfassung)

**Rahmen.** Lokal endliche Graphen mit beschränktem Grad; lokale bzw. exponentiell abfallende Wechselwirkungen.

**Lieb–Robinson‑Bound.** Es existieren Konstanten \\(C,\\mu,v_{\\mathrm{LR}}>0\\) mit
\\[
\\bigl\\|[\\,\\alpha_t(A),\\,B\\,]\\bigr\\|
\\;\\le\\;
C\\,\\|A\\|\\,\\|B\\|\\,
\\exp\\!\\bigl(-\\mu\\,\\bigl[d(X,Y)-v_{\\mathrm{LR}}\\,t\\bigr]\\bigr)
\\]
für lokalisierte Observablen \\(A\\in\\mathcal A_X\\), \\(B\\in\\mathcal A_Y\\) und Distanz \\(d(X,Y)\\).

**Frontzeit (Untergrenze) bei Toleranz \\(\\varepsilon>0\\).** Für
\\[
t_\\varepsilon(d):=\\inf\\{\\,t\\ge0:\\|[\\,\\alpha_t(A),B\\,]\\|\\le\\varepsilon\\,\\}
\\]
gilt
\\[
t_\\varepsilon(d)\\;\\ge\\;\\frac{d}{v_{\\mathrm{LR}}}-\\frac{1}{\\mu\\,v_{\\mathrm{LR}}}\\,
\\ln\\!\\frac{C\\,\\|A\\|\\,\\|B\\|}{\\varepsilon}\\,.
\\]
Damit folgt eine **maximale Gruppengeschwindigkeit** \\(v^*\\le v_{\\mathrm{LR}}\\) und ein (nahezu) **linearer emergenter Lichtkegel**. Die Voraussetzungen sind für die **ST‑Graph‑Approximanten** erfüllt; Varianten decken exponentiell abfallende Interaktionen, offene Systeme (Lindbladiane) und — mit modifizierten Exponenten — bestimmte Long‑Range‑Fälle ab.

---

## Ordnerstruktur


> **Hinweis:** Die **vollständige Dateiübersicht** steht in `files/README.md`. Nachfolgend eine Zusammenfassung der Runs (1–3).

---

## Dateiübersicht (Zusammenfassung des Unterordners `./files/`)

### Run 1 (ST‑Level 4)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 1_Level 4.py` | PY | Simulation für **ST‑Level 4** (Run 1); erzeugt First‑Crossing‑Daten/Plots. |
| `E2_run 1_results.json` | JSON | Metadaten/Parameter/Seeds & Laufinformationen von Run 1. |
| `E2_run 1_ST L4 Ball First Crossing.png` | PNG | Plot **First‑Crossing** (Setup „Ball“) auf Level 4. |
| `E2_run 1_STL4_firstcross.csv` | CSV | Tabellare **First‑Crossing**‑Messwerte für Level 4. |
| `E2_run 1_Chain First Crossing.png` | PNG | Plot **First‑Crossing** (Setup „Chain“) auf Level 4. |
| `E2_run 1_chain_firstcross.csv` | CSV | **First‑Crossing**‑Messwerte (Chain) für Level 4. |
| `E2_run 1_Ergebnis.txt` | TXT | Kurzbericht/Notizen zu Run 1. |

### Run 2 (ST‑Level 5)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 2_Level 5.py` | PY | Simulation für **ST‑Level 5** (Run 2). |
| `E2_run 2_plot.png` | PNG | Übersichts‑/Vergleichsplot für Run 2. |
| `E2_run 2_STL5_eps0p02_alltargets.csv` | CSV | **Alle Targets** für Level 5 bei ε = 0.02 (Referenz/Kalibrierung). |
| `E2_run 2_STL5_J1_fits.csv` | CSV | Ergebnisdatei mit **J1‑Fits** (angepasste Parameter/Kurven). |
| `E2_run 2_STL5_hardcheck.json` | JSON | **Hard‑Check/Validierung** der L5‑Ergebnisse (Konsistenz/Toleranzen). |
| `E2_run 2_Ergebnis.txt` | TXT | Kurzbericht/Notizen zu Run 2. |

### Run 3 (ST‑Level 6)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 3_Level 6.py` | PY | Simulation für **ST‑Level 6** (Run 3). |
| `E2_run 3_ST Level-6 ball — amplitude front.png` | PNG | **Amplitude‑Front** für Setup „Ball“ auf Level 6. |
| `E2_run 3_ST Level-6 ball — OTOC front.png` | PNG | **OTOC‑Front** für Setup „Ball“ auf Level 6. |
| `E2_run 3_STL6_amp_otoc.json` | JSON | Konsolidierte Daten **Amplitude + OTOC** (Level 6). |
| `E2_run 3_STL6_eps0p02_amp_targets.csv` | CSV | **Amplitude‑Targets** (Level 6) bei ε = 0.02. |
| `E2_run 3_STL6_eps1e-3_otoc_targets.csv` | CSV | **OTOC‑Targets** (Level 6) bei ε = 1e‑3. |
| `E2_run 3_STL5_hardcheck.json` | JSON | Hard‑Check/Referenz für Cross‑Validation (L5↔L6). |
| `E2_run 3_Ergebnisse.txt` | TXT | Kurzbericht/Notizen zu Run 3. |

---

## Nutzung & Reproduzierbarkeit

1) **Umgebung (Beispiel)**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r files/requirements.txt  # falls vorhanden

2) **Läufe starten**
```bash
python "files/E2_run 1_Level 4.py"
python "files/E2_run 2_Level 5.py"
python "files/E2_run 3_Level 6.py"

3) Parameter & Seeds
Bitte in den Skripten dokumentieren (z. B. numpy.random.seed(...), Konfiguration in cfg.py/config.yaml).
Outputs liegen als CSV/JSON/PNG/TXT in files/.

