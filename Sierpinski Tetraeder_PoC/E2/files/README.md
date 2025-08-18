# E2 / files — Proof-of-Concept (PoC) für den ST-Graph

Dieser Unterordner bündelt sämtliche Artefakte des **E2-Schritts** im PoC zum **ST-Graph** (Sierpinski-Tetraeder-basierte Ur-Geometrie): Skripte, numerische Ergebnisse (CSV/JSON/TXT) sowie Abbildungen (PNG).  
Ziel: Reproduzierbare Läufe auf unterschiedlichen ST-Leveln (L4/L5/L6) mit konsistenter Auswertung (First-Crossing, Amplitude, OTOC, Fits, Hard-Checks).

---

## Schnellüberblick

- **Thema:** ST-Graph PoC — Schritt *E2*  
- **Inhalt:** Python-Skripte, CSV/JSON-Daten, PNG-Plots, Ergebnis-Notizen  
- **Reproduzierbarkeit:** siehe Abschnitt „Nutzung & Reproduzierbarkeit“

---

## Dateiübersicht

Die Dateien sind nach „Run“ (1–3) gruppiert. Die Kurzbeschreibungen basieren auf Dateinamen und Struktur.

### Run 1 (ST-Level 4)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 1_Level 4.py` | PY | Simulation für **ST-Level 4** (Run 1); erzeugt First-Crossing-Daten/Plots. |
| `E2_run 1_results.json` | JSON | Metadaten/Parameter/ggf. Seeds & Laufinformationen von Run 1. |
| `E2_run 1_ST L4 Ball First Crossing.png` | PNG | Plot **First-Crossing** (Setup „Ball“) auf Level 4. |
| `E2_run 1_STL4_firstcross.csv` | CSV | Tabellare **First-Crossing**-Messwerte für Level 4. |
| `E2_run 1_Chain First Crossing.png` | PNG | Plot **First-Crossing** (Setup „Chain“) auf Level 4. |
| `E2_run 1_chain_firstcross.csv` | CSV | Messwerte **First-Crossing** (Chain) für Level 4. |
| `E2_run 1_Ergebnis.txt` | TXT | Kurzbericht/Notizen zu Ergebnissen und Beobachtungen in Run 1. |

### Run 2 (ST-Level 5)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 2_Level 5.py` | PY | Simulation für **ST-Level 5** (Run 2). |
| `E2_run 2_plot.png` | PNG | Übersichts-/Vergleichsplot für Run 2. |
| `E2_run 2_STL5_eps0p02_alltargets.csv` | CSV | **Alle Targets** für Level 5 bei ε = 0.02 (Referenz/Kalibrierung). |
| `E2_run 2_STL5_J1_fits.csv` | CSV | Ergebnisdatei mit **J1-Fits** (angepasste Parameter/Kurven). |
| `E2_run 2_STL5_hardcheck.json` | JSON | **Hard-Check/Validierung** der L5-Ergebnisse (Konsistenz/Toleranzen). |
| `E2_run 2_Ergebnis.txt` | TXT | Kurzbericht/Notizen zu Run 2. |

### Run 3 (ST-Level 6)
| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E2_run 3_Level 6.py` | PY | Simulation für **ST-Level 6** (Run 3). |
| `E2_run 3_ST Level-6 ball — amplitude front.png` | PNG | **Amplitude-Front** für Setup „Ball“ auf Level 6. |
| `E2_run 3_ST Level-6 ball — OTOC front.png` | PNG | **OTOC-Front** für Setup „Ball“ auf Level 6. |
| `E2_run 3_STL6_amp_otoc.json` | JSON | Konsolidierte Daten **Amplitude+OTOC** (Level 6). |
| `E2_run 3_STL6_eps0p02_amp_targets.csv` | CSV | **Amplitude-Targets** (Level 6) bei ε = 0.02. |
| `E2_run 3_STL6_eps1e-3_otoc_targets.csv` | CSV | **OTOC-Targets** (Level 6) bei ε = 1e-3. |
| `E2_run 3_STL5_hardcheck.json` | JSON | Hard-Check/Referenz für Cross-Validation (L5↔L6). |
| `E2_run 3_Ergebnisse.txt` | TXT | Kurzbericht/Notizen zu Run 3. |

---

## Nutzung & Reproduzierbarkeit

1) **Umgebung (Beispiel)**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # falls vorhanden

2) **Läufe starten**
python "E2_run 1_Level 4.py"
python "E2_run 2_Level 5.py"
python "E2_run 3_Level 6.py"

3) Seeds & Parameter

Bitte in den Skripten dokumentieren (z. B. numpy.random.seed(...), Konfigurationen in cfg.py/config.yaml).
Outputs liegen als CSV/JSON/PNG/ vor (siehe Dateiübersicht).
