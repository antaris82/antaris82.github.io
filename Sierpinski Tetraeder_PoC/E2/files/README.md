# E2/files — Simulationen und Analysen

Ordner für die **E2-Studien**: Simulationen, numerische Experimente und Auswertungen zu OTOCs, Amplitudenfronten und Crossing-Analysen auf dem ST-Graph.

## Pfad
`antaris82.github.io/Sierpinski Tetraeder_PoC/E2/files/`

**Owner:** antaris82

---

## Dateien & Kurzbeschreibung

- `E2_run 1_Level 4.py` — Python-Skript für Simulation Run 1 (Level-4 ST-Ball, First-Crossing).  
- `E2_run 1_chain_firstcross.csv` — CSV-Daten (First Crossing Chain, Run 1).  
- `E2_run 1_ST L4 Ball First Crossing.png` — Visualisierung der Crossing-Analyse (Level 4).  
- `E2_run 1_results.json` — JSON-Ergebnisse Run 1.  
- `E2_run 1_Ergebnis.txt` — Textzusammenfassung Run 1.

- `E2_run 2_Level 5.py` — Python-Skript für Simulation Run 2 (Level-5 ST-Ball).  
- `E2_run 2_plot.png` — Visualisierung Run 2.  
- `E2_run 2_STL5_eps0p02_alltargets.csv` — CSV-Daten aller Targets (ε=0.02).  
- `E2_run 2_STL5_J1_fits.csv` — Fits J1 (Level-5).  
- `E2_run 2_STL5_hardcheck.json` — Validierungs-JSON Run 2.  
- `E2_run 2_Ergebnis.txt` — Textzusammenfassung Run 2.

- `E2_run 3_Level 6.py` — Python-Skript für Simulation Run 3 (Level-6 ST-Ball).  
- `E2_run 3_ST Level-6 ball — amplitude front.png` — Visualisierung der Amplitudenfront.  
- `E2_run 3_STL6_amp_otoc.json` — OTOC-Daten (Level-6).  
- `E2_run 3_STL6_eps0p02_amp_targets.csv` — Amplitude Targets (ε=0.02).  
- `E2_run 3_STL6_eps1e-3_otoc_targets.csv` — OTOC Targets (ε=1e−3).  
- `E2_run 3_STL5_hardcheck.json` — Validierungs-JSON (Cross-Level).  
- `E2_run 3_Ergebnisse.txt` — Textzusammenfassung Run 3.

---

## Axiome & Kernpunkte

- **(A1)** ST-Graph-Simulationen werden für Level 4–6 durchgeführt.  
- **(A2)** Fokus liegt auf **First-Crossing**, **Amplitude-Fronten** und **OTOC**-Analysen.  
- **(A3)** Ergebnisse werden in Python generiert, numerisch validiert und in CSV/JSON/TXT/PNG dokumentiert.  

---

## Ergebnisse

- Crossing-Dynamik in Level-4–6 ST-Bällen analysiert.  
- Amplituden-Fronten und OTOC-Daten für unterschiedliche ε getestet.  
- JSON-Validierungen (Hardcheck) sichern Reproduzierbarkeit.  

---

## Akzeptanzkriterien

- (K1) Konsistenz der Crossing- und OTOC-Daten.  
- (K2) Validierung der Simulationen durch JSON-Hardchecks.  
- (K3) Nachvollziehbare Dokumentation (CSV/TXT/PNG).  

**Validierungsstatus:**  
| Kriterium | Status |
|-----------|--------|
| K1 | 🟢 |
| K2 | 🟡 |
| K3 | 🟢 |

---

## Reproduzierbarkeit

1. Python-Skripte (`*.py`) ausführen (benötigt NumPy, Matplotlib, evtl. SciPy).  
2. CSV/JSON-Dateien vergleichen mit den generierten Outputs.  
3. Ergebnisse mit den PNG-Plots abgleichen.  

---

## Offene Punkte / To-Do

- Weitere Parameterbereiche (ε) testen.  
- Ergänzung der Analyse für höhere Levels (≥7).  
- Vergleich mit analytischen Bounds (Heat-Kernel, LR-Lichtkegel).  

---

## Lizenz

- **Code** (`*.py`): MIT.  
- **Nicht-Code** (CSV, JSON, PNG, TXT): CC BY 4.0.  

© 2025 antaris — Code: MIT; Daten & Abbildungen: CC BY 4.0.
