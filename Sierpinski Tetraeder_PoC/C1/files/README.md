# C1/files — Simulationen & Bibliotheken

Ordner für die **C1-Studien**: Vergleich des Sierpinski-Tetraeders (ST) mit regulären Gittern (Z^d) bzgl. spektraler Dimension, Heat-Kernel und Lieb–Robinson-Dynamik.

## Pfad
`antaris82.github.io/Sierpinski Tetraeder_PoC/C1/files/`

**Owner:** antaris82

---

## Inhalt & Struktur

### Haupt-Skripte
- `C1_demo_ST_vs_Zd.py` — Demo-Skript für den Vergleich ST vs. Z^d (Heat-Kernel, CTQW).

### Bibliothek `stlib/`
- `dtn.py` — Delta-Triangle-Network Routinen.  
- `gkls.py` — GKLS-Masterequation.  
- `graphs_regular.py` — Z^d-Regulargraph Generatoren.  
- `graph_st.py` — ST-Graph Konstruktor.  
- `heat.py` — Heat-Kernel und Diffusion.  
- `linops.py` — Lineare Operatoren.  
- `quantum.py` — Quanten-Dynamik Routinen.  

### Resultate `result/`
- `C1_heat_trace_compare_from_zip.png` — Vergleich Heat-Kernel ST vs. Z^d.  
- `C1_result.md` — Markdown-Zusammenfassung.  
- `C1_spectral_dimension_median_lines.png` — Spektraldimension-Medianlinien.  

### Outputs `out/`
- JSON: `C1_dtn_delta.json`.  
- CSVs: Heat-Kernel (`C1_heat_*.csv`), ST-Basislinien (`C1_dtn_ST_baseline.csv`), Perturbationen (`C1_dtn_ST_perturbed.csv`), Lieb–Robinson Fronten (`C1_lr_front_*.csv`).  
- Plots: `C1_heat_trace_compare.png`, `C1_lr_front_ST.png`, `C1_lr_front_ST_gkls.png`, `C1_spectral_dimension_compare.png`.

---

## Axiome & Kernpunkte

- **(A1)** ST-Graph als p.c.f.-Fraktalnetz.  
- **(A2)** Vergleich mit regulären Gittern Z^2, Z^3 bzgl. Diffusion und Spektraldimension.  
- **(A3)** Lieb–Robinson-Geschwindigkeit als emergente Kausalstruktur.  

---

## Ergebnisse

- Heat-Kernel auf ST: Polynomabfall, kontrastierend zu Z^d.  
- Spektraldimension ST ≈ 1.67, deutlich niedriger als 2D/3D-Gitter.  
- LR-Front auf ST verlangsamt, aber robust gegen Störungen.  

---

## Akzeptanzkriterien

- (K1) Reproduzierbarkeit von Heat-Kernel & spektraler Dimension.  
- (K2) Konsistenz mit bekannten Exponenten ST vs. Z^d.  
- (K3) LR-Bounds respektiert.  

**Validierungsstatus:**  
| Kriterium | Status |
|-----------|--------|
| K1 | 🟢 |
| K2 | 🟢 |
| K3 | 🟢 |

---

## Reproduzierbarkeit

1. Python ≥3.10, NumPy, SciPy, Matplotlib.  
2. `python C1_demo_ST_vs_Zd.py`.  
3. Ergebnisse in `out/` und `result/` vergleichen.  

---

## Offene Punkte / To-Do

- Erweiterung auf höhere ST-Level (≥7).  
- Quantitative Fits für LR-Geschwindigkeit.  
- Verallgemeinerung auf weitere Fraktale.  

---

## Lizenz

- **Code** (`.py` in `stlib/`, Demo): MIT.  
- **Nicht-Code** (PNGs, CSV, MD): CC BY 4.0.  

© 2025 antaris — Code: MIT; Daten & Abbildungen: CC BY 4.0.
