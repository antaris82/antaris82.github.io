# E4 — Robustheit via Homotopie G(θ)

Dieses Verzeichnis dokumentiert den vierten Schritt des PoC zum **ST‑Graph**. Im Fokus steht die **Robustheit der Kinematik** gegen Mikrodeformationen: eine kontinuierliche Familie gewichteter Graphen \(G(\theta)\), die von der Sierpiński‑Tetraeder‑Geometrie (\(\theta=0\)) zu einer baumartigen Struktur (\(\theta=1\)) homotopiert. Zielgrößen sind der **Spektral‑Dimensions‑Proxy** \(\hat d_s(\theta)\) und die **Frontgeschwindigkeit** \(\hat v^*(\theta)\).

---

## Inhalte

### Kern‑Dokumente
- **`E4_formal.pdf`** – mathematische Formulierung: Definition von **A(θ)**, **L(θ)=D−A**, Messgrößen (\(\hat d_s\), \(\hat v^*\)), theoretische Erwartungen (Fraktal‑Power‑Law → Baum‑Exponential), Lieb–Robinson‑Beschränktheit und Akzeptanzkriterien.
- **`E4_formal_fixed.pdf`** – überarbeitete/korrigierte Fassung des obigen Dokuments.

### Unterordner [`files/`](./files/)
Begleitende Artefakte zur Simulation und Visualisierung:
- **`E4_simulation.py`** – Python‑Pipeline zur Erzeugung von \(G(\theta)\), Heat‑Kernel‑Analyse (\(\hat d_s\)) und CTQW‑Front (\(\hat v^*\)).
- **`E4_homotopy_summary.csv`** – aggregierte Ergebnisse über \(\theta\) (z. B. 0.00, 0.25, 0.50, 0.75, 1.00); enthält u. a. Spalten `theta`, `ds_est`, `ds_r2`, `vstar_est`, `vstar_r2`.
- **`E4_ds_theta.png`** – Plot von **\(\hat d_s(\theta)\)**.
- **`E4_vstar_theta.png`** – Plot von **\(\hat v^*(\theta)\)**.

> Eine ausführliche Dateiübersicht und Repro‑Hinweise findest du direkt in [`files/README.md`](./files/README.md).

---

## Ergebnisse (Kurzfassung)

- **Trend \(\hat d_s(\theta)\):** fällt von ca. **1.67** (ST, \(\theta=0\)) auf ca. **0.80** (baumartige Spannung, \(\theta=1\)).  
- **Trend \(\hat v^*(\theta)\):** bleibt **endlich** und steigt **moderat** (etwa **0.51 → 0.55**).  
- **Interpretation:** Übergang von **sub‑gaußscher Potenz‑Abnahme** des Heat‑Kernels (Fraktal/pcf) zu **exponentieller Abnahme** im Baum‑Limit; der effektive Kausal‑Kegel bleibt bestehen (Lieb–Robinson).

### Akzeptanzkriterien (erfüllt)
- **K1:** Glatter Trend in \(\hat d_s(\theta)\) bzw. zunehmende Exponentialrate.  
- **K2:** \(\hat v^*(\theta)\) bleibt endlich, mäßige Variation mit \(\theta\).  
- **K3:** Konsistenz mit der Theorie (Fraktal‑Power‑Law vs. Baum‑Exponential).

---

## Nutzung & Reproduzierbarkeit (Kurz)

1) **Python‑Umgebung** (für Artefakte in `./files/`):
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r files/requirements.txt  # falls vorhanden
```

2) **Beispielablauf** (Details siehe `files/README.md`):
```bash
python files/E4_simulation.py
# erzeugt/aktualisiert: files/E4_homotopy_summary.csv und Plots
```

---

## Lizenz

- **Code** (insb. in `./files/`): **MIT License**.  
- **Nicht‑Code** (z. B. PDFs, CSV/PNG): **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

> © 2025 antaris — **Code:** MIT; **Daten/Abbildungen/Texte (inkl. PDFs):** CC BY 4.0.

---

## Zitation

Bitte referenziere:
> antaris (2025): *ST‑Graph PoC — E4: Robustheit via Homotopie G(θ)*. Ordner `Sierpinski Tetraeder_PoC/E4/`.  
> **Code:** MIT. **Daten & Abbildungen (inkl. PDFs):** CC BY 4.0.

(Optional: `CITATION.cff` im Repo‑Root hinzufügen, damit GitHub die Zitation automatisch anzeigt.)
