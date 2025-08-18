# E4 / files — Homotopie‑Scan auf dem ST‑Graph

Dieser Unterordner enthält alle Artefakte des **E4‑Schritts**: die **Simulation**, die **aggregierten Ergebnisse (CSV)** und **zwei Plots** zum Verlauf der **Spektral‑Dimension** (Proxy) \(\hat d_s(\theta)\) und der **Frontgeschwindigkeit** \(\hat v^*(\theta)\) entlang eines **Homotopie‑Parameters** \(\theta\in[0,1]\).

## Inhalt

| Datei | Typ | Kurzbeschreibung |
|---|---|---|
| `E4_simulation.py` | PY | End‑to‑End‑Pipeline: ST‑Subgraph bauen, gewichtete Matrix **A(θ)** & Laplace **L(θ)** konstruieren, **Heat‑Kernel‑Fit** → \(\hat d_s\), **CTQW‑Front** → \(\hat v^*\); schreibt `E4_homotopy_summary.csv`. |
| `E4_homotopy_summary.csv` | CSV | Aggregierte Ergebnisse über \(\theta\) (z. B. 0.00, 0.25, 0.50, 0.75, 1.00). Enthält Mess‑ & Fit‑Kennzahlen. |
| `E4_ds_theta.png` | PNG | Plot: **\(\hat d_s(\theta)\)** (Spektral‑Dimensions‑Proxy) vs. Homotopie‑Parameter. |
| `E4_vstar_theta.png` | PNG | Plot: **\(\hat v^*(\theta)\)** (Frontgeschwindigkeit) vs. Homotopie‑Parameter. |

## Methode (Kurzfassung)

1. **ST‑Subgraph**: Vom ST‑Graph (Level) wird eine **Kugel** (Radius) um ein Zentrum extrahiert.
2. **Homotopie A(θ)**: Mischung aus **ST‑Kanten** und einer **Spannbaum**‑Topologie; Reihen‑normiert ⇒ vergleichbare Zeitskalen. Daraus **L(θ)=D−A**.
3. **\(\hat d_s\)**: Heat‑Kernel‑Diagonale \(p_t(x,x)=\mathrm{Tr}(e^{-tL})/|V|\) im mittleren \(t\)‑Fenster per Log–Log‑Fit ⇒ Steigung \(-d_s/2\).
4. **\(\hat v^*\)**: **CTQW** (Hamiltonian \(H=-A\)); Echo‑Schwelle \(\varepsilon\) pro Abstandsschale ⇒ Regression **Radius ≈ \(v^*\)·Zeit**.

**Standard‑Parameter (aus `E4_simulation.py`):** `level=6`, `radius=7`, `thetas=0.0,0.25,0.5,0.75,1.0`, `m=22`, `T=2800`, `eps=4e-3`

## Ergebnisse (aus `E4_homotopy_summary.csv`)

| theta | n_sub | ds_est | ds_r2 | vstar_est | vstar_r2 |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 1.750e+02 | 1.6685 | 0.9991 | 0.5114 | 0.9364 |
| 0.2500 | 1.750e+02 | 1.6465 | 0.9975 | 0.5108 | 0.9389 |
| 0.5000 | 1.750e+02 | 1.5379 | 0.9931 | 0.5248 | 0.9468 |
| 0.7500 | 1.750e+02 | 1.2824 | 0.9873 | 0.5449 | 0.9588 |
| 1.0000 | 1.750e+02 | 0.8029 | 0.9889 | 0.5512 | 0.9650 |

## Abbildungen

- `E4_ds_theta.png`: **\(\hat d_s(\theta)\)** — in den beiliegenden Ergebnissen fällt \(\hat d_s\) monoton mit \(\theta\).
- `E4_vstar_theta.png`: **\(\hat v^*(\theta)\)** — in den Daten steigt \(\hat v^*\) über \(\theta\).

## Nutzung & Reproduzierbarkeit

1) **Python‑Umgebung (Beispiel)**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib
```
2) **Simulation ausführen**
```bash
python E4_simulation.py
# erzeugt/aktualisiert: E4_homotopy_summary.csv und optional die Plots
```
3) **Plots aus CSV erzeugen** (optional)
```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('E4_homotopy_summary.csv')
plt.figure(); plt.plot(df['theta'], df['ds_est'], marker='o');
plt.title('E4: Spektral-Dimension (Proxy) vs. Homotopie-Parameter');
plt.xlabel(r'$\\theta$'); plt.ylabel(r'$\\hat d_s(\\theta)$');
plt.grid(True, linestyle='--', alpha=0.4); plt.savefig('E4_ds_theta.png', dpi=180)

plt.figure(); plt.plot(df['theta'], df['vstar_est'], marker='o');
plt.title('E4: Frontgeschwindigkeit vs. Homotopie-Parameter');
plt.xlabel(r'$\\theta$'); plt.ylabel(r'$\\hat v^*(\\theta)$');
plt.grid(True, linestyle='--', alpha=0.4); plt.savefig('E4_vstar_theta.png', dpi=180)
```
## Lizenz

- **Code** (`E4_simulation.py`): **MIT License**.
- **Nicht‑Code** (`*.csv`, `*.png`): **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

> © 2025 antaris — **Code:** MIT; **Daten & Abbildungen:** CC BY 4.0.


