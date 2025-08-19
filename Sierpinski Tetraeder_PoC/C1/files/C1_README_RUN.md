# ST_Simulations_PoC_v1

Minimal, Jupyter‑freundliche Simulationen für den **Sierpiński‑Tetraeder (ST) Ur‑Graph** und direkte Vergleiche zu regulären Gittern \(\mathbb Z^d\).

## Inhalt
- `stlib/graph_st.py` — Konstruktion des ST‑Graphen (Approximant Level *m*) + 3D‑Koordinaten + Randknoten.
- `stlib/graphs_regular.py` — 2D/3D‑Gittergraphen (periodisch oder mit Rand).
- `stlib/linops.py` — Laplace‑Operator (kombinatorisch oder normalisiert), Distanz‑BFS, Hilfsfunktionen.
- `stlib/heat.py` — Heat‑Trace‑Schätzer \(\overline p_t = |V|^{-1}\operatorname{tr}(e^{-tL})\) via Hutchinson + `expm_multiply`.
- `stlib/quantum.py` — Einteilchen‑Zeitentwicklung \(\psi(t)=e^{-\mathrm{i}tH}\psi_0\) + LR‑Front‑Metriken.
- `stlib/gkls.py` — **GKLS‑Dephasierung** (Markovian, ortsdiagonal): effiziente DGL für Dichtematrix mit Off‑Diagonal‑Dämpfung.
- `stlib/dtn.py` — **Dirichlet‑to‑Neumann (DtN)** für Graphen + gezielte Tiefen‑Perturbationen.
- `demo_ST_vs_Zd.py` — End‑to‑End‑Script (Plots & CSVs in `out/`).

## Schnelle Nutzung (lokal)
```bash
python -m pip install numpy scipy matplotlib
python demo_ST_vs_Zd.py
```
oder in Jupyter:
```python
%run demo_ST_vs_Zd.py
```

## Hinweise
- Default‑Parameter sind klein (schnell). Für „beeindruckende“ Visuals auf *m=5* anheben und `n_probes`/`t_grid` vergrößern.
- GKLS skaliert \(\mathcal O(N^2)\). Für ST‑Level > 4 bitte `N_max_gkls` begrenzen.

## Lizenz
Code: MIT. Nicht‑Code (z. B. CSV/PNG): CC BY 4.0. © 2025 antaris — Code: MIT; Daten/Abbildungen/Texte: CC BY 4.0.
