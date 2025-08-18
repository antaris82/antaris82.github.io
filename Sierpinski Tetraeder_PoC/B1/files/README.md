# B1 / files — README

Dieses Verzeichnis bündelt **Skripte, Daten und Abbildungen** zu *B1* (ST‑Graph / Partial Trace & Checks).
Die Dateien stammen aus dem Archiv `files.rar` (Stand: 18.08.2025).

---

## Inhalt (Dateiübersicht)

| Datei | Typ | Zweck / Beschreibung |
|---|---|---|
| `B1_v1_partial trace_partial trace on ST-Graph.py` | Python‑Skript | Referenz‑Implementation des **Partial Trace** auf dem ST‑Graph inkl. Generierung der v1‑GIFs/Artefakte. |
| `B1_v1_partial trace_code_mapping.md` | Doku (Markdown) | Mapping Code ↔︎ Fachbegriffe / Operatoren / Observablen. |
| `B1_v1_partial trace_density_approximant.gif` | GIF | Dichte‑Approximant (v1). |
| `B1_v1_partial trace_graph_and_coarsening.gif` | GIF | Graph & Coarse‑Graining (v1). |
| `B1_v1_partial trace_observables.gif` | GIF | Observablenentwicklung (v1). |
| `B1_v1_partial trace_static_density_urgraph.gif` | GIF | Statische Dichte auf dem Ur‑Graph (v1). |
| `B1_v2_check run 1.py` | Python‑Skript | **Check‑Run 1**; erzeugt CSV + Report. |
| `B1_v2_check run 1_alpha_observables.csv` | CSV | Observablen über **α** (Run 1). |
| `B1_v2_check run 1_checks_report.txt` | Text | Kurzbericht (Run 1). |
| `B1_v3_check run 2_checks.py` | Python‑Skript | **Check‑Run 2**; erzeugt CSV + Report. |
| `B1_v3_check run 2_alpha_observables.csv` | CSV | Observablen über **α** (Run 2). |
| `B1_v3_check run 2_checks_report.txt` | Text | Kurzbericht (Run 2). |
| `B1_v4_check run 3_make_two_tables.py` | Python‑Skript | Aus den Check‑Runs **zwei Tabellen** generieren (Aggregation/Comparison). |

> **Hinweis zu Dateinamen mit Leerzeichen:** Beim Ausführen bitte Anführungszeichen verwenden (siehe unten).

---

## Quick‑Start (Reproduktion)

1) **Python 3.10+** empfehlen. Übliche Pakete (je nach Skript): `numpy`, `pandas`, `scipy`, `matplotlib`, ggf. `networkx`, `imageio` oder `Pillow` für GIFs.  
2) Ausführung (Beispiele, Pfad ggf. anpassen):

```bash
python "B1_v1_partial trace_partial trace on ST-Graph.py"
python "B1_v2_check run 1.py"
python "B1_v3_check run 2_checks.py"
python "B1_v4_check run 3_make_two_tables.py"
```

**Erwartete Artefakte**
- v1‑Skript erzeugt die im Paket enthaltenen GIFs (o. ä.).
- v2/v3 erzeugen je eine `*_alpha_observables.csv` sowie `*_checks_report.txt`.
- v4 aggregiert die Check‑Runs zu **zwei Tabellen** (Format je nach Implementierung, typ. CSV/Markdown/LaTeX).

---

## Datenformate

### \(\alpha\)‑Observablen (CSV)

Die beiden Dateien
- `B1_v2_check run 1_alpha_observables.csv`
- `B1_v3_check run 2_alpha_observables.csv`

enthalten die über einen α‑Sweep ausgewerteten Observablen.  
**Typische Spalten** (richtige Bezeichnungen im CSV‑Header nachsehen):
- `alpha` — Skalenparameter/Deformationsgrad,
- Kenngrößen des reduzierten Zustands (z. B. `trace`, `purity = Tr(ρ_A^2)`, `S = −Tr(ρ_A log ρ_A)`),
- ggf. Stabilitäts-/Constraint‑Checks, Metriken/Fehler, Laufzeit usw.

Die zugehörigen Reports `*_checks_report.txt` fassen die wichtigsten Messwerte und Checks pro Run zusammen.

---

## Mathematischer Kontext (Kurz)

**Partial Trace** (Subsystem \(B\) wird getraced):

\[
\mathrm{Tr}_B(\rho_{AB}) \;=\; \sum_j \bigl(\mathbb{I}_A \otimes \langle j|\bigr)\,\rho_{AB}\,\bigl(\mathbb{I}_A \otimes |j\rangle\bigr).
\]


**Beispielhafte Observablen-Abbildung** über α:

\[\alpha \;\mapsto\; \mathcal{O}(\alpha)\;=\;\Big(S(\rho_A),\;\mathrm{Tr}(\rho_A^2),\;\dots\Big).\]


**Lieb–Robinson‑artige Schranke** (Motivation für effektive Kegel/Kausalität auf Gittern/Netzwerken):
//
//
\[\bigl\|[\alpha_t(A),B]\bigr\| \;\le\; C\,\|A\|\,\|B\|\,\exp\!\Big(-\mu\,\big[d(X,Y)-v_{\mathrm{LR}}\,t\big]\Big).\]
/

Diese Größen dienen als **konsistente, nicht‑klassische Diagnostik** der Dynamik und der Reduktionen auf dem ST‑Graph.

---

## Provenienz (Zeitstempel, Größe)

Aus dem Archiv-Listing (UTC±0 notiert, lokale Zeitzone Berlin: UTC+2):

- `B1_v1_partial trace_partial trace on ST-Graph.py` — 11 744 B — 2025‑08‑18 12:29:29  
- `B1_v1_partial trace_code_mapping.md` — 5 466 B — 2025‑08‑18 12:47:58  
- `B1_v1_partial trace_density_approximant.gif` — 176 013 B — 2025‑08‑18 12:28:40  
- `B1_v1_partial trace_graph_and_coarsening.gif` — 49 160 B — 2025‑08‑18 12:28:52  
- `B1_v1_partial trace_observables.gif` — 122 686 B — 2025‑08‑18 12:29:07  
- `B1_v1_partial trace_static_density_urgraph.gif` — 205 049 B — 2025‑08‑18 12:29:49  
- `B1_v2_check run 1.py` — 6 027 B — 2025‑08‑18 12:56:49  
- `B1_v2_check run 1_alpha_observables.csv` — 324 B — 2025‑08‑18 12:56:58  
- `B1_v2_check run 1_checks_report.txt` — 585 B — 2025‑08‑18 12:56:53  
- `B1_v3_check run 2_checks.py` — 6 027 B — 2025‑08‑18 13:07:45  
- `B1_v3_check run 2_alpha_observables.csv` — 324 B — 2025‑08‑18 13:08:04  
- `B1_v3_check run 2_checks_report.txt` — 585 B — 2025‑08‑18 13:07:49  
- `B1_v4_check run 3_make_two_tables.py` — 5 761 B — 2025‑08‑18 13:36:06  

---

## Reproduzierbarkeit & Hinweise

- **Determinismus:** Wenn Zufall im Spiel ist, Seeds setzen (z. B. `numpy.random.seed(...)`), damit CSV/Plots reproduzierbar sind.  
- **Performance:** GIF‑Erzeugung kann speicher‑/zeitintensiv sein; ggf. Bildgrößen/Frames reduzieren.  
- **Pfad‑Spezifika:** Leerzeichen in Dateinamen korrekt quoten.  
- **Weiterverarbeitung:** v4‑Skript kann die beiden α‑CSV automatisch einlesen und zu Vergleichstabellen (bspw. LaTeX) zusammenführen.

---

## Lizenz / Attribution

Sofern nicht anders vermerkt, © Autor*in des Projekts „Sierpinski Tetraeder PoC / B1“. Bitte Projekthaupt‑README beachten.
