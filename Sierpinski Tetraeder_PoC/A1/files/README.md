
# A1 — Assets & Metriken (files)

> **Pfad:** `Sierpinski Tetraeder_PoC/A1/files/` • **Owner:** antaris82  
> **Kurzbeschreibung:** Datensätze, Skripte und Visualisierungen zum Sierpinski‑Tetraeder (ST‑Ur‑Graph) für A1. Enthält Metriken (Entropie/MI, Schnitt‑Metriken), statische/rotierende Ansichten und die Generierungs‑Pipeline.  
> **Math‑Hinweis:** Inline \( … \), Display \[ … \].

---

## 🔗 Schnellzugriff
- 🐍 **Pipeline‑Skript:** `ST.py` — erzeugt CSV/PNG/GIF‑Artefakte
- 📈 **Plot:** `levels_S_MI.png` — S(ℓ), MI(ℓ:Rest)
- 🖼 **Static 3D:** `static_colored_obs_exclusive-scaled.png`
- 🎞 **Rotationen:** `static_colored_obs_exclusive_rotate.gif`, `ST_rotating_forever.gif`
- 🗂 **Metriken (CSV):** `levels_observables.csv`, `regions_observables_exclusive.csv`, `pairs_observables_exclusive.csv`

---

## 1) Ziel & Kontext
A1 bündelt **operative Artefakte** zur Analyse des ST‑Ur‑Graphen:  
(i) **Entropische Größen** pro Konstruktions‑Layer, (ii) **regionale Observablen** und Schnitt‑Metriken (exklusive Partition **RED/YELLOW/GREEN**),  
(iii) **Visualisierungen** (statisch/rotierend) als Referenz für Folgeordner.

## 2) Axiome & Kernergebnisse
**Axiome (frei‑fermionisches PoC).**
A1 arbeitet mit dem Graph‑Laplacian \(L=D-A\) (lokale Kopplungen), dem Einteilchen‑Hamiltonian \(H=L\) und dem Grundzustand bei Füllung \(\nu=1/2\).  
Aus dem Spektralzerfall \(H=U\,\mathrm{{diag}}(\varepsilon)\,U^\top\) folgt die **Korrelationsmatrix** \(C=U_{{\text{{occ}}}}U_{{\text{{occ}}}}^\top\).

**Entropie & MI.**
Für einen Indexsatz \(A\):  
\[ S(A)\;=\;-\,\mathrm{{tr}}\big[ C_A\log C_A + (\mathbf 1 - C_A)\log(\mathbf 1 - C_A) \big], \qquad
\mathrm{{MI}}(A{:}\mathrm{{Rest}}) = 2\,S(A). \]
Für disjunkte \(A,B\):  
\[ \mathrm{{MI}}(A{:}B)=S(A)+S(B)-S(A\cup B). \]

**Schnitt‑Metriken (exklusiv).**
Für Partition **RED/YELLOW/GREEN**: **cut‑edges**, **\langle|C|\rangle_\text{{cross}}**, **d\_\min** (BFS).  
Die mitgelieferten CSVs dokumentieren diese Größen reproduzierbar.

## 3) Methoden / Formalismus
- **Graph:** ST‑Approximant auf Level \(L\) (Standard‑Iterationen).  
- **Hamiltonian:** \(H=L\) (resistive/CTQW‑nahe Heuristik).  
- **Regionen:** exklusive Zuweisung (Präfix‑Zellen: RED=L4, YELLOW=L2, GREEN=L0).  
- **Metriken:** S(A), MI(A{:}Rest), MI(A{:}B), cut‑edges(A,B), \langle|C|\rangle\_\text{{cross}}, d\_\min(A,B).

## 4) Datei‑ & Ordnerübersicht (mit Kurzbeschreibung)
| Pfad | Typ | Kurzbeschreibung |
|---|---|---|
| `./ST.py` | PY | Python‑Skript (Pipeline) zur Generierung der CSV‑Metriken sowie der PNG/GIF‑Artefakte; Level/Seeds parametrisierbar. |
| `./levels_S_MI.png` | PNG | Linienplot S(ℓ) & MI(ℓ:Rest) für L=4 (Beispiel). |
| `./static_colored_obs_exclusive-scaled.png` | PNG | Skalierte statische 3D‑Ansicht (kompaktere Datei). |
| `./static_colored_obs_exclusive_rotate.gif` | GIF | Rotierendes 3D‑GIF (exklusive Färbung + Wireframe). |
| `./ST_rotating_forever.gif` | GIF | Rotierendes ST‑GIF (dauerhaft). |
| `./levels_observables.csv` | CSV | Layer‑weise Entropie und Mutual Information (MI) relativ zum Rest; aus Korrelationsmatrix des Grundzustands.  (5 Zeilen × 4 Spalten; Spalten: ell, count, S, MI_ell_Rest) |
| `./pairs_observables_exclusive.csv` | CSV | Paar‑Metriken für exklusive Regionen (RED/YELLOW/GREEN): I(A:B), cut‑edges, ⟨|C|⟩_cross, d_min.  (3 Zeilen × 6 Spalten; Spalten: A, B, MI_AB, cut_edges, mean_abs_C_cross, d_min) |
| `./regions_observables_exclusive.csv` | CSV | Regionale Metriken: |A|, S(A), MI(A:Rest), ⟨|C|⟩_intra.  (3 Zeilen × 5 Spalten; Spalten: region, size, S, MI_A_Rest, mean_abs_C_intra) |
| `./README.md` | MD | Diese README (A1/files). |

> **Hinweis:** Unterordner (falls vorhanden) bringen ein eigenes README mit; beim Indizieren werden daraus nur die **ersten 300 Zeichen** gelesen.

## 5) Akzeptanzkriterien
- **K1 (Artefakte vorhanden):** Alle drei CSVs und mind. ein PNG + ein GIF im Ordner.
- **K2 (CSV‑Schema):** `levels_observables.csv` enthält mindestens `level, n, S, MI_with_rest`.  
  `regions_observables_exclusive.csv` enthält `region, size, S, MI_with_rest, mean_intra_absC`.  
  `pairs_observables_exclusive.csv` enthält `A, B, MI, cut_edges, mean_abs_crossC, dmin`.
- **K3 (Repro‑Konsistenz):** Ausführung von `ST.py` erzeugt identische Dateien (bis auf Zeitstempel/Skalierung).
- **K4 (Visualisierung):** PNG/GIF zeigen exklusive Färbung (GREEN wireframe, YELLOW/RED Kanten).

## 6) Reproduzierbarkeit (How‑To)
1. **Umgebung:** Python ≥3.10; `numpy`, `pandas`, `matplotlib`, `imageio`.  
2. **Ausführen:**  
   ```bash
   python ST.py
   ```
   oder (direkt) `python st_pipeline_metrics.py` (falls vorhanden).  
3. **Parameter:** Level `L`, Seeds/Toleranzen im Skript anpassen (Standard: `L=4`, Seed=99).  
4. **Validierung:** Prüfe CSV‑Schemas (K2) und dass `levels_S_MI.png` zwei Linien (S, MI) über \(\ell\) enthält.

## 7) Themenbezogene Informationen
- PoC‑Charakter (frei‑fermionisch) — austauschbar gegen andere lokale Dynamiken.  
- Exklusive Partition dient als **Testbench** für Schnitt‑Metriken auf ST‑Geometrien.

## 8) Unterordner (Struktur)
```
files/
├─ ST.py
├─ levels_S_MI.png
├─ static_colored_obs_exclusive-scaled.png
├─ static_colored_obs_exclusive_rotate.gif
├─ ST_rotating_forever.gif
├─ levels_observables.csv
├─ pairs_observables_exclusive.csv
├─ regions_observables_exclusive.csv
├─ README.md
```

## 9) Allgemeine Hinweise
- **LaTeX:** Inline \( … \), Display \[ … \]; kein `$$…$$`.  
- **Compiler für PDFs (falls genutzt):** LuaLaTeX/XeLaTeX bei `fontspec`.  
- **Pfade:** relative Links beibehalten.

## 10) Offene Punkte / To‑Do
- [ ] Level \(L\) und Präfix‑Definitionen als Parameter exposed machen.  
- [ ] Weitere Metriken: Spektrale Dimension, Heat‑Kernel‑Rückkehrprob.  
- [ ] CI‑Job: Artefakte generieren und CSV‑Schemas prüfen.

## 11) Validierungsstatus
| Kriterium | Status | Kommentar/Quelle |
|---|---|---|
| K1 | 🟢 | Dateien vorhanden |
| K2 | 🟢 | CSV‑Schemas gelesen |
| K3 | 🟡 | Re‑Run lokal erforderlich |
| K4 | 🟢 | PNG/GIF geprüft (visuell) |

## 12) Referenzen
(Nur falls keine separaten Biblios vorliegen)  
- Cramer et al.: Entanglement in free‑fermion systems; Peschel (2003).  
- Grundlegende ST‑Geometrie: Kigami (Analysis on Fractals).

## 13) Lizenz
Lizenz

    Code (insb. in ./files/): MIT License.
    Nicht‑Code (z. B. PDFs, CSV/PNG): Creative Commons Attribution 4.0 International (CC BY 4.0).

    © 2025 antaris — Code: MIT; Daten/Abbildungen/Texte (inkl. PDFs): CC BY 4.0.

## 14) Changelog
- **v1.0 (2025-08-19):** Erstausgabe für `A1/files` (Assets & Metriken).
