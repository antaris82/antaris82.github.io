# E3 / files — Proof‑of‑Concept (PoC) für den ST‑Graph
Dieser Unterordner enthält die **Artefakte für E3** im ST‑Graph‑PoC. Die unten aufgeführte Liste wurde **direkt aus dem bereitgestellten Archiv** rekonstruiert (Dateinamen & Größen).
- Quelle: `files.rar`
- Stand: 2025-08-18 12:22 UTC
---
## Dateiübersicht
| Datei | Typ | Größe | Kurzbeschreibung |
|---|---|---:|---|
| `E2_run 1_Level 4.py` | PY | 0.0 KB | Python‑Skript für ST‑Level 4 (Run 1): erzeugt First‑Crossing‑Daten/Plots. |
| `E2_run 1_results.json` | JSON | 0.0 KB | JSON‑Metadaten/Parameter/Ergebniszusammenfassung. |
| `E2_run 1_ST L4 Ball First Crossing.png` | PNG | 0.1 KB | Plot First‑Crossing (Frontpositionen/Zeiten). |
| `E2_run 1_STL4_firstcross.csv` | CSV | 87 B | CSV‑Messwerte First‑Crossing (Frontzeiten/Schwellen). |
| `E2_run 2_Ergebnis.txt` | TXT | 0.0 KB | Kurzbericht/Ergebnis‑Notizen. |
| `E2_run 2_Level 5.py` | PY | 0.0 KB | Python‑Skript für ST‑Level 5 (Run 2): erzeugt Targets/Fits/Prüfungen. |
| `E2_run 2_plot.png` | PNG | 0.1 KB | Übersichts‑/Vergleichsplot. |
| `E2_run 2_STL5_eps0p02_alltargets.csv` | CSV | 560 B | CSV‑Sammlung „alle Targets“ (Referenzen/Kalibrierung). |
| `E2_run 2_STL5_hardcheck.json` | JSON | 0.0 KB | JSON‑Hard‑Check (Konsistenz/Toleranzprüfung). |
| `E2_run 2_STL5_J1_fits.csv` | CSV | 366 B | CSV‑Ergebnisse zu Kurvenanpassungen (Fits). |
| `E2_run 3_Ergebnisse.txt` | TXT | 0.0 KB | Kurzbericht/Ergebnis‑Notizen. |
| `E2_run 3_Level 6.py` | PY | 0.0 KB | Python‑Skript für ST‑Level 6 (Run 3): erzeugt Amplitude/OTOC‑Daten/Plots. |
| `E2_run 3_ST Level-6 ball — amplitude front.png` | PNG | 0.1 KB | Plot Amplitude‑Front (Level‑6, Setup „ball“). |
| `E2_run 3_ST Level-6 ball — OTOC front.png` | PNG | 0.1 KB | Plot OTOC‑Front (Level‑6, Setup „ball“). |
| `E2_run 3_STL5_hardcheck.json` | JSON | 0.0 KB | JSON‑Hard‑Check (Konsistenz/Toleranzprüfung). |
| `E2_run 3_STL6_amp_otoc.json` | JSON | 0.0 KB | JSON‑Konsolidierung Amplitude+OTOC. |
| `E2_run 3_STL6_eps0p02_amp_targets.csv` | CSV | 956 B | CSV‑Ziel‑/Messdaten Amplitude. |
| `E2_run 3_STL6_eps1e-3_otoc_targets.csv` | CSV | 962 B | CSV‑Ziel‑/Messdaten OTOC. |
| `E2_run 1_Chain First Crossing.png` | PNG | 0.1 KB | Plot First‑Crossing (Frontpositionen/Zeiten). |
| `E2_run 1_chain_firstcross.csv` | CSV | 147 B | CSV‑Messwerte First‑Crossing (Frontzeiten/Schwellen). |
| `E2_run 1_Ergebnis.txt` | TXT | 0.0 KB | Kurzbericht/Ergebnis‑Notizen. |

---
## Nutzung & Reproduzierbarkeit
1) **Python‑Umgebung (Beispiel)**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # falls vorhanden
```
2) **Läufe starten**
```bash
python "E2_run 1_Level 4.py"
python "E2_run 2_Level 5.py"
python "E2_run 3_Level 6.py"
```
3) **Parameter & Seeds**
Bitte in den Skripten dokumentieren (z. B. `numpy.random.seed(...)`, Konfiguration in `cfg.py`/`config.yaml`).

---
## Lizenz
- **Code** (`*.py`): **MIT License** (Namensnennung per Copyright‑Hinweis; freie Nutzung/Weitergabe/Änderung).
- **Nicht‑Code** (`*.csv`, `*.json`, `*.png`, `*.txt`): **Creative Commons Attribution 4.0 (CC BY 4.0)**.

> © 2025 antaris — **Code:** MIT; **Daten/Abbildungen/Texte:** CC BY 4.0.

---
## Zitation
> antaris (2025): *ST‑Graph PoC — E3 / files*. GitHub‑Repo `antaris82.github.io`, Ordner `Sierpinski Tetraeder_PoC/E3/files/`. **Code:** MIT. **Daten & Abbildungen:** CC BY 4.0.

