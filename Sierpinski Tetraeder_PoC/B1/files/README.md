# B1 – Partial Trace auf ST-Graph

## Pfad
`antaris82.github.io/Sierpinski Tetraeder_PoC/B1/files/`  
**owner:** antaris82

---

## Kurzbeschreibung
In diesem Ordner werden **Teilspur-Operationen** (Partial Trace) auf dem **Sierpiński-Tetraeder-Graph** (ST-Graph) untersucht.  
Ziel: Konstruktion realistischer reduzierter Dichtematrizen durch explizite Teilspur über Umgebungsfreiheitsgrade, inkl. Coarse-Graining und numerischer Validierung.  
Ergebnisse werden durch **Python-Skripte**, **CSV-Daten**, **Berichte** und **animierte GIFs** dokumentiert.

---

## Dateien

### Code
- **B1_v1_partial trace_partial trace on ST-Graph.py**  
  Hauptskript für Level-4 ST-Graph, Teilspur-Berechnung, Coarse-Graining, GIF-Ausgabe.  
  - Funktionen: `build_graph_by_addresses`, `L_A_alpha`, `reduced_density_via_partial_trace`, `energy_from_spectrum`, `entropy_from_p`, `purity_from_p`.  
  - Kernaxiome:  
    \[
    \rho_{\mathrm{red}} = \operatorname{Tr}_{E}\bigl(e^{-\beta (L \oplus H_E)}\bigr) \; / \; Z
    \]
    mit thermischem Gewicht, ST-Graph Laplace-Operator \(L\), Environment-Spektrum \(H_E\).  
  - Implementiert Approximant-Verkleinerung (Level-4 → Level-0).  

- **B1_v2_check.py**  
  Validierungsskript (Checks 1–3). Prüft Symmetrie, Spurtreue und Energie/Kanonische Konsistenz.

- **B1_v1_partial trace_code_mapping.md**  
  Übersicht der Codefunktionen, Parameter (z. B. `BETA=3.0`, `LEVEL_FINE=4`), Struktur der Matrizen.

### Daten
- **B1_v2_check_alpha_observables.csv**  
  CSV mit Observablen für verschiedene α-Werte (0.00–1.00).  

- **B1_v2_check_checks_report.txt**  
  Validierungsprotokoll. Ergebnisse:  
  - [Check1] Symmetrie-Bewahrung und minimale Eigenwerte ~ numerisch null.  
  - [Check2] Frobenius-Abstand \(||\mathrm{Tr}_E \rho_{\text{tot}} - \rho(L)||_F \approx 4.2 \times 10^{-16}\).  
  - [Check3] Energie- und Varianzprüfung konsistent (Fehler ~ \(10^{-10}\)).  
  → Bestätigung der korrekten Implementierung.  

### Visualisierungen
- **B1_v1_partial trace_graph_and_coarsening.gif**  
  Animierte Darstellung des ST-Graphen und des Coarse-Graining-Prozesses.  

- **B1_v1_partial trace_observables.gif**  
  Zeitabhängige Entwicklung von Observablen (z. B. Entropie, Energie).  

- **B1_v1_partial trace_static_density_urgraph.gif**  
  Thermische Dichtematrix des Urgraphen (statisch).  

- **B1_v1_partial trace_density_approximant.gif**  
  Vergleich: Approximantendichte vs. feiner Graph.  

---

## Unterordner
- `../files/` (dieser Ordner) → enthält alle Skripte, Daten & GIFs.  
- keine weiteren Unterordner.

---

## Ergebnisse
- Explizite Konstruktion von **reduzierten Dichtematrizen** via Teilspur, ohne klassische Input-Postulate.  
- Konsistenzprüfungen (Check1–3) zeigen numerisch stabile Ergebnisse (Fehler < \(10^{-10}\)).  
- Visualisierungen verdeutlichen Coarse-Graining und thermische Reduktion.

---

## Akzeptanzkriterien
- [x] Teilspur-Implementierung mathematisch korrekt (Check1–3 bestanden).  
- [x] Reproduzierbare Resultate bei fester Seedwahl (`np.random.seed(7)`).  
- [x] Alle Observablen (Energie, Entropie, Reinheit) konsistent mit kanonischer Thermodynamik.  
- [x] GIF-Visualisierungen erzeugt.

---

## Reproduzierbarkeit
1. Python ≥ 3.10, mit `numpy`, `matplotlib`, `Pillow`.  
2. Skript `B1_v1_partial trace_partial trace on ST-Graph.py` ausführen.  
3. `B1_v2_check.py` starten → erzeugt Report & CSV.  
4. Ergebnisse sollten bis auf Rundungsfehler mit den vorhandenen Dateien übereinstimmen.

---

## Offene Punkte / To-Do
- Erweiterung auf höhere Level (m > 4).  
- Analyse der Semi-Klassischen Grenze (große β).  
- Systematische Untersuchung verschiedener Environment-Spektren.  
- Integration in globalen ST-Graph-Formalismus (A1–E4).  

---

## Validierungsstatus
| Kriterium | Beschreibung                          | Status |
|-----------|---------------------------------------|--------|
| K1        | Symmetrieerhaltung (L_A)              | 🟢 |
| K2        | Spurtreue (Tr_E ρ_tot = ρ(L))         | 🟢 |
| K3        | Energie/Kanonische Konsistenz         | 🟢 |
| K4        | GIF-/Output-Generierung               | 🟢 |
| K5        | Dokumentation (Mapping, Report)       | 🟢 |

---

## Allgemeine Hinweise
- Alle numerischen Abweichungen liegen innerhalb der erwarteten Toleranz (≤ 1e-10).  
- Visualisierungen dienen der Illustration; die exakten Werte sind in CSV/Reports gespeichert.  
- Seed ist festgelegt (`np.random.seed(7)`) → deterministisch reproduzierbar.

---

## Lizenz
© 2025 antaris — Code: MIT;  
Daten/Abbildungen/Texte (inkl. PDFs, CSV, PNG, GIF): **CC BY 4.0**

---
