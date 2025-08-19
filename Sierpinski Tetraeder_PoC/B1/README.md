# B1 – Formale Ergebnisse und Artefakte (ST-Graph, Teilspur, Thermodynamik)

Dieses Verzeichnis dokumentiert die Ergebnisse und Beweise zu **B1** im Rahmen des *Sierpinski-Tetraeder-PoC*.  
Zentrale Inhalte sind die formale Ausarbeitung (PDF), begleitende Codes und die erzeugten Daten/Animationen im Unterordner [`files/`](./files).

---

## 📄 Inhalt

### Hauptdokumente
- **B1_formal.pdf**  
  Enthält die vollständigen mathematischen Beweise und Ergebnisse zu:
  - Definition und Eigenschaften des Approximanten  
    \
    L_A(\alpha) = (1-\alpha)L + \alpha L_{\text{lift}}
    \
  - Positivität und Kerneigenschaft (Lemma 1)  
  - Kronecker-Summenoperator und Spektralzerlegung (Proposition 2)  
  - Teilspur-Reduktion des Gibbs-Zustands (Satz 3)  
  - Thermodynamische Identitäten (Proposition 4)  
  - Stetigkeit der Observablen in α (Proposition 5)  
  - Numerische Verifikation (Checks 1–3, Tabellen mit Observablen über α)  

### Unterordner
- [`files/`](./files)  
  Enthält die zugehörigen Python-Skripte, CSV-Daten, Berichte sowie animierte GIFs (Simulationen).  
  Beispiele:
  - `B1_v2_check.py` – Kernskript zur numerischen Überprüfung der Aussagen.  
  - `B1_v1_partial trace_partial trace on ST-Graph.py` – Simulation der Teilspur auf dem ST-Graph.  
  - `B1_v2_check_alpha_observables.csv` – numerische Observablen (Energie, Entropie, Purity) über α.  
  - `B1_v2_check_checks_report.txt` – numerischer Report (Symmetrie, Positivität, Teilspur-Gleichheit, Thermo-Identitäten).  
  - mehrere GIF-Animationen (z. B. dynamische Dichte und Graph-Coarsening).

---

## 🔬 Zusammenfassung der Ergebnisse

- **Reduktion:** Für jedes endliche Environment H_E gilt
  \
  \mathrm{Tr}_E \!\left(e^{-\beta(L\otimes 1 + 1 \otimes H_E)}\right) \propto e^{-\beta L},
  \
  sodass die reduzierte Dichte nach Normierung genau dem Gibbs-Zustand von L entspricht.

- **Thermodynamik:**  
  \
  E(\beta) = -\partial_\beta \log Z(\beta), 
  \quad 
  \partial^2_\beta \log Z(\beta) = \mathrm{Var}_\rho(L) \geq 0 .
  \

- **Stetigkeit:** Die thermischen Observablen hängen stetig von \(\alpha\) ab.  

- **Numerische Bestätigung:** Alle Kernaussagen (Symmetrie, Positivität, Teilspur-Gleichheit, Thermo-Identitäten) wurden bis zu Toleranzen von 10^{-10}–10^{-16} bestätigt.

---

## 📊 Beispieltabellen (aus B1_formal.pdf)

**Thermische Observablen über den Approximanten L_A(\alpha):**

| α       | Energie E | Entropie S | Purity P |
|---------|-----------|------------|----------|
| 0.000000 | 0.244396 | 1.939091 | 0.186146 |
| 0.250000 | 0.275970 | 2.209655 | 0.145043 |
| 0.500000 | 0.313159 | 2.623000 | 0.097992 |
| 0.750000 | 0.371128 | 3.443204 | 0.045853 |
| 1.000000 | 0.003005 | 4.602681 | 0.010043 |

---

## 📜 Lizenz

- **Code** (insb. in `./files/`): MIT License  
- **Nicht-Code** (z. B. PDFs, CSV/PNG/GIF): Creative Commons Attribution 4.0 International (CC BY 4.0)  

© 2025 antaris — Code: MIT; Daten/Abbildungen/Texte (inkl. PDFs): CC BY 4.0.
