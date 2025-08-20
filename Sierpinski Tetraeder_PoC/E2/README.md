# E2 — Lieb–Robinson & Frontzeiten

Ordner für die **E2-Studie**: Mathematischer Beweis und Simulationen zur Ableitung von Frontzeiten, maximaler Gruppengeschwindigkeit und emergenten Lichtkegeln auf dem ST-Graph.

## Pfad
`antaris82.github.io/Sierpinski Tetraeder_PoC/E2/`

**Owner:** antaris82

---

## Dateien & Kurzbeschreibung

- `E2_proof.pdf` — Formaler Beweis: **Lieb–Robinson ⇒ Frontzeit und maximale Geschwindigkeit**.  
  Enthält Definitionen, Sätze und Beweise zur Herleitung der Frontzeit-Untergrenze, der maximalen Gruppengeschwindigkeit \( v^* \leq v_{\mathrm{LR}} \) sowie des emergenten Lichtkegels für ST-Graphen.

### Unterordner
- [`files/`](./files) — Simulationen, numerische Experimente und Auswertungen zu OTOCs, Crossing-Analysen und Amplitudenfronten (Level 4–6).  
  → Details siehe [README im Unterordner](./files/README.md).

---

## Axiome & Kernpunkte

- **(A1)** Lieb–Robinson-Bounds gelten auf ST-Graph-Approximanten (lokal endlich, uniform beschränkter Grad).  
- **(A2)** Aus dem Bound folgt eine **Untergrenze für die Frontzeit** \( t_\varepsilon(d) \).  
- **(A3)** Daraus ergibt sich eine **maximale Gruppengeschwindigkeit** \( v^* \leq v_{\mathrm{LR}} \).  
- **(A4)** Emergenz eines **nahezu linearen Lichtkegels** in der Dynamik auf ST-Graphen.  
- **(A5)** Ergänzende numerische Analysen (E2/files) bestätigen Crossing- und OTOC-Strukturen.

---

## Ergebnisse

- Formale Herleitung: **Frontzeit ≥ linear in Distanz** mit Korrekturgliedern.  
- Beweis: **Maximale Geschwindigkeit beschränkt durch \( v_{\mathrm{LR}} \)**.  
- Korollar: **Emergenter Lichtkegel** auch auf fraktalen ST-Graphen.  
- Robustheit: Resultate bleiben gültig für exponentiell abfallende Interaktionen und Lindblad-Dynamik.  
- Numerische Simulationen (Unterordner `files/`) konsistent mit den theoretischen Bounds.

---

## Akzeptanzkriterien

- (K1) Formaler Beweis der LR-Folgerungen auf ST-Graphen.  
- (K2) Konsistenz mit numerischen Simulationen (Crossings, OTOCs).  
- (K3) Reproduzierbarkeit durch Dokumentation in CSV/JSON/PNG im Unterordner.  

**Validierungsstatus:**  
| Kriterium | Status |
|-----------|--------|
| K1 | 🟢 |
| K2 | 🟡 |
| K3 | 🟢 |

---

## Reproduzierbarkeit

1. **Theorie:**  
   - E2_proof.pdf nachvollziehen (Beweise Schritt für Schritt).  
   - Literatur [Lieb & Robinson 1972], [Nachtergaele & Sims 2006], [Bravyi–Hastings–Verstraete 2006] etc. prüfen.  

2. **Simulation:**  
   - Python-Skripte im Unterordner `files/` ausführen.  
   - CSV/JSON-Daten mit erzeugten Outputs vergleichen.  
   - Ergebnisse mit PNG-Plots abgleichen.  

---

## Offene Punkte / To-Do

- Übertragung der Bounds auf **längere Reichweiten** (Power-Law-Interaktionen, \(1/r^\alpha\)).  
- Vergleich analytischer Frontzeit-Bounds mit den numerischen Crossing-Daten.  
- Tests für höhere Level (≥7) zur Stabilität der Ergebnisse.  

---

## Lizenz

- **Code** (`*.py` in `files/`): MIT.  
- **Nicht-Code** (PDF, CSV, JSON, PNG, TXT): CC BY 4.0.  

© 2025 antaris — Code: MIT; Daten/Abbildungen/Texte: CC BY 4.0.
