# E4 — Robustheit via Homotopie \(G(\theta)\): ST → Baum

Dieser Ordner dokumentiert den **E4-Schritt** im Sierpinski-Tetraeder-PoC.  
Ziel: **Stabilität der Kinematik** gegen Mikrodeformationen durch Homotopie von der ST-Geometrie (\(\theta=0\)) hin zu einer Baumstruktur (\(\theta=1\)).

## Inhalt

| Datei | Typ | Beschreibung |
|-------|-----|--------------|
| `E4_formal.pdf` | PDF | Formalisierung, Theorie & Akzeptanzkriterien (Lieb–Robinson-Bounds, Heat-Kernel-Vergleich Fraktal↔Baum). |
| `files/` | Ordner | Enthält Simulation (`E4_simulation.py`), CSV-Ergebnisse (`E4_homotopy_summary.csv`), Plots (`E4_ds_theta.png`, `E4_vstar_theta.png`) — siehe [README im files-Ordner](../files/README.md). |

## Methode

- **Graph-Konstruktion:** \(A(\theta) = \frac{(1-\theta)A_{ST}+\theta A_{tree}}{\|(1-\theta)A_{ST}+\theta A_{tree}\|_1}\), \(L(\theta)=D(\theta)-A(\theta)\).  
- **Messgrößen:**  
  - (E1) Spektral-Dimensions-Proxy \(\hat d_s(\theta)\) aus Heat-Kernel Fit.  
  - (E2) Frontgeschwindigkeit \(\hat v^*(\theta)\) aus Continuous-Time-Quantum-Walk (CTQW).  

## Ergebnisse

- \(\hat d_s(\theta)\): fällt von ca. **1.67** (ST) auf ca. **0.80** (Baum).  
- \(\hat v^*(\theta)\): steigt moderat von **0.51 → 0.55**.  
- Übergang: Potenzgesetz (Fraktal) → exponentielle Abnahme (Baum).  

## Akzeptanzkriterien

- (K1) Glatter Trend in \(\hat d_s(\theta)\) / Exponentialrate \(\alpha(\theta)\) ✅  
- (K2) \(\hat v^*(\theta)\) bleibt endlich und moderat variierend ✅  
- (K3) Konsistenz mit Theorie (Fraktal vs. Baum) ✅  

**Validierungsstatus:**  
| Kriterium | Status |
|-----------|--------|
| K1 | 🟢 |
| K2 | 🟢 |
| K3 | 🟢 |

## Offene Punkte / To-Do

- Erweiterung der Fits: kombinierter Power- & Exponential-Fit über gesamtes \(\theta\)-Intervall.  
- Systematische Analyse größerer ST-Level (≥7) für Robustheit.  
- Numerische Stabilität für lange Zeitfenster prüfen.  

## Referenzen

- Kliesch, Gogolin, Eisert — Lieb–Robinson Bounds.  
- Woess — Random Walks on Trees.  
- Sato et al., PRA 101:022312 — Quanten-Suche auf Fraktalen.  

## Lizenz

- **Code** (im `files/`-Ordner): MIT License.  
- **Nicht-Code** (PDFs, CSV, PNG): CC BY 4.0.  

© 2025 antaris — Code: MIT; Daten & Abbildungen: CC BY 4.0.
