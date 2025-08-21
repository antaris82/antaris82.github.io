test13 deutsch
# A1 — Exklusive Metriken & Formale Herleitung

> **Pfad:** `Sierpinski Tetraeder_PoC/A1/` • **Owner:** antaris82  
> **Kurzbeschreibung:** Dieser Ordner enthält die **formale Herleitung** der in A1 verwendeten Metriken (Entropie, Mutual Information, Schnittmetriken) sowie die zugehörigen Assets im Unterordner `files/`.  
> **Math-Hinweis:** Inline \( … \), Display \[ … \].

---

## 🔗 Schnellzugriff
- 📄 **Formale Herleitung:** `A1_ST_exclusive_metrics_formal.pdf`
- 🗂 **Assets & Daten:** `./files/` — siehe [README im files-Ordner](./files/README.md)

---

## 1) Ziel & Kontext
A1 etabliert die **exklusiven Observablen und Schnittmetriken** auf dem ST-Ur-Graphen.  
- Formaler Teil: Beweise und Definitionen im PDF (Graphkonstruktion, Korrelationsmatrix, Entropie- und MI-Gleichungen, Exklusivregel, Tabellen/Plots).  
- Operativer Teil: Assets im Unterordner `files/` (CSV, PNG, GIF, Python-Skripte).

## 2) Axiome & Kernergebnisse
**Axiome (frei-fermionisches Modell).**  
- ST-Graph (Level L=4) über IFS-Konstruktion.  
- Hamiltonian \(H=L\) (Graph-Laplacian).  
- Füllung \(\nu=1/2\), Gaußscher Grundzustand, Korrelationsmatrix \(C=U_{occ}U_{occ}^\top\).  

**Kernresultate (aus PDF).**  
\[ S(\rho_A) = -\, \mathrm{tr}[C_A\log C_A + (I-C_A)\log(I-C_A)] \]  
\[ MI(A:B) = S(A)+S(B)-S(A\cup B) \]  
Exklusivregel: Präfix-Zuweisung (RED, YELLOW, GREEN), disjunkt und total.  

**Messwerte (L=4).**  
- Regionale Entropien: S(GREEN)=1.1933, S(YELLOW)=30.2059, S(RED)=1.8920.  
- Paarmetriken: MI(RED:YELLOW)=0.1649, cut=3, d_min=1.  
- Layer-Skalierung: S(4)=198.43, MI(4:Rest)=396.86.

## 3) Methoden / Formalismus
- **Graphkonstruktion:** Präfix-IFS, Kanten als Vereinigung lokaler Tetraeder.  
- **Korrelationsmatrix:** aus Hamilton-Eigenvektoren.  
- **Exklusivregel:** Priorität RED > YELLOW > GREEN.  
- **Metriken:** cut(A,B), \langle|C|\rangle_{cross}, d_min, S(A), MI(A:Rest), MI(A:B).

## 4) Datei- & Ordnerübersicht
| Pfad | Typ | Kurzbeschreibung |
|---|---|---|
| `./A1_ST_exclusive_metrics_formal.pdf` | PDF | Formale Herleitung und Auswertung von Verschränkungs- und Schnittmetriken auf dem ST-Graphen (L=4); enthält Definitionen, Propositionen, Tabellen, Abbildungen. |
| `./files/` | Ordner | Enthält Assets (CSV, PNG, GIF, ST.py). Siehe eigenes README (A1/files). |

---

## 5) Akzeptanzkriterien
- **K1:** PDF dokumentiert Definitionen, Propositionen und numerische Werte konsistent.  
- **K2:** Unterordner `files/` enthält Assets (CSV, PNG, GIF) und ist mit PDF konsistent.  
- **K3:** Exklusivzuweisung (RED/YELLOW/GREEN) disjunkt und total (Lemma 3.2).  
- **K4:** Reproduzierbarkeit: Zahlen in PDF stimmen mit CSVs im `files/`-Ordner überein.

## 6) Reproduzierbarkeit
1. **Theorie:** PDF konsultieren (Herleitungen und Tabellen).  
2. **Assets:** Unterordner `files/` nutzen (CSV/PNG/GIF, siehe eigenes README).  
3. **Pipeline:** `ST.py` ausführen, vergleiche Ergebnisse mit PDF-Tabellen.  
4. **Validierung:** Prüfe, dass MI(A:A)=2S(A) gilt; Layer-Skalierung mit CSV konsistent.

## 7) Themenbezogene Informationen
- A1 liefert das Fundament für Folge-Ordner (A2, A3, …).  
- Verknüpfung zur Literatur: Peschel (2003), Eisert/Cramer/Plenio (2010).  

## 8) Unterordner (Struktur)
```
A1/
├─ A1_ST_exclusive_metrics_formal.pdf
└─ files/
```

## 9) Allgemeine Hinweise
- Math-Delimiters: Inline \( … \), Display \[ … \].  
- Unterordner bringen eigene README (hier: files).  
- Pfade relativ halten.

## 10) Offene Punkte / To-Do
- [ ] Konsistenzcheck zwischen PDF und allen CSVs automatisieren.  
- [ ] Layer-Skalierung für höhere L > 4 testen.  
- [ ] Integration in ST-Graph PoC (A2+).

## 11) Validierungsstatus
| Kriterium | Status | Kommentar |
|---|---|---|
| K1 | 🟢 | PDF geprüft |
| K2 | 🟢 | files/ vorhanden, README konsistent |
| K3 | 🟢 | Exklusivregel formal bewiesen |
| K4 | 🟡 | CSV/PDF-Abgleich noch manuell |

## 12) Referenzen
- J. Eisert, M. Cramer, M. B. Plenio: *Colloquium: Area laws for the entanglement entropy*. Rev. Mod. Phys. 82 (2010) 277–306.  
- I. Peschel: *Calculation of reduced density matrices from correlation functions*. J. Phys. A 36 (2003) L205.  
- I. Peschel, V. Eisler: *Reduced density matrices and entanglement entropy in free lattice models*. J. Phys. A 42 (2009) 504003.

## 13) Lizenz
Lizenz

    Code (insb. in ./files/): MIT License.
    Nicht-Code (z. B. PDFs, CSV/PNG): Creative Commons Attribution 4.0 International (CC BY 4.0).

    © 2025 antaris — Code: MIT; Daten/Abbildungen/Texte (inkl. PDFs): CC BY 4.0.

## 14) Changelog
- **v1.0 (2025-08-19):** Erstausgabe für `A1/` (Formale Herleitung + Assets-Verweis).
