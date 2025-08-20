# E1 — Bühne verifiziert: Laplace‑Operator, Heat‑Kernel & spektrale Dimension (ST)

> **Pfad:** `Sierpinski Tetraeder_PoC/E1/` • **Owner:** antaris82  
> **Kurzbeschreibung:** Dieses Verzeichnis enthält das formale PDF zum Beweis von **E1**: Existenz von Resistance‑/Dirichlet‑Form \((\mathcal E,\mathcal F)\), selbstadjungiertem Laplace‑Operator \(-\Delta\), stetigem Heat‑Kernel \(p_t(x,y)\) auf dem **Sierpiński‑Tetraeder (ST)** sowie der zugehörigen Exponenten \(d_w, d_s\).  
> **Math‑Hinweis:** Inline \( … \), Display \[ … \].

---

## 🔗 Schnellzugriff
- 📄 **Beweis:** `E1_proof.pdf` — vollständige Herleitung, Definitionen, Lemma‑Kette, Literatur.

---

## 1) Ziel & Kontext
E1 fixiert die **Bühne** für alle folgenden PoC‑Schritte: wohldefinierte Analysis auf dem ST (pcf‑Fraktal), sub‑gaußsche Heat‑Kernel‑Schranken bzgl. Widerstandsmetrik und die **spektrale Dimension** als Leitgröße für asymptotische Skalen.

## 2) Axiome & Kernergebnisse (aus dem PDF)
- **Existenz:** Auf pcf‑Räumen mit regulärer harmonischer Struktur existieren \((\mathcal E,\mathcal F)\), \(-\Delta)\) und ein stetiger \(p_t\).  
- **Sub‑gaußsche Schranken:** Es gelten zweiseitige Estimates relativ zur Widerstandsmetrik \(R\) und Walk‑Dimension \(d_w>2\).  
- **Spektrale Dimension (ST):**
  \[
  d_s \,=\, \frac{2\ln 4}{\ln 6} \approx 1.5474, 
  \qquad p_t(x,x)\asymp t^{-d_s/2}\;(t\downarrow 0).
  \]

## 3) Methoden / Formalismus
- **pcf‑Framework (Kigami):** Grenzbildung renormierter Energien auf Graph‑Approximanten; Widerstandsmetrik \(R\).  
- **Heat‑Kernel:** Konstruktion über die zugehörige reguläre Dirichlet‑Form; Positivität, Stetigkeit.  
- **Exponenten:** Ableitung/Verifikation für Sierpiński‑Simplexe; Spezialisierung auf ST.

## 4) Datei‑ & Ordnerübersicht
| Pfad | Typ | Kurzbeschreibung |
|---|---|---|
| `./E1_proof.pdf` | PDF | Formale Herleitung und Beweisführung zu E1 (Bühne, Heat‑Kernel, \(d_s\), Literatur). |

---

## 5) Akzeptanzkriterien
- **K1:** PDF enthält konsistente Definitionen (pcf, Resistance‑Form, \(-\Delta\), \(p_t\)).  
- **K2:** Sub‑gaußsche Struktur (Parameterschema, Abhängigkeiten von \(R,d_w\)) ist explizit dokumentiert.  
- **K3:** **ST‑Spezialisierung:** \(d_s=2\ln 4/\ln 6\) und On‑Diagonal‑Skalierung werden explizit angegeben.  
- **K4:** Literaturangaben vorhanden (Primärquellen).

## 6) Reproduzierbarkeit
1. **Lesen:** `E1_proof.pdf` vollständig durcharbeiten (Definitionen → Propositionen → Sätze).  
2. **Quervergleich:** Formeln/Exponenten mit den im PDF zitierten Primärquellen abgleichen.  
3. **(Optional):** Numerische Checks (Heat‑Trace) sind nicht erforderlich, da E1 formal bewiesen ist.

## 7) Offene Punkte / To‑Do
- [ ] Ergänzende Abbildungen (Heat‑Trace, log‑periodische Modulation) für spätere Ordner.  
- [ ] Kurze Notiz zu Zeitskalierungskonventionen (Normalisierungskonstante).

## 8) Validierungsstatus
| Kriterium | Status | Kommentar |
|---|---|---|
| K1 | 🟢 | Vollständig im PDF |
| K2 | 🟢 | Sub‑gaußsche Form dargelegt |
| K3 | 🟢 | \(d_s\) und Diagonalskalierung angegeben |
| K4 | 🟢 | Primärquellen gelistet |

## 9) Referenzen
Siehe Literaturliste in `E1_proof.pdf`.

## 10) Lizenz
Lizenz

    Code (insb. in ./files/): MIT License.
    Nicht‑Code (z. B. PDFs, CSV/PNG): Creative Commons Attribution 4.0 International (CC BY 4.0).

    © 2025 antaris — Code: MIT; Daten/Abbildungen/Texte (inkl. PDFs): CC BY 4.0.

## 11) Changelog
- **v1.0 (2025-08-20):** Erstausgabe für `E1/`.
