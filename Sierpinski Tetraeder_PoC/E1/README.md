# Create README for E1 folder and save for download
readme_e1 = """# E1 — Proof-of-Concept (PoC) für den ST-Graph

Dieser Ordner enthält **E1** des ST-Graph‑PoC und umfasst ausschließlich das Beweis‑PDF **`E1_proof.pdf`**.
Der Schritt **E1** verifiziert die **Bühne** (Mathematik/Analysis) auf dem Sierpiński‑Tetraeder (ST): Existenz von
Resistance‑/Dirichlet‑Form \\((\\mathcal E,\\mathcal F)\\), des selbstadjungierten Laplace‑Operators \\(-\\Delta\\) und eines stetigen
Heat‑Kernels \\(p_t(x,y)\\); ferner sub‑gaußsche Heat‑Kernel‑Schranken bzgl. der Widerstandsmetrik sowie explizite Exponenten
(\\(d_w, d_s\\)) für ST.

- 📄 **Beweis:** [`E1_proof.pdf`](./E1_proof.pdf)

---

## Kurzüberblick (Kernaussagen)

- **Existenz:** Auf ST existieren (bis auf Zeitskalierung eindeutig) eine lokale, reguläre **Resistance‑/Dirichlet‑Form** \\((\\mathcal E,\\mathcal F)\\),
  der zugehörige **selbstadjungierte Laplace‑Operator** \\(-\\Delta\\) sowie ein **(streng positiver) stetiger Heat‑Kernel** \\(p_t(x,y)\\).

- **Sub‑gaußsche Schranken:** Relativ zur **Widerstandsmetrik** \\(R\\) gelten **zweiseitige sub‑gaußsche Bounds** mit **Walk‑Dimension**
  \\(d_w>2\\):
  \\[
  c_1\\, t^{-d_s/2} \\exp\\!\\Big(-c_2\\,\\Big(\\tfrac{R(x,y)^{d_w}}{t}\\Big)^{\\!\\frac{1}{d_w-1}}\\Big)
  \\;\\le\\; p_t(x,y) \\;\\le\\;
  c_3\\, t^{-d_s/2} \\exp\\!\\Big(-c_4\\,\\Big(\\tfrac{R(x,y)^{d_w}}{t}\\Big)^{\\!\\frac{1}{d_w-1}}\\Big).
  \\]

- **Exponenten auf ST:** Für das Sierpiński‑Tetraeder (Sierpiński‑Simplex mit \\(d=3\\)) gilt
  \\[
  d_w = \\frac{\\ln 6}{\\ln 2},\\qquad
  d_s = \\frac{2\\,\\ln 4}{\\ln 6} \\approx 1{.}5474,
  \\]
  woraus \\(p_t(x,x) \\asymp t^{-d_s/2}\\) für \\(t\\downarrow 0\\) folgt (bis auf **log‑periodische Modulationen**).

---

## Dateiübersicht

| Datei | Typ | Beschreibung |
|---|---|---|
| `E1_proof.pdf` | PDF | Vollständiges Beweis‑Dokument zu **E1** (Bühne verifiziert: Laplace‑Operator, Heat‑Kernel, spektrale Dimension auf ST). |

**PDF‑Inhalt (Struktur):**
1. **Voraussetzungen und Definitionen** (pcf‑Set, harmonische Struktur, Widerstandsmetrik, Walk‑ & spektrale Dimension).  
2. **Existenz von \\((\\mathcal E,\\mathcal F)\\), \\(-\\Delta\\) und \\(p_t\\) auf ST** (Kigami‑Theorie, Markov‑Prozess, Stetigkeit/Positivität).  
3. **Sub‑gaußsche Heat‑Kernel‑Abschätzungen & \\(d_w\\)** (VD/Poincaré/Ketten‑Bedingungen; Charakterisierung der Schranken).  
4. **Sierpiński‑Simplex: \\(d_w\\) und \\(d_s\\) explizit; Spezialisierung auf ST** (Formeln für \\(d\\)-Simplexe; Einsetzen von \\(d=3\\)).  
5. **Schluss** (Robustheit der Bühne; Unabhängigkeit von numerischen Experimenten).

---

## Nutzung

- Dieses Verzeichnis dient der **Dokumentation des Beweises**. Zur numerischen Verifikation/Simulation siehe die
  weiteren PoC‑Schritte (z. B. `E2/`).

---

## Lizenz

Da dieser Ordner ausschließlich **Nicht‑Code** enthält, gilt für den Inhalt **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
Das erlaubt **freie Nutzung**, **Weitergabe** und **Bearbeitung**, erfordert aber **Namensnennung** des Urhebers.

> **Lizenzhinweis für dieses Verzeichnis:**  
> © 2025 antaris — **`E1_proof.pdf` unter CC BY 4.0**.  
> *(Sollte künftig Code ergänzt werden, empfehlen wir dafür zusätzlich die **MIT‑Lizenz**.)*

---

## Zitation

Bitte zitiere wie folgt:
> antaris (2025): *E1 — Bühne verifiziert: Laplace‑Operator, Heat‑Kernel und spektrale Dimension auf dem Sierpiński‑Tetraeder*.  
> GitHub‑Repo `antaris82.github.io`, Ordner `Sierpinski Tetraeder_PoC/E1/`. **Lizenz:** CC BY 4.0.

(Optional kann im Repo‑Root eine `CITATION.cff` gepflegt werden, damit GitHub die Zitation automatisch anzeigt.)

---

## Kontakt

Maintainer: **@antaris** — Feedback/Fragen bitte als GitHub‑Issue einreichen.
"""
out_path = "/mnt/data/README_E1.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(readme_e1)
out_path

