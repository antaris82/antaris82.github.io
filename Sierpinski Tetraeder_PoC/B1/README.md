# B1 — Formalisierung, PoC-Brücke und Dateiübersicht

> Dieses README beschreibt den Ordner **`Sierpinski Tetraeder_PoC/B1/`** samt Unterordner **`files/`**.  
> **Math-Delimiters:** Inline mit \( … \), Display mit \[ … \].

---

## 🔗 Schnellzugriff

- 📄 **Beweis/Details:** `B1_v4_check run 3_formal_ST.pdf`
- 📄 **Zusammenfassung (self-contained):** `B1_summary.pdf`
- 📦 **Artefakte & Daten:** `files/`

---

## 1) Ziel & Kontext

**B1** liefert eine endlichdimensionale, saubere Formalisierung auf ST-Graphen (Sierpiński-Tetraeder) mit

- exakter **Gibbs-Reduktion (Teilspur)** für Kronecker-Summen,
- **Laplace-kohärentem Coarse-Graining** via Lift \(L_{\mathrm{lift}}=C^{\top}L_0C\),
- **Approximanten** \(L_A(\alpha)=(1-\alpha)L+\alpha L_{\mathrm{lift}}\) für \(\alpha\in[0,1]\),
- Observablen \(E,S,P\) aus dem Gibbs-Zustand, **Thermo-Checks** über \(Z(\beta)\).

Die PDFs dokumentieren Beweise, Mapping **Code ↔ Formel**, und numerische Checks.

---

## 2) Kernformeln (ohne Platzhalter)

### Partial Trace (Subsystem \(E\) wird getraced)

\[
\rho_S(\beta)
=\operatorname{Tr}_E\!\left(\frac{e^{-\beta\bigl(H_S\otimes \mathbf 1+\mathbf 1\otimes H_E\bigr)}}{\operatorname{Tr}e^{-\beta\bigl(H_S\otimes \mathbf 1+\mathbf 1\otimes H_E\bigr)}}\right)
=\frac{e^{-\beta H_S}}{\operatorname{Tr}e^{-\beta H_S}}\;.
\]

**Kommentar:** Nutzung von \([H_S\otimes \mathbf 1,\mathbf 1\otimes H_E]=0\) ⇒ Faktorisation von \(e^{-\beta H}\) ⇒ exakte Reduktion.

---

### Beispielhafte Observablen-Abbildung über \(\alpha\)

\[
L_A(\alpha)=(1-\alpha)L+\alpha\,C^{\top}L_0C,\qquad 
p_i(\alpha,\beta)=\frac{e^{-\beta\lambda_i(\alpha)}}{\sum_j e^{-\beta\lambda_j(\alpha)}}\;,
\]
\[
E(\alpha,\beta)=\sum_i p_i(\alpha,\beta)\,\lambda_i(\alpha),\quad
S(\alpha,\beta)=-\sum_i p_i(\alpha,\beta)\,\log p_i(\alpha,\beta),\quad
P(\alpha,\beta)=\sum_i p_i(\alpha,\beta)^2\;.
\]

Mit \(Z(\alpha,\beta)=\operatorname{Tr}\,e^{-\beta L_A(\alpha)}\) gelten
\[
\partial_\beta\log Z(\alpha,\beta)=-E(\alpha,\beta),\qquad
\partial_\beta^2\log Z(\alpha,\beta)=\mathrm{Var}_{\rho(\alpha,\beta)}(L_A(\alpha))\ge 0\;.
\]

---

### Lieb–Robinson-artige Schranke (Motivation für effektive Kegel/Kausalität)

\[
\bigl\|[\alpha_t(A),B]\bigr\|
\;\le\;
C\,\|A\|\,\|B\|\,
\exp\!\Bigl(-\mu\,\bigl[d(X,Y)-v_{\mathrm{LR}}\,t\bigr]\Bigr)\,.
\]

Dies motiviert einen **effektiven Lichtkegel** und eine maximale Gruppengeschwindigkeit \(v^*\le v_{\mathrm{LR}}\) auf Gittern/Netzwerken.

---

## 3) PoC: Code ↔ Formel (Mapping)

- **Graph-Aufbau:** `build_graph_by_addresses(level)` → \((V_m,E_m)\), Adjazenz \(A\), Laplacian \(L=D-A\) (symmetrisch, PSD, \(L\mathbf 1=0\)).
- **Aggregation/Rekonstruktion:** \(C\) (Zeilensummen \(1\)), \(R=C^{\top}\); **Lift:** \(L_{\mathrm{lift}}=R\,L_0\,C\).
- **Approximanten:** `L_A_alpha(alpha)` → \(L_A(\alpha)\), PSD und \(\ker\)-Erhalt für \(\alpha\in[0,1]\).
- **Gibbs & Observablen:** `rho_from_spectrum(L,beta)`, `energy/entropy/purity_from_p` → \(E,S,P\) aus Spektrum & \(p_i\).
- **Thermo-Checks:** numerisch \(\partial_\beta\log Z=-E\) und \(\partial_\beta^2\log Z=\mathrm{Var}(L)\) innerhalb Toleranz.

Konkrete Referenzen und numerische Tabellen siehe PDFs und `files/`-Artefakte.

---

## 4) Reproduzierbarkeit (How-To)

1. **Graph-Level** wählen (z. B. ST-Level 4–6) und \(L\) erzeugen.  
2. **Aggregation \(C\)** und **groben Laplacian \(L_0\)** festlegen → \(L_{\mathrm{lift}}=C^{\top}L_0C\).  
3. **Approximanten** \(L_A(\alpha)\) über \(\alpha\in[0,1]\) scannen.  
4. **Gibbs-Zustände** \(\rho(\alpha,\beta)\propto e^{-\beta L_A(\alpha)}\) evaluieren; **\(E,S,P\)** berechnen.  
5. **Thermo-Checks**: \(\partial_\beta \log Z\) und \(\partial_\beta^2 \log Z\) gegen \(E\) bzw. \(\mathrm{Var}(L)\) prüfen.  
6. **Vergleich Urgraph vs. Approximant-Subgraph** (Trends von \(E,S,P\) über \(\alpha\) und Dimension \(n\)).

> Outputs (CSV/JSON/PNG/GIF) liegen in `files/` und sind in den PDFs referenziert.

---

## 5) Ordnerstruktur

```
B1/
├─ B1_v4_check run 3_formal_ST.pdf       # Vollständiger Beweis & PoC-Mapping
├─ B1_summary.pdf                         # Self-contained Zusammenfassung
└─ files/                                 # Daten, Plots, Tabellen, GIFs, Snippets
```

**Typische Inhalte in `files/`:**  
CSV/JSON (Spektren, Sweeps, Hard-Checks), PNG/PDF (Plots), GIF (rotierender ST), TEX-Snippets (Tabellen/Einfügeblöcke).

---

## 6) Hinweise für LaTeX-Nutzung

- **Math-Delimiters:** Inline \( … \), Display \[ … \].  
- **Tabellen:** `booktabs` verwenden (`\toprule`, `\midrule`, `\bottomrule`).  
- **Compiler:** Bei `fontspec` **LuaLaTeX/XeLaTeX** nutzen (Hinweis „inputenc package ignored…“ ist normal).  
- **Bilder:** Pfade prüfen; fehlende Dateien erzeugen Fehler wie *“File `…png` not found: using draft setting”*.  
- **Zitate & Bib:** Quellenangaben wie in den PDFs; zusätzliche `\bibitem` ggf. in `files/` hinterlegt.

---

## 7) FAQ (Kurz)

- **Exakte Teilspur?**  
  Kommutierende Summanden ⇒ \(e^{-\beta H}\) faktorisiert ⇒ \(\rho_S(\beta)=e^{-\beta H_S}/\operatorname{Tr}e^{-\beta H_S}\).

- **Kern-Erhalt unter Lift/Approximation?**  
  \(C\mathbf 1=\mathbf 1\), \(L_0\mathbf 1=0\Rightarrow L_{\mathrm{lift}}\mathbf 1=0\) ⇒ \(L_A(\alpha)\mathbf 1=0\).

- **Thermo-Kohärenz?**  
  \(\partial_\beta\log Z=-E\), \(\partial_\beta^2\log Z=\mathrm{Var}(L)\ge 0\) bestätigt numerisch in `files/`/PDFs.

---

## 8) Changelog (B1)

- **v1.1** — Delimiters auf **\( … \)** und **\[ … \]** umgestellt.
- **v1.0** — Erstveröffentlichung dieses README; Platzhalter `MATHBLOCK_*` ersetzt.
