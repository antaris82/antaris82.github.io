# E3 — Operative Kinematik auf dem ST‑Ur‑Graph (Radar/k‑Kalkül)

> Dieses README beschreibt den Ordner **`Sierpinski Tetraeder_PoC/E3/`** und verweist auf den **Assets‑Ordner `../files/`**.  
> **Math‑Delimiters:** Inline mit \( … \), Display mit \[ … \].

---

## 🔗 Schnellzugriff

- 📄 **Vollständiger Beweis:** `E3_full_proof.pdf`
- 📄 **Operative Kinematik & Protokolle:** `E3_kinematics.pdf`
- 📄 **Zusammenfassung & Tests:** `E3_summary.pdf`
- 🗂 **Assets & Daten:** `../files/` (siehe dortiges README für E3‑Artefakte)

---

## 1) Ziel & Kontext

**E3** etabliert die **Lorentz‑Kinematik** **ohne Maxwell‑Input**, allein aus
1) **Relativitätsprinzip**, **Homogenität**, **Isotropie (IR)** und  
2) der in **E2** operativ gemessenen **invarianten Frontgeschwindigkeit** \(v^*\) (Lieb–Robinson‑Kegel).

Kernidee: Setze die dimensionslose Geschwindigkeit \(u:=v/v^*\). Dann ergeben sich die Lorentz‑Transformationen und die Geschwindigkeitskomposition rein **gruppentheoretisch**. Operativ wird die Kinematik **aus Radar-/Echozeiten (Bondi‑k‑Kalkül)** gewonnen; die **Kette (CTQW)** dient als Goldstandard, der **ST‑Pfad** als analytischer Check.

---

## 2) Axiome & Konsequenzen (ohne Lichtpostulat)

**Axiome.** (A1) Relativität; (A2) Homogenität; (A3) IR‑Isotropie; (A4) lokale Dynamik mit Lieb–Robinson‑Schranke.  
**Konsequenz.** Es existiert eine **invariante Geschwindigkeitsskala** \(c_{\mathrm{inv}}\) (operativ: \(c_{\mathrm{inv}}\equiv v^*\)), so dass mit \(u=v/c_{\mathrm{inv}}\):
\[
t'=\gamma(u)\,\bigl(t-u\,x/c_{\mathrm{inv}}\bigr),\quad
x'=\gamma(u)\,\bigl(x-u\,t\bigr),\qquad
\gamma(u)=\frac{1}{\sqrt{1-u^2}}\,.
\]

**Geschwindigkeitsaddition (Einstein):**
\[
u\oplus w=\frac{u+w}{1+u\,w}\,.
\]

---

## 3) Bondi‑k‑Kalkül & Radar (operativ)

**Bondi‑Faktor und Rapidität.** Für \(u=v/c_{\mathrm{inv}}\) sei
\[
k(u)=\sqrt{\frac{1+u}{1-u}}\,,\qquad
\theta=\ln k\,.
\]
Dann gilt \(\gamma(u)=\tfrac{1}{2}\bigl(k+1/k\bigr)\) und die **Rapidität addiert**: \(\theta(u\oplus w)=\theta(u)+\theta(w)\).

**Echo‑Protokoll (bewegter Spiegel).** A sendet zwei Pulse mit Eigenabstand \(\Delta T\); B bewegt sich radial mit \(u\) und reflektiert sofort. A misst Echozeiten \(T_1,T_2\). Dann
\[
\frac{T_2-T_1}{\Delta T}=k(u)^2=\frac{1+u}{1-u}\,,
\]
woraus \(k\) und \(\gamma=\frac12(k+1/k)\) unmittelbar folgen.

**Lieb–Robinson (Motivation für \(v^*\)).** Für lokal erzeugte Heisenberg‑Dynamik \(\alpha_t\) und disjunkte Träger \(X,Y\):
\[
\bigl\|[\alpha_t(A),B]\bigr\|\le C\,\|A\|\,\|B\|\,\exp\!\Bigl(-\mu\,[d(X,Y)-v\,t]\Bigr)\,,
\]
was **lineare Frontzeiten** und damit eine **operative** \(v^*\le v\) rechtfertigt.

---

## 4) Akzeptanzkriterien (E3)

- **K1 (Gamma‑Law):** \(\hat\gamma(u)\) folgt \(\gamma(u)=1/\sqrt{1-u^2}\) für \(u\in\{0.2,0.4,0.6\}\) (innerhalb Toleranz).
- **K2 (Rapidität):** \(\theta=\ln k\) ist additiv (Mehr‑Beobachter‑Setup, \(k\)-Multiplikation).
- **K3 (Addition):** \(u\)-Komposition via \(u\oplus w\) stimmt mit \(k\)-Multiplikation überein.
- **K4 (IR‑Isotropie):** Richtungs‑Spread der \(v^*\)‑Schätzungen \(\le 20\%\) auf hinreichend großen ST‑Leveln.

Details, Nachweise und numerische Evidenz: siehe PDFs (s. Schnellzugriff).

---

## 5) Reproduktion (How‑To)

1. **Datengrundlage laden:** Assets aus `../files/` (Radar‑CSV/JSON, Plots) gemäß dortigem README.
2. **Kette (Goldstandard):** Radar‑Protokoll implementieren, \(u\) wählen, \(k=\Delta t_B/\Delta\tau_A\) bestimmen, \(\hat\gamma=\tfrac12(k+1/k)\) plotten.
3. **ST‑Analytik (Pfad):** den geschlossenen Radar‑Check entlang eines ST‑Pfads verwenden; die Ratio \((T_2-T_1)/\Delta T\) bestätigt \(k^2\).
4. **ST‑CTQW (PoC):** Radar‑Pings auf zentralen ST‑Bällen; Echo‑Erkennung über \(|U_{ij}(t)|\ge\varepsilon\); längere Horizonte/Threshold‑Sweep empfohlen.
5. **Validierung:** \(\partial_u\theta=1/(1-u^2)\) und Additivität der Rapidität per \(k\)‑Multiplikation testen; Gamma‑Residuen \(\hat\gamma(u)-\gamma(u)\) vs. \(u\) plotten.
6. **Isotropie:** Shell‑Sampling über Richtungen; Mittelwert \(\hat v^*\) und Streuung (K4) berichten.

**Hinweis:** Pfad‑ und Dateinamen sind projekt‑/laufabhängig; befolge das README in `../files/` (E3‑Abschnitt) für konkrete Artefakt‑Namen und Pfade.

---

## 6) Ordnerstruktur (E3)

```
E3/
├─ E3_full_proof.pdf        # Vollständiger Beweis (Axiome→Lorentz, LR→v*, Bondi→k,γ)
├─ E3_kinematics.pdf        # Operative Kinematik & Protokolle (Radar/k, Messdesign)
├─ E3_summary.pdf           # Kurzfassung & Teststatus (K1–K4)
└─ ../files/                # Daten, Plots, CSV/JSON, ggf. GIFs; siehe dortiges README
```

---

## 7) Hinweise für LaTeX/Markdown

- **Math‑Delimiters:** Inline \( … \), Display \[ … \] (kein `$$…$$`/``\(…\)`` in Markdown‑Dateien).
- **Tabellen/Plots:** Aus `../files/` einbinden; Pfade prüfen.
- **Compiler:** Für LaTeX‑Dokumente mit `fontspec` **LuaLaTeX/XeLaTeX** verwenden.
- **Quellen:** Siehe Literatur/Quellenangaben in den PDFs.

---

## 8) Lizenz

Lizenz

    Code (insb. in ./files/): MIT License.
    Nicht‑Code (z. B. PDFs, CSV/PNG): Creative Commons Attribution 4.0 International (CC BY 4.0).

    © 2025 antaris — Code: MIT; Daten/Abbildungen/Texte (inkl. PDFs): CC BY 4.0.

---

## 9) Changelog (E3‑README)

- **v1.0 (2025‑08‑18):** Erstausgabe mit \(…\) / \[ … \]‑Delimitern, Radar/k‑Formeln, Akzeptanzkriterien und Repro‑Anleitung.

