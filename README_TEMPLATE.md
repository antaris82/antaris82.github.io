---
# Optional YAML front matter for GitHub (keine Pflicht)
title: "<TITEL DES ORDNERS/PROJEKTS>"
description: "<1-3 Sätze, worum es geht>"
tags: ["<tag1>","<tag2>"]
# Hinweis: Für Maschinenlesbarkeit gibt es zusätzlich den JSON-Block weiter unten.
---

# <TITEL DES ORDNERS/PROJEKTS>

> **Kurzfassung (TL;DR):** 1–2 Sätze Nutzen/Zweck. Optional ein Link zur Demo/Docs.

<!-- dirindex:quickfacts -->
**Quick Facts**
- **Repo:** `<owner>/<repo>`
- **Pfad:** `<ordner/pfad>`
- **Version/Status:** `draft|stable` &nbsp;•&nbsp; **Stand:** `<YYYY-MM-DD>`
- **Lizenz:** `<Lizenz>`
- **Themen:** `tag1`, `tag2`, `tag3`
<!-- /dirindex:quickfacts -->

<!-- dirindex:cover -->
<!-- Optionales Titelbild. Benenne eine Bilddatei im Ordner als background.jpg/png, 
     damit deine index.html sie automatisch als Hintergrund nutzt. -->
<!-- /dirindex:cover -->

## Inhalt/Übersicht

<!-- dirindex:files:start -->
| Name | Größe | Typ | Rolle | Kurzbeschreibung | Tags |
|---|---:|---|---|---|---|
| file1.ext | 12 KB | data | Rohdaten | Messreihe 1 | data,raw |
| script.py | 3 KB | code | Analyse | erzeugt Abbildung 2 | code,analysis |
| README.md | – | doc | Doku | diese Datei | doc |
<!-- dirindex:files:end -->

> **Hinweis:** Tabelle kann automatisch erzeugt werden – siehe `generate_readme.py`.

## Nutzung

- **Schnellstart:**  
  ```bash
  # ggf. im Unterordner ausführen
  python3 script.py --input data.csv --out results/
  ```

- **Voraussetzungen/Dependencies:**  
  - Python ≥ 3.10
  - Pakete: numpy, matplotlib

- **Struktur:**  
  ```text
  <ordner>
  ├─ data/           # Eingabedaten
  ├─ results/        # Ausgaben/Plots
  ├─ script.py       # Analyse-Skript
  └─ README.md
  ```

## Mathematischer Kontext (optional)

Inline: $E = mc^2$  
Display:
$$
\mathrm{Tr}_B(\rho_{AB})=\sum_j (\mathbb{I}_A\otimes \langle j|)\,\rho_{AB}\,(\mathbb{I}_A\otimes |j\rangle).
$$

Mehrzeilig:
$$
\begin{aligned}
  \bigl\|[\alpha_t(A),B]\bigr\|
  &\le C\,\|A\|\,\|B\|\,\exp\!\bigl(-\mu\,\bigl[d(X,Y)-v_{\mathrm{LR}}\,t\bigr]\bigr).
\end{aligned}
$$

## Referenzen/Links
- Doku: <https://example.com/docs>
- Paper/Preprint: <https://arxiv.org/abs/XXXX.XXXXX>

## Changelog
- YYYY-MM-DD: Erste Version.

## Lizenz
MIT (oder andere) – siehe `LICENSE`.

---

<!-- dirindex-json
{
  "repo": "<owner>/<repo>",
  "path": "<ordner/pfad>",
  "title": "<TITEL DES ORDNERS/PROJEKTS>",
  "updated": "<YYYY-MM-DD>",
  "status": "draft",
  "tags": ["tag1","tag2"],
  "links": {
    "docs": "https://…",
    "paper": "https://…",
    "demo": "https://…"
  },
  "files": [
    {"name": "file1.ext", "role": "data", "desc": "Kurzbeschreibung", "tags": ["data","raw"]},
    {"name": "script.py", "role": "code", "desc": "Analyse-Skript", "tags": ["code","analysis"]}
  ]
}
dirindex-json -->
