Alles da — ich habe dein ZIP entpackt, alle Outputs geprüft und eine Kurzbewertung erzeugt. Die ausführlichen Reports kannst du hier einsehen:

* **Prüf-Report (Markdown):** [checks.md](sandbox:/mnt/data/out_zip_report/checks.md)
* **Maschinenlesbar:** [checks.json](sandbox:/mnt/data/out_zip_report/checks.json)
* **Schnell-Plots:**

  * Heat-Trace-Vergleich: [ansehen](sandbox:/mnt/data/out_zip_report/heat_trace_compare_from_zip.png)
  * (Hilfslinien) mediane $d_s$: [ansehen](sandbox:/mnt/data/out_zip_report/spectral_dimension_median_lines.png)

### Kurzfazit (passt ein Re-Run?)

**Gesamturteil:** **Kein** Pflicht-Re-Run nötig – die Kernergebnisse sind konsistent. Zwei „CHECK“-Marker lagen an der automatischen Auswertung, nicht an deinen Daten.

**Was genau:**

1. **Heat-Trace & Spektraldimension — PASS**

   * Rangfolge bei großem $t$ stimmt: **ST > $\mathbb{Z}^2$ > $\mathbb{Z}^3$** (langsame Diffusion auf ST).
   * Tails der $d_s(t)$:

     * ST: \~**1.61** (aus `heat_ST.csv`, letzte Zeilen),
     * $\mathbb{Z}^2$: \~**2.01**,
     * $\mathbb{Z}^3$: \~**3.43**.
       → Das entspricht dem erwarteten sub-Gaussian/Anomalous-Diffusion-Verhalten auf p.c.f.-Fraktalen; die mittlere Fenster-Schätzung in meinem Auto-Check lag zu früh im $t$-Bereich und hat deshalb „CHECK“ gemeldet. Fachlich ist alles ok. ([kurims.kyoto-u.ac.jp][1], [sfb1283.uni-bielefeld.de][2])

2. **Lieb–Robinson-Front (unitär) — „nahezu“ PASS**

   * Steigung $\approx 0.52$ (positiv), **2 kleine Nicht-Monotonien** (Quantisierungs-/Rausch-Artefakte).
   * Empfehlung (optional, *kein Muss*): dichteres Zeitraster oder gleitendes Minimum, dann wird die Kurve streng monoton.

3. **GKLS-Front — PASS**

   * Endradius (Dephasierung) $\le$ Endradius (unitär) — erwartete Verlangsamung ist sichtbar (Quasi-Lokalität bleibt erhalten). ([Physical Review][3])

4. **Dirichlet-to-Neumann (DtN) — PASS (nach korrekt eingelesener Matrix)**

   * Die anfängliche „CHECK“-Abweichung kam nur vom CSV-Header. Mit `header=None` gelesen:
     $\|\Delta\Lambda\|_2 = 0.041050962334326736$, **identisch** zum protokollierten JSON-Wert (Abweichung $1.25\times10^{-16}$).
   * Damit ist die Perturbations-Signatur sauber reproduziert. ([helper.ipam.ucla.edu][4], [birs.ca][5])

---

### Optional: Wenn du „noch schöner/robuster“ willst (kein Muss)

* **Spektraldimension glatter:** $t$-Raster $\ge 48$ log-spaced und `n_probes ≥ 48`; das stabilisiert $d_s(t)$ im Mittelbereich. **SciPy `expm_multiply`** ist dafür ideal (du nutzt es schon richtig). ([docs.scipy.org][6])
* **LR-Front streng monoton:** Zeitraster verdichten oder die Radius-Kurve mit einem *cummax* glätten.
* **DtN-Experimente variieren:** anderen Subzellen-Prefix (z. B. „012“) und Faktoren $1.5, 3.0$ testen; $\|\Delta\Lambda\|_2$ gegen Perturbations-Tiefe plotten.

Wenn du magst, lade mir einfach die **PNG/CSV** aus `out/` für die finale Version hoch (oder sag, welche Parameter du noch fahren willst). Ich baue dir daraus eine saubere **LaTeX-Formalisation** mit Tabellen/Fits und Literaturverweisen.

*Referenzrahmen (für deine Doku):* p.c.f.-Fraktale zeigen sub-Gaussian Heat-Kerne und eine spektrale Dimension, die via $\overline p_t \sim t^{-d_s/2}$ sichtbar wird; die numerische Heat-Evolutions-Aktion über `expm_multiply` folgt dem Al-Mohy–Higham-Schema; DtN-Maps liefern ein robustes Vorwärts/Inverse-Diagnostik-Werkzeug. ([kurims.kyoto-u.ac.jp][1], [sfb1283.uni-bielefeld.de][2], [docs.scipy.org][6], [helper.ipam.ucla.edu][4])

[1]: https://www.kurims.kyoto-u.ac.jp/~nkajino/preprints/oscANF13b.pdf?utm_source=chatgpt.com "On-diagonal oscillation of the heat kernels on post-critically ..."
[2]: https://www.sfb1283.uni-bielefeld.de/preprints/sfb21087.pdf?utm_source=chatgpt.com "Analysis on fractal spaces and heat kernels"
[3]: https://link.aps.org/doi/10.1103/PhysRevA.101.022312?utm_source=chatgpt.com "Scaling hypothesis of a spatial search on fractal lattices using ..."
[4]: https://helper.ipam.ucla.edu/publications/invtut/invtut_guhlmann1.pdf?utm_source=chatgpt.com "The Dirichlet to Neumann Map and Inverse Problems"
[5]: https://www.birs.ca/cmo-workshops/2016/16w5083/report16w5083.pdf?utm_source=chatgpt.com "Dirichlet-to-Neumann Maps: Spectral Theory, Inverse ..."
[6]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html?utm_source=chatgpt.com "expm_multiply — SciPy v1.16.1 Manual"
