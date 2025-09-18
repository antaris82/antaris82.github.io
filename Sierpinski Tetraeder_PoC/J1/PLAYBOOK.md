
<!-- Math uses $$ ... $$ delimiters throughout -->

# PLAYBOOK — Presets & Reproduzierbare Läufe

Dieses Playbook liefert **3 Presets** (schwach, mittel, extrem) mit **Parameter‑Blöcken**, **Erwartungen** und **Abnahmekriterien** (A–E). Für alle Presets gilt: Integrator **Strang (CPTP, 2. Ordnung)**; **Substeps** wie angegeben.

---

## Gemeinsame Formeln

Bad‑Statistik:
$$
p_{\mathrm{exc}}=\frac{1}{1+e^{\beta\omega}},\qquad
p_{\mathrm{gnd}}=1-p_{\mathrm{exc}}.
$$

Raten (aus Mikro, mit $$K=\sin^2\theta/\Delta t$$):
$$
\gamma_\downarrow=Kp_{\mathrm{gnd}},\qquad
\gamma_\uparrow=Kp_{\mathrm{exc}},\qquad
T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\quad
T_2=2T_1.
$$

---

## Preset A — **Schwach / Davies** (ziel: alles grün)

**Parameter**
- $$\omega=1.0,\ \beta=2.0$$
- $$g=0.5,\ \tau_{\mathrm{int}}=0.2\ \Rightarrow\ \theta=g\tau_{\mathrm{int}}=0.1\ \mathrm{rad},\ \sin^2\theta\approx 0.01$$
- $$\Delta t=0.2,\ n_{\mathrm{steps}}=60$$
- Start: $$\theta_0=\pi/3,\ \phi_0=0$$
- **Strang**, **Substeps** = 20

**Erwartungen**
- $$p_{\mathrm{exc}}\approx \frac{1}{1+e^{2}}\approx 0.1192,\quad p_{\mathrm{gnd}}\approx 0.8808.$$
- $$\gamma_\downarrow\approx 5\cdot 0.01\cdot 0.8808\approx 0.0440,\quad
  \gamma_\uparrow\approx 5\cdot 0.01\cdot 0.1192\approx 0.0060.$$
- $$T_1\approx 20.0,\quad T_2\approx 40.0.$$
- **Trace‑Distance** GKSL vs. Trace‑out: $$\lesssim 4\cdot 10^{-3}.$$

**Abnahme (A–E)** — erwartetes Ergebnis: **grün** in allen Checks.

---

## Preset B — **Mittel / stärkerer Stoß** (ziel: grün/gelb)

**Parameter**
- Wie A, aber $$\theta=0.2\ \mathrm{rad}\ (\sin^2\theta\approx 0.0395),\ \Delta t=0.2.$$
- **Strang**, **Substeps** = 20–30

**Erwartungen**
- $$\gamma_\downarrow+\gamma_\uparrow\approx 5\cdot 0.0395\approx 0.1975,$$
  $$T_1\approx 5.1,\quad T_2\approx 10.1.$$
- **Semigroup‑Defekt** und **GADC‑Distanz** können in **gelb** rutschen, falls Substeps zu klein.

**Abnahme** — A, D meist **grün**; B/C **grün/gelb**; E **grün** bei genügend Trajektorien.

---

## Preset C — **Extrem / Stress‑Test** (ziel: sichtbare Kipp‑Punkte)

**Parameter**
- Wie A, aber $$\theta=0.5\ \mathrm{rad}\ (\sin^2\theta\approx 0.2298),\ \Delta t=0.2.$$
- **Strang**, **Substeps** = 40 (oder mehr)

**Erwartungen**
- Sehr kurze Zeiten: $$T_1\approx \frac{1}{5\cdot 0.2298}\approx 0.87,\ T_2\approx 1.74.$$
- **CPTP** bleibt mit Strang i.d.R. grün; **Semigroup‑Defekt**/ **GADC‑Distanz** können **gelb/rot** werden (Nicht‑Markov‑Effekte, Diskretisierung).

**Abnahme** — A **grün**, B/C **gelb/rot** möglich, D **grün** (monoton), E **gelb** bei zu wenigen Trajektorien.

---

## Workflow (für alle Presets)

1. **Tab 1**: Parameter setzen → *Run micro trace‑out*.  
2. **Tab 2**: Integrator **Strang**, **Substeps** wie oben → GKSL vs. Trace‑out vergleichen; **Fits** \(T_1/T_2\).  
3. **Tab 3**: A–E kontrollieren (Tabelle in `README.md`).  
4. **Tab 4**: Export → ZIP als Artefakt ablegen.

**Tipp:** Bei Δt‑Sweeps $$K=\sin^2\theta/\Delta t$$ **konstant** halten, um reine **Numerik‑Konvergenz** zu messen.
