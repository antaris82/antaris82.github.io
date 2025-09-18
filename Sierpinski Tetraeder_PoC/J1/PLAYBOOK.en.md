
<!-- Math uses $$ ... $$ delimiters throughout -->

# PLAYBOOK - Presets & Reproducible runs

This playbook provides **3 presets** (weak, medium, extreme) with **parameter blocks**, **expectations** and **acceptance criteria** (A-E). The following applies to all presets: Integrator **Strand (CPTP, 2nd order)**; **Substeps** as specified.

---

## Common formulas

Bathroom statistics:
$$
p_{\mathrm{exc}}=\frac{1}{1+e^{\beta\omega}},\qquad
p_{\mathrm{gnd}}=1-p_{\mathrm{exc}}.
$$

Rates (from micro, with $$K=\sin^2\theta/\Delta t$$):
$$
\gamma_\downarrow=Kp_{\mathrm{gnd}},\qquad
\gamma_\uparrow=Kp_{\mathrm{exc}},\qquad
T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\quad
T_2=2T_1.
$$

---

## Preset A - **Weak / Davies** (target: all green)

**§Parameter**
- $$\omega=1.0,\ \beta=2.0$$
- $$g=0.5,\ \tau_{\mathrm{int}}=0.2\ \Rightarrow\ \theta=g\tau_{\mathrm{int}}=0.1\ \mathrm{rad},\ \sin^2\theta\approx 0.01$$
- $$\Delta t=0.2,\ n_{\mathrm{steps}}=60$$
- Start: $$\theta_0=\pi/3,\ \phi_0=0$$
- **Strand§**, **Substeps§** = 20

**§Expectations**
- $$p_{\mathrm{exc}}\approx \frac{1}{1+e^{2}}\approx 0.1192,\quad p_{\mathrm{gnd}}\approx 0.8808.$$
- $$\gamma_\downarrow\approx 5\cdot 0.01\cdot 0.8808\approx 0.0440,\quad
  \gamma_\uparrow\approx 5\cdot 0.01\cdot 0.1192\approx 0.0060.$$
- $$T_1\approx 20.0,\quad T_2\approx 40.0.$$
- **§Trace distance** GKSL vs. trace-out: $$\lesssim 4\cdot 10^{-3}.$$§

**§Acceptance (A-E)** - expected result: **§green§** in all checks.

---

## Preset B - **Medium / stronger impact** (target: green/yellow)

**Parameter§**
- Like A, but $$\theta=0.2\ \mathrm{rad}\ (\sin^2\theta\approx 0.0395),\ \Delta t=0.2.$$
- **Strand§**, **Substeps§** = 20-30

**Expectations**
- $$\gamma_\downarrow+\gamma_\uparrow\approx 5\cdot 0.0395\approx 0.1975,$$
  $$T_1\approx 5.1,\quad T_2\approx 10.1.$$
- **§Semigroup defect§** and **GADC distance§** can slip into **§yellow§** if substeps are too small.

**§Decrease** - A, D mostly **§green§**; B/C **§green/yellow§**; E **§green§** if there are enough trajectories.

---

## Preset C - **Extreme / stress test** (target: visible tipping points)

**Parameter§**
- Like A, but $$\theta=0.5\ \mathrm{rad}\ (\sin^2\theta\approx 0.2298),\ \Delta t=0.2.$$
- **§Strand§**, **Substeps§** = 40 (or more)

**Expectations**
- Very short times: $$T_1\approx \frac{1}{5\cdot 0.2298}\approx 0.87,\ T_2\approx 1.74.$$
- **CPTP** usually remains green with strand; **Semigroup defect**/ **GADC distance** can become **yellow/red** (non-Markov effects, discretization).

**§Decrease** - A **§green§**, B/C **§yellow/red§** possible, D **green§** (monotonic), E **yellow§** if too few trajectories.

---

## §Workflow (for all presets)

1. **§Tab 1**: Set parameters → *Run micro trace-out*.  
2. **Tab 2§**: Integrator **§Strand§**, **Substeps§** as above → GKSL vs. Compare trace-out; **§Fits** \(T_1/T_2\).  
3. **Tab 3§**: Check A-E (table in `README.md`).  
4. **Tab 4§**: Export → Save ZIP as artifact.

**Tip:** For Δt sweeps, keep $$K=\sin^2\theta/\Delta t$$ **constant** to measure pure **numeric convergence**.
