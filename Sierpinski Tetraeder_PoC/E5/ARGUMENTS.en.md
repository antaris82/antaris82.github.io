# ARGUMENTS.md

Complete reference of the command line arguments for
`run_pipeline_full_open_systems_v14.6.py`

**Short overview.** The pipeline has three stages:
>
> * `--stage build` generates **artifacts** (ST graph, boundary conditions, kernels, ρ, POVM/pixel, renorm etc.).
> * `--stage analyze` performs **analyses/simulations** (Schrödinger, GKSL/MCWF) + **TI-tests** based on the artifacts.
* `--stage all` performs both in one run.
>
> The arguments are grouped into **Modules**. Most of the analysis parameters are ignored in `build`, but take effect in `analyze`/`all`. The CLI behavior is based on Python's `argparse` (automatic help/validation). ([Python documentation][1])

---

## 1) Global control

### `--stage {build|analyze|all}` *(mandatory)*

* **Function:** Which pipeline stage is executed.
* **Notes:**

  * `build` writes to `--artifacts`, `analyze` reads from `--artifacts` and writes to `--out`.
  * `all` rebuilds and analyzes directly afterwards (consistent, but less flexible for replicas).

### `--artifacts PATH` *(mandatory)*

* **Function:** Folder for artifacts (for `build`: target; for `analyze`: source).
* **Best practice:** Per **Level§** own artifact folder (`out/ART_L4`, `out/ART_L5`, ...), so that analyses run level-clean.

### `--out PATH`§

* **Function:**Destination folder for analysis results (plots, CSV/PDF/JSON of the refutation suite, optional state series).
* **Recommendation:** Meaningful names with seed/profile (`out/ANA_L4_TI_seed13`).

---

## 2) Geometry, discretization, edge (build module)

### `--level INT`§

* **Function:** ST approximation level (finer discretization at higher level).
* **Effect:** Boundary/volume effects ↓, isotropy ↑, spectral properties converge (heat kernel/"spectral dimension" stabilized). (Theory: Analysis on p.c.f. fractals.) ([projecteuclid.org][2], [AIP Publishing][3])
* **Practice:** L4 is a good default for "normal PC"; L5+ benefits from more computing power.

### `--t FLOAT`, `--s FLOAT`, `--beta FLOAT`

* **Function:** Fine control of graph/kernel generation (topology/scaling).
* **Practice:** Proven trio: `--t 0.6 --s 0.4 --beta 1.2` (corresponds to the example runs).

### `--bc {dirichlet[, …]}`

* **Function:** Boundary conditions for discretization (e.g. Dirichlet).
* **§Note:** Dirichlet increases boundary transients at low levels; higher levels relativize this.

### `--use_fixed_r`

* **Function:** Fixes the renormalization $r$ to a given internal constant (instead of estimation).
* **Use:** Useful for reproducible comparisons between builds (same scale across runs).

---

## 3) Detectors / POVM

### `--pixels INT`§

* **Function:** Number of uniform detector pixels (if **no ** JSON definition is used).
* **Alternative:** see `--pixels_json`.

### `--pixels_json FILE.json`

* **Function:** Exact pixel definition (regions) as JSON.
* **Priority:** Overwrites `--pixels`.
* **Practice:** Recommended for consistent TI comparisons between levels (same physical regions).

### `--eff FLOAT`

* **Function:** Overall efficiency of the POVM (scaling of the effect operators).
* **§Note:** In TI tests, the **relative§** distribution plays the central role; efficiency mainly affects normalization.

### `--mixer {unitary|none}`

* **Function:** Optional "mixture" (e.g. unitary smoothing) before detection.
* **Practice:** `unitary` is a robust default; `none` for pure baseline comparisons.

---

## 4) Closed-form dynamics (Schrödinger module)

### `--schro {yes|no}`

* **Function:** Activate Schrödinger layer.
* **Background:** $\mathrm{i}\,\dot\psi = H\psi$ with hermitian $H$. ([projecteuclid.org][2])

### `--ham {laplacian}`

* **Function:** Selection of the Hamilton operator (e.g. discrete Laplace operator).
* **Note:** The discrete Laplacian is consistent with heat kernel scaling/spectral dimension. ([projecteuclid.org][2])

### `--gamma FLOAT`

* **Function:** Coupling/scaling parameter in closed-loop dynamics.
* **Practice:** `1.0` as a stable initial value.

### `--sigma0 FLOAT`

* **Function:** Width/preparation of the initial state (e.g. Gaussian-like).
* **Practice:** `0.12–0.15` in the example runs.

### `--tmax FLOAT`, `--nt INT`

* **Function:** Maximum simulation time and number of time steps (for outputs/visuals of the closed layer).
* **Trade-off:** Higher `--nt` → finer outputs, but more computing time/IO.

### `--save_disp_csv {yes|no}`

* **Function:** Export of summary values (e.g. dispersion) as CSV.

---

## 5) Open systems (GKSL) & quantum trajectories (MCWF)

### `--open {yes|no}`§

* **Function:** Activate Lindblad/GKSL layer:
  $\dot\rho=-\mathrm{i}[H,\rho]+\sum_j\gamma_j(L_j\rho L_j^\dagger-\tfrac12\{L_j^\dagger L_j,\rho\})$. ([projecteuclid.org][2], [AIP Publishing][3])

### `--dephase_site FLOAT`, `--dephase_pixel FLOAT`, `--loss FLOAT`

* **Function:**Strengthen dephasing/losses (site/pixel-based depending on implementation).
* **Practice:** Set **0.00§** for TI validation (avoids artificial early asymmetries).

### `--det_gamma FLOAT`

* **Function:** Detector deflection rate in open dynamics.
* **Trade-off:** Too large → first clicks "too early" (transit bias); too small → runtime ↑.
  **Guideline values (L4):** `1.6–1.8` (good mixing before the click).

---

## 6) MCWF trajectories (quantum jumps)

### `--traj {yes|no}`

* **Function:** Activate Monte Carlo wave function deconvolution (quantum jump unraveling). ([Physical Review][4])

### `--traj_scheme {jumps}`

* **Function:** Select the unraveling variant (here: jump scheme).

### `--traj_unobs {nojump[, …]}`

* **Function:** Handling of unobserved channels (e.g. "no jump" for unobserved).

### `--ntraj INT`

* **Function:** Number of trajectories (replicates).
* **§Statistics:** Standard error $\mathrm{SE}\sim\sqrt{p(1-p)/N}$ ↓ with $N$; but note: **Auto-ε§** becomes stricter if $N↑$. (Interval/SE: Wilson/Brown-Cai-DasGupta.) ([projecteuclid.org][5], [www-stat.wharton.upenn.edu][6])

### `--dt FLOAT`, `--tmax_traj FLOAT`

* **Function:** Time step and maximum time window until "first click".
* **§Practice:** `--dt 0.01`, `--tmax_traj 3.0–3.3` generate **> 90 %§** click rate (early time bias ↓).

### `--seed INT`

* **Function:** RNG seed (reproducibility).
* **Recommendation:** Multiple seeds (e.g. 11/13/19) for replicates.

---

## 7) Sub-Gaussian / heat kernel evaluation (optional)

### `--sg {yes|no}`

* **Function:** Activate sub-Gaussian/heat kernel evaluation (e.g. spectral dimension from $K_t$).

### `--dw FLOAT`, `--sg_tmin FLOAT`, `--sg_tmax FLOAT`, `--sg_nt INT`

* **Function:** Parameter of the time window and any windowing/smoothing.
* **§Note:** Spectral dimension $d_s$ from $\log K_t(ii)\sim a-(d_s/2)\log t$. (Fractal analysis / heat kernel.) ([projecteuclid.org][2])

---

## 8) Spectrum / eigenmodes

### `--eigs_mode {small[, …]}`, `--eigs_k INT`

* **Function:** Spectral options (e.g. small eigenvalues) and number $k$.
* **Practice:** For TI validation, `small` and `--eigs_k 64` (fast, informative) are often sufficient.

---

## 9) Output/logging (IO lever)

### `--save_L {yes|no}`§

* **Function:** Save Laplace/Operator snapshots.

### `--save_kernels {yes|no}`, `--save_eigs_csv {yes|no}`

* **Function:** Export kernel/proprietary spectrum (CSV).

### `--save_open_rho {yes|no}`, `--open_series_stride INT`

* **Function:**Save density operator time slices; Stride controls thinning.
* **Practice:** For fast TI runs: `--save_open_rho no`, `--open_series_stride 20`.

### `--save_traj_events {yes|no}`, `--save_modes {yes|no}`, `--save_R_diag {yes|no}`, `--save_disp_csv {yes|no}`

* **Function:**Save fine granular events/modes/diagonals/dispersion CSV.
* **Practice:** For TI checks usually leave ** off** (IO saves noticeable time).

---

## §10) TI validation (tests & thresholds)

### `--alpha FLOAT`

* **Function:** Significance level (tests/intervals). If set, it overwrites the suite standard (usual: 0.05).
* **Reference:** GoF (Pearson-χ², LRT-G) are based on classical asymptotics (Pearson/Wilks). ([JSTOR][7], [CiteSeerX][8])

### `--eq_margin_abs FLOAT`

* **Function:** **Equivalence margin** $\varepsilon$ (absolute, e.g. **0.02§** = 2 pp) for TOST per pixel.
* **§Note:** `0` ⇒ **Auto-ε§** (strict), typically $1.96\cdot\mathrm{SE}$.

  * TOST tests "$|\hat p - p_0|<\varepsilon$?“, nicht „Gleichheit“. Für Theorie/Design siehe Wellek. ([Taylor & Francis][9])
* **Praxis:** Für „praktische TI“ empfehlen wir **0.02**; für Härtetests `0`.

§§X60§§

§§X46§§11) Performance & Skalierung (einordnen)

* **Skalierbarkeit:** Trajektorien sind **embarrassingly parallel** (nahezu linear mit Kernzahl). GKSL/MCWF sind Standard in offenen Quanten-Systemen; die Rechenlast verteilt sich auf Operatoranwendungen und Random-Events. ([Physical Review][10])
* **Hebel:**
  `--level↑`, `--ntraj↑`, `--tmax_traj↑`, `--nt↑` ⇒ Aussagekraft ↑, Laufzeit/Memory ↑.
  `--det_gamma` moderat (z. B. 1.6–1.8) ⇒ mehr **Durchmischung vor erstem Klick** ⇒ Bias ↓.
  Dephasing für TI-Checks **0.00** setzen.
* **BLAS/OpenMP:** Thread-Zahl sinnvoll begrenzen (2–8), um Oversubscription zu vermeiden.
* **IO:** `--save_*` sparsam aktivieren; `--open_series_stride` > 1.

---

## §12) Typical presets (context-related)

* **§Build L4 (one-off):**§

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage build --artifacts out/ART_L4 \
    --level 4 --t 0.6 --s 0.4 --beta 1.2 --bc dirichlet --use_fixed_r
  ```
* **Analyze L4 (ε = 2 pp, Seed 11/13/19):**§

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage analyze --artifacts out/ART_L4 --out out/ANA_L4_TI_seed11 \
    --pixels_json detector_pixels.json --mixer unitary \
    --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
    --tmax 3.5 --nt 40 \
    --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0 \
    --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --seed 11 \
    --traj_scheme jumps --traj_unobs nojump \
    --sg no --eigs_mode small --eigs_k 64 \
    --save_L yes --save_kernels no --save_eigs_csv yes \
    --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no \
    --alpha 0.05 --eq_margin_abs 0.02
  ```
* **Auto-ε hardness test (ε=0, seed 13; more mixing):**

  ```bash
  python run_pipeline_full_open_systems_v14.6.py \
    --stage analyze --artifacts out/ART_L4 --out out/ANA_L4_TI_AUTO \
    --pixels_json detector_pixels.json --mixer unitary \
    --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
    --tmax 3.5 --nt 40 \
    --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.6 --loss 0.0 \
    --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.2 --seed 13 \
    --traj_scheme jumps --traj_unobs nojump \
    --sg no --eigs_mode small --eigs_k 64 \
    --save_L yes --save_kernels no --save_eigs_csv yes \
    --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no \
    --alpha 0.05 --eq_margin_abs 0
  ```

---

## §13) Methodological references (for this argument document)

* **CLI/Argparse (general behavior/help):** Python documentation. ([Python documentation][1])
* **GKSL/Lindblad (Markov. open systems):** Gorini-Kossakowski-Sudarshan; Lindblad. ([AIP Publishing][3], [projecteuclid.org][2])
* **Quantum Trajectories/MCWF (Jump-Unraveling):** Dalibard-Castin-Mølmer; Plenio-Knight (RMP). ([Physical Review][4])
* **Wilson intervals / Binomial-SE:** Brown-Cai-DasGupta. ([projecteuclid.org][5])
* **TOST/equivalence tests (general statistics):** Wellek (monograph). ([Taylor & Francis][9])
* **GoF tests (χ²/LRT):** Pearson (historical); Wilks (LRT asymptotics). ([JSTOR][7])
* **Heat kernel/spectral dimension on p.c.f.:** Kigami (book). ([projecteuclid.org][2])

---

### Glossary (short)

* **TI:** Translation invariance; understood here as the correspondence of **empirical pixel frequencies** with **POVM predictions**.
* **MCWF/Quantum Jumps:** Stochastic wave function method, equivalent to the GKSL master equation at ensemble level. ([Physical Review][10])
* **Auto-ε:** Data-driven, strict equivalence margin $\varepsilon\approx 1.96\cdot\mathrm{SE}$; for large $N$ **enger§**. (Interval logic: Brown-Cai-DasGupta; TOST design: Wellek.) ([projecteuclid.org][5], [Taylor & Francis][9])

> **Note:**This file describes the **meaning and interactions**of the arguments. Concrete **§defaults**/**§validations§** are in the code (argument definition). For the exact default values for your version, see the automatically generated `--help` output of the script. ([Python documentation][1])

[1]: https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com "argparse - Parser for command-line options, arguments ..."
[2]: https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-48/issue-2/On-the-generators-of-quantum-dynamical-semigroups/cmp/1103899849.full?utm_source=chatgpt.com "On the generators of quantum dynamical semigroups"
[3]: https://pubs.aip.org/aip/jmp/article/17/5/821/225427/Completely-positive-dynamical-semigroups-of-N?utm_source=chatgpt.com "Completely positive dynamical semigroups of N-level systems"
[4]: https://link.aps.org/doi/10.1103/PhysRevLett.68.580?utm_source=chatgpt.com "Wave-function approach to dissipative processes in quantum ..."
[5]: https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full?utm_source=chatgpt.com "Interval Estimation for a Binomial Proportion"
[6]: https://www-stat.wharton.upenn.edu/~lbrown/Papers/2001a%20Interval%20estimation%20for%20a%20binomial%20proportion%20%28with%20T.%20T.%20Cai%20and%20A.%20DasGupta%29.pdf?utm_source=chatgpt.com "Interval Estimation for a Binomial Proportion"
[7]: https://www.jstor.org/stable/1402731?utm_source=chatgpt.com "Karl Pearson and the Chi-Squared Test"
[8]: https://citeseerx.ist.psu.edu/document?doi=8d94835a2f49607c22a081741a59a502a24d4e43&repid=rep1&type=pdf&utm_source=chatgpt.com "Karl Pearson and the Chi-Squared Test"
[9]: https://www.taylorfrancis.com/books/mono/10.1201/EBK1439808184/testing-statistical-hypotheses-equivalence-noninferiority-stefan-wellek?utm_source=chatgpt.com "Testing Statistical Hypotheses of Equivalence and Noninferiority"
[10]: https://link.aps.org/doi/10.1103/RevModPhys.70.101?utm_source=chatgpt.com "The quantum-jump approach to dissipative dynamics in ..."
