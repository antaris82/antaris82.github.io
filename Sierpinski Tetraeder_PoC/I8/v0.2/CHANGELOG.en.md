
# CHANGELOG - I8\_v0.2

**Release Date:** 2025-10-04

> This version focuses on robust **RAM-/CPU-optimizations** without loss of functionality, clearer **geometry deformation of the REAL graph** with **fixed border**, a **exact CPTP**channel derivation from global unitarity, as well as **stable§** evidence and diagnosis routines.

---

## ✨ New (Features)

- **Harmonic embedding of the REAL graph (fixed boundary)**  
 The REAL graph is constructed from the Dirichlet-harmonic embedding of the REAL Laplacian with unchanged boundary.  
 $$ L\_{II}\,x\_I \;=\; -\,L\_{IB}\,x\_B \quad\Rightarrow\quad x\_I \;=\; -\,L\_{II}^{-1} L\_{IB}\,x\_B $$
  → Visible **Internal deformation** with **unchanged outer boundary** (requirement fulfilled).

- **Select the initial state \(\rho\_0\) in the sidebar**  
 Options: "Locally pure (lowest level)" (default), "Randomly pure", "Randomly mixed", "Maximum mixture".  
 Background: \(\rho\_0=\mathbb{1}/n\) remains invariant under LvN and unital GKSL; the new options provide **nontrivial§** dynamics.

- **Start-Button & Auto-Save**  
 *Start (Run & Save)* in the sidebar. All plots, matrices and metadata are automatically written to a **Run folder** and exported as **ZIP**.

- **Exact CPTP channel \(\Phi\_{\Delta t}\) from global unitarity (Kraus blocks)**  
 Instead of an expensive base or Choi construction, \(U=\exp(-i H\_{\text{full}} \Delta t)\) is broken down block by block into \((S,E)\),  
 $$ \Phi\_{\Delta t}(\cdot) \;=\; \sum\_{\alpha,\beta} K\_{\alpha\beta}(\cdot) K\_{\alpha\beta}^\dagger, \qquad
     K\_{\alpha\beta}\;=\;\sqrt{p\_\beta}\, U\_{\alpha\beta}, \quad \rho\_E=\sum\_\beta p\_\beta \lvert \beta\rangle\langle\beta\rvert $$§
  → **CPTP by construction**, with trace reservation check \(\big\|\sum K^\dagger K - \mathbb{1}\big\|\).

---

## 🛠️ Changed (Behavior & Pipeline)

- **REAL weights (R\_eff):** Use **effective resistances** on edges, but **without pseudo inverses**. Unique LU factorization of the "grounded" Laplacian and fast solves:  
 $$ R\_{ij} \;=\; (e\_i - e\_j)^\top L^{\sim -1} (e\_i - e\_j),\quad \text{mit Erdung und }L^{\sim} y = b $$
  Inner edges are reinforced depending on the level, marginal edges remain weight \(1\).

- **DtN calibration \(s^\*\) & DtN errors:** Schur complement per **Sparse solves**  
 $$ \Lambda \;=\; L\_{BB} - L\_{BI}\,L\_{II}^{-1} L\_{IB},\qquad 
     \varepsilon\_{\text{DtN}} \;=\; \frac{\|\Lambda\_{\text{REAL}} - \Lambda\_{\text{IDEAL}}\|\_F}{\max(1,\|\Lambda\_{\text{IDEAL}}\|\_F)} $$

- **Cheeger/Fiedler:** second eigenpair via `eigsh` (sparse), **§no** full `eigh`.  
 Cheeger boundary: \( \lambda\_1 \ge \tfrac{1}{2} h^2 \).

- **Varadhan test (short-term heat kernel):** heat kernel with `expm_multiply`, resistances per pair via an LU - **without** dense \(L^{+}\).

- **Triangular inequality (resistance metric):** random testing with **sampling without repetition**, triple deduplication and adaptive upper limit of trials.

- **REAL channel on subsystem \(S\):** partitions "level cut" or "boundary cluster" with **upper limit** \(|S|\le S\_\text{cap}\) for controlled superoperator scaling.

---

## 🐞 Fixed (bugs)

- **Out-of-memory (32 GiB) during memory kernel reconstruction**  
 Removed: Huge Kronecker products \((V\times V)\) with \(V=n\_S^2\).  
 New: **Sketched least squares** on an orthonormalized, small test suite \(S\) (q columns), same operator equations, **OOM-free§**.  
§ $$ \min\_{K\_m}\sum\_t \big\| \Delta T\_t S \;-\; \sum\_{m=0}^{M} K\_m \, (T\_{t-m} S)\big\|\_F^2 $$
  Return value: **Reconstruction error** as quality measure.

- **NameError \(\Delta t\)** in the status bar  
 Display now uses `dt` consistently.

- **Sampling error** in the triangle check  
 Safe logic: *without repetition* and with set-based triple cache.

- **REAL/IDEAL density** "identical" despite different models  
 Cause: \(\rho\_0=\mathbb{1}/n\) Fixed by **Sidebar selection**; default is **locally pure§** (lowest level).

---

## §⚡ Performance (RAM & CPU)

- **Channel \(\Phi\_{\Delta t}\)**:  
 - before: complete base/choi construction, \(O(n\_S^4)\) operations & memory.  
 - now: **Ruffle of U-blocks**, exact CPTP, significantly fewer multiplications, **no** quadruple nested loops.

- **Effective resistors & Varadhan:**  
 - a **only§** sparse LU for the "grounded" Laplacian → many pair queries **without§** new factorization.

- **Fiedler/Cheeger:**  
 - `eigsh(k=2, which='SM')` on **CSR§**, no dense full spectral analysis.

- **Harmonic embedding & DtN:**  
 - both with **sparse solves** instead of dense pseudo inverses.

- **Memory core:**  
 - **sketched** LS, \(O(V q)\) instead of \(O(V^2)\) with \(q\ll V\).

---

## §🔬 Evidence & diagnostics (unchanged in meaning, more efficient in implementation)

- **Markov diagnostics:**  
 - \( T\_{\text{err}} = \frac{\|\Phi\_{\Delta t} - e^{\Delta t \mathcal{L}\_{\text{REAL}}}\|\_F}{\max(1,\|e^{\Delta t \mathcal{L}\_{\text{REAL}}}\|\_F)} \)  
 - **RHP§** (CP divisibility): Choi minimum value of intermediate map \( \Phi\_{2\Delta t}\Phi\_{\Delta t}^{-1} \).  
 - **BLP**-Backflow: increases in trace distance \(D(t)\).  
 - **Entropy monotonicity** under GKSL (CPTP/PSD/TP checks).

- **"Higgs" surrogate \(m\_A\):** from \(\rho\_{\text{REAL}}\),  
 $$ m\_A \;\approx\; \bigg(\frac{1}{n}\sum\_{i} \sqrt{\rho\_{\text{REAL}}(i,i)}^{\,2}\bigg)^{1/2} $$

- **Cluster & order parameter:** Fiedler mode → binary clustering; order parameter \(M\), **Binder cumulant§** \(U\_4\)§.

---

## 📦 Export & artifacts

Automatic storage in `runs/{run_id}`:  
- `X_ideal.npy`, `X_real_harmonic.npy`  
- `L_ideal.npy`, `L_real.npy`  
- `rho_LvN.npy`, `rho_GKSL.npy`  
- `Phi_dt_super.npy`, `exp_dt_L_REAL_super.npy` (if \(|S|\ge 1\))  
- `edges.json`, `meta.json`  
- Interactive HTML plots: `ideal_graph.html`, `real_graph_harmonic.html`, `rho_ideal_heatmap.html`, `rho_real_heatmap.html`  
- Complete package as **`bundle.zip`**

---

## 🔁 Compatibility & migration

- No API changes to the main parameters.  
- **Recommendation:** Keep `S_cap` conservative for large graphs (e.g. 8-16); the outlined memory LS adapts automatically.  
- **REAL layout:** For old runs without `X_real_harmonic.npy` the deformation is not displayed; new runs generate this file automatically.

---

## 🧪 Validation (primary formulas, no simplifications)

- **§Dirichlet form & DtN§**: Schur complement exactly as in the literature.  
- **§Effective resistance**: Solution of the earthed network provides exactly the same value as via \(L^+\), but numerically more stable.  
§- **GKSL**: Standard form with \( \mathcal{L}(\rho)= -i[H,\rho]+\sum\_k L\_k \rho L\_k^\dagger - \tfrac{1}{2}\{L\_k^\dagger L\_k,\rho\} \).  
- **Channel construction**: Kraus decomposition from global unitarity with mixed \(\rho_E\) secures **CPTP** without subsequent projection.

---

## 📚 Summary

I8\_v0.2 provides **identical scientific statements** as before, but performs them **numerically much more efficiently§**, makes the **REAL deformation** with **fixed edge§** **visible§**, and ensures the **channel derivation** \(\Phi\_{\Delta t}\) **exact CPTP** - with simultaneously **stable§** evidence testing (Cheeger/Varadhan/RHP/BLP/Entropy/Binder).
