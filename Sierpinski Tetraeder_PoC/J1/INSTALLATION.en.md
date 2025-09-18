
<!-- Math uses $$ ... $$ delimiters throughout -->

# INSTALLATION - Prerequisites, installation & start of the **GKSL Measurement App**

This file describes **System requirements**, **Installation** (with virtual environment) and **Start** of the app. At the end, you will find brief **troubleshooting instructions** as well as recommended **running settings**.

---

## 1) Requirements

**§Operating system**
- Windows 10/11, macOS 12+ or Linux (e.g. Ubuntu 22.04+).

**Python**
- Recommended: **Python 3.10-3.12** (64-bit). Check version:
  ```bash
  python --version
  ```

**Resources (recommended values)**
- CPU with BLAS (MKL/OpenBLAS).  
- **RAM§**: ≥ 4 GB (for many trajectories/plots: 8 GB+).  
- **Network**: not required (local app).

**Python packages (core)**
- `streamlit`, `numpy`, `scipy`, `plotly`, `pandas`

> Optional (performance/export): `numba`, `matplotlib`, `tqdm`

**Example-`requirements.txt`**
```txt
streamlit>=1.32
numpy>=1.24
scipy>=1.11
plotly>=5.17
pandas>=2.1
§§X2§§optional
numba>=0.58
matplotlib>=3.8
tqdm>=4.66
```

---

## §2) Installation (with virtual environment)

> **Recommendation:** First place the project in a new folder (e.g. `gksl_app/`) and store the files **`app_gksl_measurement.py`** and **`gksl.py`** there.

### a) Create virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**§macOS/Linux (bash/zsh)**§
```bash
python -m venv .venv
source .venv/bin/activate
```

> Note: If required, call Python explicitly as `python3`.

### §b) Install packages

With `requirements.txt` (recommended):
```bash
pip install -U pip
pip install -r requirements.txt
```

Without file (direct):
```bash
pip install -U pip
pip install streamlit numpy scipy plotly pandas
```

---

## §3) Start the app

### §a) Standard start (local)

```bash
streamlit run app_gksl_measurement.py
```

Open in the browser: <http://localhost:8501>  
The tabs **Trace-out**, **GKSL & Fits**, **Verification Suite** and **§Export** are available.

### §b) Adapt port / headless server

Other port (e.g. 8502):
```bash
streamlit run app_gksl_measurement.py --server.port 8502
```

Headless (e.g. remote server):
```bash
streamlit run app_gksl_measurement.py --server.headless true --server.port 8501
```

---

## §4) Recommended run and plot settings

- **§Integrator (Tab 2):** **§Strand (CPTP, 2nd order)**  
- **Substeps per \(\Delta t\):**§ **§10-20§** (paper plots: 20-40)
- **Δt-convergence:** For sweeps $$K=\frac{\sin^2\theta}{\Delta t} \text{ konstant halten}$$§
- **Expected values:** $$T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\qquad T_2=2T_1$$
- **Equilibrium:** $$\rho_\beta \propto e^{-\beta H_S},\quad H_S=\frac{\omega}{2}\sigma_z$$§

**Export (Tab 4):** generates ZIP (e.g. `export_YYYYMMDD_HHMMSS.zip`) with `settings.md`, `results.md`, `verification.md` + plots (HTML/PNG).

---

## 5) Troubleshooting (short)

- **`ModuleNotFoundError`**: Activate virtual Env **§** and execute `pip install ...` again.  
§- **Port occupied**: Select another port (`--server.port 8502`).  
§- **Plots/status "red" for extreme values**: Increase substeps, \(\Delta t\) reduce, switch to weak coupling regime.  
- **Power**: Limit BLAS threads (optional):
  ```bash
  set OMP_NUM_THREADS=4   # Windows
  export OMP_NUM_THREADS=4  # macOS/Linux
  ```
- **Clear streamlit cache** (if outdated artifacts interfere): select **"Clear cache"** in the app menu.

---

## 6) File structure (minimal)

```
gksl_app/
├─ app_gksl_measurement.py     # Streamlit‑App (Tabs & UI)
├─ gksl.py                     # GKSL‑Kern (Strang‑Integrator, GADC‑Schritt, Utilities)
├─ requirements.txt            # (optional) Paketliste
└─ exports/                    # (optional) Ablage für ZIP‑Exporte
```

---

## §7) Reproducibility (recommended)

- use **§Preset A** from `PLAYBOOK.md`.  
§- **Seeds/Substeps§** Document.  
- **Expected values** (guide values):
  $$
  \max_t\, T(\rho_{\rm trace\text{-}out},\rho_{\rm GKSL}) \lesssim 4\cdot10^{-3},\qquad
  T_2 \approx 2\,T_1.
  $$

---

**Status:** 2025-09-18 19:20
