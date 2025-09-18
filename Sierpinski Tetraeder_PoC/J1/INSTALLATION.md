
<!-- Math uses $$ ... $$ delimiters throughout -->

# INSTALLATION — Voraussetzungen, Installation & Start der **GKSL Measurement App**

Diese Datei beschreibt **Systemvoraussetzungen**, **Installation** (mit virtuellem Environment) und **Start** der App. Am Ende findest du kurze **Troubleshooting‑Hinweise** sowie empfohlene **Laufeinstellungen**.

---

## 1) Voraussetzungen

**Betriebssystem**
- Windows 10/11, macOS 12+ oder Linux (z. B. Ubuntu 22.04+).

**Python**
- Empfohlen: **Python 3.10–3.12** (64‑bit). Prüfe Version:
  ```bash
  python --version
  ```

**Ressourcen (Richtwerte)**
- CPU mit BLAS (MKL/OpenBLAS).  
- **RAM**: ≥ 4 GB (für viele Trajektorien/Plots: 8 GB+).  
- **Netzwerk**: nicht erforderlich (lokale App).

**Python‑Pakete (Kern)**
- `streamlit`, `numpy`, `scipy`, `plotly`, `pandas`

> Optional (Performance/Export): `numba`, `matplotlib`, `tqdm`

**Beispiel‑`requirements.txt`**
```txt
streamlit>=1.32
numpy>=1.24
scipy>=1.11
plotly>=5.17
pandas>=2.1
# optional
numba>=0.58
matplotlib>=3.8
tqdm>=4.66
```

---

## 2) Installation (mit virtuellem Environment)

> **Empfehlung:** Projekt zuerst in einen neuen Ordner legen (z. B. `gksl_app/`) und dort die Dateien **`app_gksl_measurement.py`** und **`gksl.py`** ablegen.

### a) Virtuelle Umgebung anlegen

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash/zsh)**
```bash
python -m venv .venv
source .venv/bin/activate
```

> Hinweis: Bei Bedarf Python explizit als `python3` aufrufen.

### b) Pakete installieren

Mit `requirements.txt` (empfohlen):
```bash
pip install -U pip
pip install -r requirements.txt
```

Ohne Datei (direkt):
```bash
pip install -U pip
pip install streamlit numpy scipy plotly pandas
```

---

## 3) Start der App

### a) Standardstart (lokal)

```bash
streamlit run app_gksl_measurement.py
```

Öffne im Browser: <http://localhost:8501>  
Die Tabs **Trace‑out**, **GKSL & Fits**, **Verification Suite** und **Export** stehen zur Verfügung.

### b) Port anpassen / Headless‑Server

Anderer Port (z. B. 8502):
```bash
streamlit run app_gksl_measurement.py --server.port 8502
```

Headless (z. B. Remote‑Server):
```bash
streamlit run app_gksl_measurement.py --server.headless true --server.port 8501
```

---

## 4) Empfohlene Lauf‑ und Plot‑Einstellungen

- **Integrator (Tab 2):** **Strang (CPTP, 2. Ordnung)**  
- **Substeps pro \(\Delta t\):** **10–20** (Paper‑Plots: 20–40)
- **Δt‑Konvergenz:** Bei Sweeps $$K=\frac{\sin^2\theta}{\Delta t} \text{ konstant halten}$$
- **Erwartungswerte:** $$T_1=\frac{1}{\gamma_\downarrow+\gamma_\uparrow},\qquad T_2=2T_1$$
- **Gleichgewicht:** $$\rho_\beta \propto e^{-\beta H_S},\quad H_S=\frac{\omega}{2}\sigma_z$$

**Export (Tab 4):** erzeugt ZIP (z. B. `export_YYYYMMDD_HHMMSS.zip`) mit `settings.md`, `results.md`, `verification.md` + Plots (HTML/PNG).

---

## 5) Troubleshooting (Kurz)

- **`ModuleNotFoundError`**: Virtuelles Env **aktivieren** und `pip install ...` erneut ausführen.  
- **Port belegt**: anderen Port wählen (`--server.port 8502`).  
- **Plots/Status „rot“ bei Extremwerten**: Substeps erhöhen, \(\Delta t\) verkleinern, ins Weak‑Coupling‑Regime wechseln.  
- **Leistung**: BLAS‑Threads begrenzen (optional):
  ```bash
  set OMP_NUM_THREADS=4   # Windows
  export OMP_NUM_THREADS=4  # macOS/Linux
  ```
- **Streamlit‑Cache leeren** (falls veraltete Artefakte stören): im App‑Menü **„Clear cache“** wählen.

---

## 6) Dateistruktur (minimal)

```
gksl_app/
├─ app_gksl_measurement.py     # Streamlit‑App (Tabs & UI)
├─ gksl.py                     # GKSL‑Kern (Strang‑Integrator, GADC‑Schritt, Utilities)
├─ requirements.txt            # (optional) Paketliste
└─ exports/                    # (optional) Ablage für ZIP‑Exporte
```

---

## 7) Reproduzierbarkeit (empfohlen)

- **Preset A** aus `PLAYBOOK.md` verwenden.  
- **Seeds/Substeps** dokumentieren.  
- **Erwartete Größen** (Richtwerte):
  $$
  \max_t\, T(\rho_{\rm trace\text{-}out},\rho_{\rm GKSL}) \lesssim 4\cdot10^{-3},\qquad
  T_2 \approx 2\,T_1.
  $$

---

**Stand:** 2025-09-18 19:20
