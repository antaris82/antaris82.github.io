# ===============================
# ST-GRAPH: Build + Analyze (L4)
# ===============================
# Requirements:
# - PowerShell 7.x
# - Python accessible as "python"
# - run_pipeline_full_open_systems_v14.6.py in current directory
# - detector_pixels.json present

$PY = "python"
$SCRIPT = "run_pipeline_full_open_systems_v14.6.py"

# 1) BUILD — Level 4 artifacts (once)
& $PY $SCRIPT `
  --stage build `
  --artifacts out/ART_L4 `
  --level 4 `
  --t 0.6 --s 0.4 --beta 1.2 `
  --bc dirichlet --use_fixed_r

# 2) ANALYZE — Three replicated runs (ε = 2 pp) -------------------------------
# Common analyze flags
$COMMON = @(
  "--stage", "analyze",
  "--artifacts", "out/ART_L4",
  "--pixels_json", "detector_pixels.json", "--mixer", "unitary",
  "--schro", "yes", "--ham", "laplacian", "--gamma", "1.0", "--sigma0", "0.12",
  "--tmax", "3.5", "--nt", "40",
  "--open", "yes", "--dephase_site", "0.00", "--dephase_pixel", "0.00", "--det_gamma", "1.8", "--loss", "0.0",
  "--traj", "yes", "--ntraj", "3000", "--dt", "0.01", "--tmax_traj", "3.0", "--traj_scheme", "jumps", "--traj_unobs", "nojump",
  "--sg", "no",
  "--eigs_mode", "small", "--eigs_k", "64", "--save_L", "yes", "--save_kernels", "no", "--save_eigs_csv", "yes",
  "--save_open_rho", "no", "--open_series_stride", "20", "--save_modes", "no", "--save_R_diag", "no", "--save_traj_events", "no",
  "--alpha", "0.05", "--eq_margin_abs", "0.02"
)

# Seed 11
& $PY $SCRIPT $COMMON `
  --out out/ANA_L4_TI_seed11 `
  --seed 11

# Seed 13
& $PY $SCRIPT $COMMON `
  --out out/ANA_L4_TI_seed13 `
  --seed 13

# Seed 19
& $PY $SCRIPT $COMMON `
  --out out/ANA_L4_TI_seed19 `
  --seed 19

# 3) ANALYZE — Auto-ε hard test (ε = auto) -----------------------------------
$AUTO = @(
  "--stage", "analyze",
  "--artifacts", "out/ART_L4",
  "--out", "out/ANA_L4_TI_AUTO",
  "--pixels_json", "detector_pixels.json", "--mixer", "unitary",
  "--schro", "yes", "--ham", "laplacian", "--gamma", "1.0", "--sigma0", "0.12",
  "--tmax", "3.5", "--nt", "40",
  "--open", "yes", "--dephase_site", "0.00", "--dephase_pixel", "0.00", "--det_gamma", "1.8", "--loss", "0.0",
  "--traj", "yes", "--ntraj", "3000", "--dt", "0.01", "--tmax_traj", "3.0", "--seed", "13", "--traj_scheme", "jumps", "--traj_unobs", "nojump",
  "--sg", "no",
  "--eigs_mode", "small", "--eigs_k", "64", "--save_L", "yes", "--save_kernels", "no", "--save_eigs_csv", "yes",
  "--save_open_rho", "no", "--open_series_stride", "20", "--save_modes", "no", "--save_R_diag", "no", "--save_traj_events", "no",
  "--alpha", "0.05", "--eq_margin_abs", "0"
)
& $PY $SCRIPT $AUTO



# Linux/bash


#!/usr/bin/env bash
# ===============================
# ST-GRAPH: Build + Analyze (L4)
# ===============================
# Requirements:
# - bash
# - Python accessible as "python"
# - run_pipeline_full_open_systems_v14.6.py in current directory
# - detector_pixels.json present
set -euo pipefail

PY=python
SCRIPT=run_pipeline_full_open_systems_v14.6.py

# 1) BUILD — Level 4 artifacts (once)
$PY "$SCRIPT" \
  --stage build \
  --artifacts out/ART_L4 \
  --level 4 \
  --t 0.6 --s 0.4 --beta 1.2 \
  --bc dirichlet --use_fixed_r

# 2) ANALYZE — Three replicated runs (ε = 2 pp) -------------------------------
COMMON_ANALYZE=(
  --stage analyze
  --artifacts out/ART_L4
  --pixels_json detector_pixels.json --mixer unitary
  --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12
  --tmax 3.5 --nt 40
  --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0
  --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --traj_scheme jumps --traj_unobs nojump
  --sg no
  --eigs_mode small --eigs_k 64 --save_L yes --save_kernels no --save_eigs_csv yes
  --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no
  --alpha 0.05 --eq_margin_abs 0.02
)

# Seed 11
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA_L4_TI_seed11 \
  --seed 11

# Seed 13
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA_L4_TI_seed13 \
  --seed 13

# Seed 19
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA_L4_TI_seed19 \
  --seed 19

# 3) ANALYZE — Auto-ε hard test (ε = auto) -----------------------------------
$PY "$SCRIPT" \
  --stage analyze \
  --artifacts out/ART_L4 \
  --out out/ANA_L4_TI_AUTO \
  --pixels_json detector_pixels.json --mixer unitary \
  --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
  --tmax 3.5 --nt 40 \
  --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0 \
  --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --seed 13 --traj_scheme jumps --traj_unobs nojump \
  --sg no \
  --eigs_mode small --eigs_k 64 --save_L yes --save_kernels no --save_eigs_csv yes \
  --save_open_rho no --open_series_stride 20 --save_modes no --save_R_diag no --save_traj_events no \
  --alpha 0.05 --eq_margin_abs 0



Hinweise

Artefakte werden einmalig erzeugt (out/ART_L4), danach beliebig viele Analysen ohne erneuten Build.
Die drei Replikate (Seeds 11/13/19) laufen mit fester Äquivalenzmarge 2 pp.
Der Auto-ε Härtetest (--eq_margin_abs 0) prüft die TI-Äquivalenz mit der strengen, datengetriebenen Marge.