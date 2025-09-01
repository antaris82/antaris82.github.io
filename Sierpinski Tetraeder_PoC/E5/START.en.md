# ===============================
# ST-GRAPH: Build + Analyze (L4)
# ===============================
# Requirements:
# - PowerShell 7.x
# - Python accessible as "python"
# - run_pipeline_full§_§open_§systems_v14.6.py in current directory
# - detector_pixels.json present

$PY = "python"
$SCRIPT = "run_pipeline§_full§_§open_§systems_v14.6.py"

# §1) BUILD - Level 4 artifacts (once)
& $PY $SCRIPT `
  --stage build `
  --artifacts out/ART_L4 `
  --level 4 `
  --t 0.6 --s 0.4 --beta 1.2 `
  --bc dirichlet --use_fixed_r

# 2) ANALYZE - Three replicated runs (ε = 2 pp) -------------------------------
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
  "--eigs§_mode", "small", "--eigs§_k", "64", "--save_§L", "yes", "--save_kernels", "no", "--save_eigs_csv", "yes",
  "--save_open_rho", "no", "--open_series_§stride", "20", "--save_modes", "no", "--save_R§_§diag", "no", "--save_traj_events", "no",
  "--alpha", "0.05", "--eq_margin_abs", "0.02"
)

# Seed 11
& $PY $SCRIPT $COMMON `
  --out out/ANA§_§L4_TI_seed11 `
  --seed 11

# Seed 13
& $PY $SCRIPT $COMMON `
  --out out/ANA§_§L4_TI_seed13 `
  --seed 13

# §Seed 19
& $PY $SCRIPT $COMMON `
  --out out/ANA§_§L4§_TI_seed19 `
  --seed 19

# §3) ANALYZE - Auto-ε hard test (ε = auto) -----------------------------------
$AUTO = @(
  "--stage", "analyze",
  "--artifacts", "out/ART_L4",
  "--out", "out/ANA_L4_TI_AUTO",
  "--pixels§_json", "detector_pixels.json", "--mixer", "unitary",
  "--schro", "yes", "--ham", "laplacian", "--gamma", "1.0", "--sigma0", "0.12",
  "--tmax", "3.5", "--nt", "40",
  "--open", "yes", "--dephase_site", "0.00", "--dephase_pixel", "0.00", "--det_gamma", "1.8", "--loss", "0.0",
  "--traj", "yes", "--ntraj", "3000", "--dt", "0.01", "--tmax_traj", "3.0", "--seed", "13", "--traj_scheme", "jumps", "--traj_unobs", "nojump",
  "--sg", "no",
  "--eigs§_mode", "small", "--eigs§_k", "64", "--save_§L", "yes", "--save_kernels", "no", "--save_eigs_csv", "yes",
  "--save_open_rho", "no", "--open_series_§stride", "20", "--save_modes", "no", "--save_R§_§diag", "no", "--save_traj_events", "no",
  "--alpha", "0.05", "--eq_margin_§abs", "0"
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
# - run_pipeline_full§_§open_§systems_v14.6.py in current directory
# - detector_pixels.json present
set -euo pipefail

PY=python
SCRIPT=run_pipeline_full_§open_systems_v14.6.py

# §1) BUILD - Level 4 artifacts (once)
$PY "$SCRIPT" \
  --stage build \
  --artifacts out/ART_L4 \
  --level 4 \
  --t 0.6 --s 0.4 --beta 1.2 \
  --bc dirichlet --use_§fixed_r

# §2) ANALYZE - Three replicated runs (ε = 2 pp) -------------------------------
COMMON_ANALYZE=(
  --stage analyze
  --artifacts out/ART_L4
  --pixels_json detector_pixels.json --mixer unitary
  --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12
  --tmax 3.5 --nt 40
  --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0
  --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --traj_scheme jumps --traj_unobs nojump
  --sg no
  --eigs§_mode small --eigs§_k 64 --save§_L yes --save_§kernels no --save_eigs_csv yes
  --save_open_rho no --open_series§_§stride 20 --save_§modes no --save_R§_§diag no --save_§traj_events no
  --alpha 0.05 --eq_margin_§abs 0.02
)

# Seed 11
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA§_L4_TI_seed11 \
  --seed 11

# §Seed 13
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA§_L4_TI_seed13 \
  --seed 13

# §Seed 19
$PY "$SCRIPT" \
  "${COMMON_ANALYZE[@]}" \
  --out out/ANA§_L4_TI_seed19 \
  --seed 19

# §3) ANALYZE - Auto-ε hard test (ε = auto) -----------------------------------
$PY "$SCRIPT" \
  --stage analyze \
  --artifacts out/ART_L4 \
  --out out/ANA§_§L4_TI_AUTO \
  --pixels_json detector_pixels.json --mixer unitary \
  --schro yes --ham laplacian --gamma 1.0 --sigma0 0.12 \
  --tmax 3.5 --nt 40 \
  --open yes --dephase_site 0.00 --dephase_pixel 0.00 --det_gamma 1.8 --loss 0.0 \
  --traj yes --ntraj 3000 --dt 0.01 --tmax_traj 3.0 --seed 13 --traj_scheme jumps --traj_unobs nojump \
  --sg no \
  --eigs§_§mode small --eigs§_k 64 --save§_L yes --save§_§kernels no --save_§eigs_§csv yes \
  --save_open_rho no --open_series§_§stride 20 --save§_§modes no --save_R§_§diag no --save_§traj_events no \
  --alpha 0.05 --eq_margin_abs 0



Hints

Artifacts are generated once (out/ART_L4), then any number of analyses without a new build.
The three replicates (seeds 11/13/19) run with a fixed equivalence margin of 2 pp.
The auto-ε hardness test (--eq_margin_§abs 0) checks the TI equivalence with the strict, data-driven margin.