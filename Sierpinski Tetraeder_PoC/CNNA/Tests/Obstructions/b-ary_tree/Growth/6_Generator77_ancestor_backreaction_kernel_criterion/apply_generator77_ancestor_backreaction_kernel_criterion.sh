#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${CNNA_ROOT:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DST="$ROOT/Repository/repo_snapshot/CNNA"

cp "$SRC_DIR/AncestorBackreactionKernelCriterion.lean" \
  "$DST/AncestorBackreactionKernelCriterion.lean"
cp "$SRC_DIR/GeneratorSpine.lean" "$DST/GeneratorSpine.lean"
cp "$SRC_DIR/CLAIMS.md" "$DST/CLAIMS.md"

mkdir -p "$DST/verification_scripts"
cp "$SRC_DIR/test_conductance_scaling_generalization.py" \
  "$DST/verification_scripts/conductance_scaling_generalization.py"

cd "$ROOT"
lake build CNNA.AncestorBackreactionKernelCriterion CNNA.GeneratorSpine
