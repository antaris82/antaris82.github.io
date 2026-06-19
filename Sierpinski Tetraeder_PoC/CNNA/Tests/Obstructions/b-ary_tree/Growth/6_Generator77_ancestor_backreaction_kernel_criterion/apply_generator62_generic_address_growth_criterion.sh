#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${1:-$(pwd)}"
ROOT="${CNNA_ROOT:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DEST="$ROOT/Repository/CNNA"
cp "$SRC_DIR/GenericAddressGrowthCriterion.lean" "$DEST/GenericAddressGrowthCriterion.lean"
cp "$SRC_DIR/GeneratorSpine.lean" "$DEST/GeneratorSpine.lean"
cd "$ROOT"
lake build CNNA.GeneratorSpine
