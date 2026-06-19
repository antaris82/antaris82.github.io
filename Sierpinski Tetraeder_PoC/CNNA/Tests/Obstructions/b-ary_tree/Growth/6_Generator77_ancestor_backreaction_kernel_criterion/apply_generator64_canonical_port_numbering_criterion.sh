#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${CNNA_ROOT:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DEST="$ROOT/Repository/CNNA"
cd "$ROOT"
cp "$SRC_DIR/CanonicalPortNumberingCriterion.lean" "$DEST/CanonicalPortNumberingCriterion.lean"
cp "$SRC_DIR/GeneratorSpine.lean" "$DEST/GeneratorSpine.lean"
lake build CNNA.GeneratorSpine
