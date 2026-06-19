#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${CNNA_REPO_ROOT:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DEST="$ROOT/Repository/CNNA"
cp "$SRC_DIR/ConcreteGrowthIrreversibilityG1Test.lean" "$DEST/ConcreteGrowthIrreversibilityG1Test.lean"
cp "$SRC_DIR/GeneratorSpine.lean" "$DEST/GeneratorSpine.lean"
cd "$ROOT"
lake build CNNA.GeneratorSpine
