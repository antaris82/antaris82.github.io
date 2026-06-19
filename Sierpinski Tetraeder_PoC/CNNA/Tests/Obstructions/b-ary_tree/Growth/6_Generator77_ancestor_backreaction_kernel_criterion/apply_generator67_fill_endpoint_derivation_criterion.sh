#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$(pwd)}"
REPO="${2:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
cd "$REPO"
cp "$SRC/FillEndpointDerivationCriterion.lean" Repository/CNNA/
cp "$SRC/GeneratorSpine.lean" Repository/CNNA/
cp "$SRC/CLAIMS.md" Repository/CNNA/ 2>/dev/null || true
lake build CNNA.GeneratorSpine
