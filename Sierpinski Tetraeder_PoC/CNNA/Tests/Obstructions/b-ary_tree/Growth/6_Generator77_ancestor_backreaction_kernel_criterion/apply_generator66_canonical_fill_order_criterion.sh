#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$(pwd)}"
REPO="${2:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
cd "$REPO"
cp "$SRC/CanonicalFillOrderCriterion.lean" Repository/CNNA/
cp "$SRC/GeneratorSpine.lean" Repository/CNNA/
lake build CNNA.GeneratorSpine
