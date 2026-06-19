#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${CNNA_ROOT:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DST="$ROOT/Repository/repo_snapshot/CNNA"

cp "$SRC_DIR/CycleCirculationGate.lean" "$DST/CycleCirculationGate.lean"
cp "$SRC_DIR/GeneratorSpine.lean" "$DST/GeneratorSpine.lean"
cp "$SRC_DIR/CLAIMS.md" "$DST/CLAIMS.md"

cd "$ROOT"
lake build CNNA.CycleCirculationGate CNNA.GeneratorSpine
