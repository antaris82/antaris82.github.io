#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-}"
if [ -z "$SRC" ]; then
  echo "usage: $0 /path/to/Generator60" >&2
  exit 2
fi
cd ~/Workspace/Lean/CNNA_Planning_Doc_Tool
cp "$SRC/LocalGrowthEdgeCriterion.lean" Repository/CNNA/
cp "$SRC/GeneratorSpine.lean" Repository/CNNA/
lake build CNNA.GeneratorSpine
