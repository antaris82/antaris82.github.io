#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$(pwd)}"
cd ~/Workspace/Lean/CNNA_Planning_Doc_Tool
cp "$SRC/AbstractAddressTypeGrowthCriterion.lean" Repository/CNNA/
cp "$SRC/GeneratorSpine.lean" Repository/CNNA/
lake build CNNA.GeneratorSpine
