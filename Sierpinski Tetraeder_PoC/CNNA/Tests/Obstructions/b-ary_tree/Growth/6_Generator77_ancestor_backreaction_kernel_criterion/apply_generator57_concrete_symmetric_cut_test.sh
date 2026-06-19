#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${1:-$(pwd)}"
REPO_DIR="${2:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
cd "$REPO_DIR"
cp "$SRC_DIR/ConcreteSymmetricCutG1Test.lean" Repository/CNNA/
cp "$SRC_DIR/GeneratorSpine.lean" Repository/CNNA/
lake build CNNA.GeneratorSpine
