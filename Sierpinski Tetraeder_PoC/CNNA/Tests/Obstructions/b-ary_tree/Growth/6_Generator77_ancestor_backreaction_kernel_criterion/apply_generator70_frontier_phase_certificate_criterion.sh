#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="${2:-$PWD}"
TARGET_DIR="$REPO_ROOT/CNNA"

if [ ! -d "$TARGET_DIR" ]; then
  echo "missing target CNNA directory: $TARGET_DIR" >&2
  exit 1
fi

cp "$SRC_DIR/FrontierPhaseCertificateCriterion.lean" \
  "$TARGET_DIR/FrontierPhaseCertificateCriterion.lean"
cp "$SRC_DIR/GeneratorSpine.lean" \
  "$TARGET_DIR/GeneratorSpine.lean"
cp "$SRC_DIR/CLAIMS.md" \
  "$TARGET_DIR/GENERATOR_CLAIMS.md"

lake build CNNA.FrontierPhaseCertificateCriterion CNNA.GeneratorSpine
