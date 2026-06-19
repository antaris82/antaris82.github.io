#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$(pwd)}"
ROOT="${2:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DEST="$ROOT/Repository/CNNA"
cd "$ROOT"
mkdir -p "$DEST"
find "$SRC" -maxdepth 1 -type f -name '*.lean' -exec cp {} "$DEST" \;
if [ -d "$DEST/Quarantine/GeneratorMicrosteps" ]; then
  rm -rf "$DEST/Quarantine/GeneratorMicrosteps"
fi
if [ -d "$DEST/Quarantine" ] && [ -z "$(find "$DEST/Quarantine" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
  rmdir "$DEST/Quarantine"
fi
lake build CNNA.GeneratorSpine
