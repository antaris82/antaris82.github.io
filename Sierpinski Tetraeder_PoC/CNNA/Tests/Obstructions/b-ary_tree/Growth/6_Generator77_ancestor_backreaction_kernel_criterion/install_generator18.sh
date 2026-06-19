#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/Workspace/Lean/CNNA_Planning_Doc_Tool}"
DST="$ROOT/Repository/repo_snapshot/CNNA"
mkdir -p "$DST"
cp "$(dirname "$0")"/*.lean "$DST"/
echo "Copied Generator18_fixed Lean files to $DST"
