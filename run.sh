#!/usr/bin/env bash
set -euo pipefail

# Simple run wrapper. Assumes the binary was built under ./build
# Parameters mirror the previous Makefile defaults; tweak as desired.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$ROOT_DIR/build/fdock_cpu"

GRID_FILE="$ROOT_DIR/input_data/1hvr_vegl.maps.fld"
LIGAND_FILE="$ROOT_DIR/input_data/1hvrl.pdbqt"
NEV=2500
NRUN=1

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found or not executable. Build first (see README)." >&2
  exit 1
fi

exec "$BIN" -ffile "$GRID_FILE" -lfile "$LIGAND_FILE" -nev "$NEV" -nrun "$NRUN"
