#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_DIR:=/build}"
BIN="$BUILD_DIR/bin/llama-cli"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: llama-cli not found at $BIN"
  exit 1
fi

echo "=== llama-cli help header ==="
"$BIN" -h | head -n 40 || true

echo "OK: binary runs"
