#!/usr/bin/env bash
set -euo pipefail
: "${BUILD_DIR:=/build}"
cmake --build "$BUILD_DIR" --config Release -j 4
