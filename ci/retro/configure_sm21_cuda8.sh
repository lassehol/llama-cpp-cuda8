#!/usr/bin/env bash
set -euo pipefail

: "${SRC_DIR:=/repo}"
: "${BUILD_DIR:=/build}"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=21
