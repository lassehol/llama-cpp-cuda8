#!/usr/bin/env bash
set -euo pipefail

echo "=== Host kernel ==="
uname -a || true

echo "=== CPU flags (container view) ==="
grep -m1 -E 'model name|flags' /proc/cpuinfo || true

echo "=== NVIDIA devices (container) ==="
ls -l /dev/nvidia* || true

echo "=== CUDA toolchain ==="
which nvcc || true
nvcc --version || true

echo "=== Toolchain ==="
cmake --version || true
g++ --version || true
