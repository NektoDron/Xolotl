#!/usr/bin/env bash
set -euo pipefail
M31_ORACLE_ROOT=/mnt/ai/Xolotl-m31
M31_ORACLE_BUILD="$M31_ORACLE_ROOT/build-port5-bench-oracle"
M31_ORACLE_LOG=/mnt/ai/logs/m31-port/port5-oracle-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_ORACLE_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
if systemctl is-active -q codex-m31-ref; then
    echo 'reference measurement active; refusing build' >&2
    exit 75
fi
cmake -S "$M31_ORACLE_ROOT/tools/m31/oracle" -B "$M31_ORACLE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DM31_LLAMA_SOURCE=/mnt/ai/llama.cpp-m31pr-1026 \
  -DM31_LLAMA_BUILD=/mnt/ai/llama.cpp-m31pr-1026/build
cmake --build "$M31_ORACLE_BUILD" --target m31_oracle -j24
sha256sum "$M31_ORACLE_BUILD/m31_oracle" \
  "$M31_ORACLE_ROOT/tools/m31/oracle/m31_oracle.cpp" \
  "$M31_ORACLE_ROOT/tools/m31/m31_run.hpp"
