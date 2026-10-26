#!/usr/bin/env bash
set -euo pipefail
M31_READ_ROOT=/mnt/ai/Xolotl-m31
M31_READ_LOG=/mnt/ai/logs/m31-port/cpu-host-read-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_READ_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_READ_ROOT/tools/m31/cpu_host_read_build" -B "$M31_READ_ROOT/build-cpu-host-read" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build "$M31_READ_ROOT/build-cpu-host-read" -j24
python3 "$M31_READ_ROOT/tools/m31/cpu_host_read_test.py" --engine-check "$M31_READ_ROOT/build-cpu-host-read/cpu_host_read_check"
