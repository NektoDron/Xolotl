#!/usr/bin/env bash
set -euo pipefail
M31_VISION_ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/vision-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_VISION_ROOT/tools/m31/vision_build" -B "$M31_VISION_ROOT/build-vision-prepare" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build "$M31_VISION_ROOT/build-vision-prepare" -j24
