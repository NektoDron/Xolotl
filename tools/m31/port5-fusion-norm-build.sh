#!/bin/bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-fusion-norm-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
ROOT=/mnt/ai/Xolotl-m31
/usr/bin/g++-14 -std=c++20 -O3 -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" -c "$ROOT/tools/m31/port5-fusion-norm-check.cpp" -o "$ROOT/build-port5-cache/fusion-norm-check.o"
python3 - <<'INNER'
import shlex,subprocess
from pathlib import Path
r=Path('/mnt/ai/Xolotl-m31');src=r/'build-m31-cuda';out=r/'build-port5-cache'
a=shlex.split((src/'CMakeFiles/m31_generate.dir/link.txt').read_text());i=a.index('-o');a[i-1]=str(out/'fusion-norm-check.o');a[i+1]=str(out/'fusion-norm-check');a.remove('libm31_engine.a');subprocess.run(a,cwd=src,check=True)
INNER
