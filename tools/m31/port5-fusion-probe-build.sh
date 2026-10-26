#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-fusion-probe-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
M31_FUSION_ROOT=/mnt/ai/Xolotl-m31
M31_FUSION_BUILD=$M31_FUSION_ROOT/build-port5-fusion
M31_FUSION_VENDOR=$M31_FUSION_ROOT/build-m31-cpu/_deps/strata_llamacpp-src
mkdir -p "$M31_FUSION_BUILD/source"
cp "$M31_FUSION_ROOT/tools/m31/port5-fusion-mmq-check.cpp" "$M31_FUSION_BUILD/source/"
cp "$M31_FUSION_ROOT/tools/m31/port5-fusion-norm-check.cpp" "$M31_FUSION_BUILD/source/"
for M31_FUSION_KIND in mmq norm; do
 /usr/bin/g++-14 -std=c++20 -O3 -I"$M31_FUSION_ROOT/include" -I"$M31_FUSION_ROOT/tools/m31" -I"$M31_FUSION_VENDOR/ggml/include" -I"$M31_FUSION_VENDOR/vendor/nlohmann" -c "$M31_FUSION_BUILD/source/port5-fusion-$M31_FUSION_KIND-check.cpp" -o "$M31_FUSION_BUILD/$M31_FUSION_KIND-check.o"
done
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/ai/Xolotl-m31');src=root/'build-m31-cuda';out=root/'build-port5-fusion'
for kind in ('mmq','norm'):
 a=shlex.split((src/'CMakeFiles/m31_generate.dir/link.txt').read_text());i=a.index('-o')
 a[i-1]=str(out/(kind+'-check.o'));a[i+1]=str(out/(kind+'-check'));a.remove('libm31_engine.a');subprocess.run(a,cwd=src,check=True)
PY
sha256sum "$M31_FUSION_BUILD/source/"* "$M31_FUSION_BUILD/"*-check
