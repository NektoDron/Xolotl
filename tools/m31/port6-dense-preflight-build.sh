#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port6-dense-preflight-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
root=/mnt/ai/Xolotl-m31
inc=$root/build-m31-cpu/_deps/strata_llamacpp-src
/usr/bin/g++-14 -std=c++20 -O2 -DNDEBUG -DGGML_USE_CPU -I"$root/tools/m31" -I"$root/include" -I"$inc/ggml/include" -I"$inc/ggml/src" -I"$inc/vendor/nlohmann" -c "$root/tools/m31/port6-dense-preflight-check.cpp" -o "$root/build-port6-dense/preflight-check.o"
python3 - <<'PY'
from pathlib import Path
import shlex,subprocess
r=Path('/mnt/ai/Xolotl-m31');b=r/'build-m31-cpu';o=r/'build-port6-dense'
a=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());i=a.index('-o');a[i-1]=str(o/'preflight-check.o');a[i+1]=str(o/'preflight-check');subprocess.run(a,cwd=b,check=True)
with Path('/mnt/ai/logs/m31-port/port6-dense-preflight-cpu-v1.json').open('x') as f:subprocess.run([str(o/'preflight-check')],stdout=f,check=True)
PY
