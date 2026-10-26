#!/usr/bin/env bash
set -euo pipefail
M31_SHARED_ROOT=/mnt/ai/Xolotl-m31
M31_SHARED_LOG=/mnt/ai/logs/m31-port/cpu-shared-overlap-serve-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_SHARED_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
# Preserve tested proof ELF before rebuilding this directory without instrumentation.
python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json
root=Path('/mnt/ai/Xolotl-m31');log=Path('/mnt/ai/logs/m31-port')
source=root/'build-cpu-shared-overlap/m31_generate';frozen=log/'cpu-shared-overlap-proof-m31_generate-v1'
with source.open('rb') as src,frozen.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
frozen.chmod(0o555)
(log/'cpu-shared-overlap-proof-elf-v1.json').write_text(json.dumps({'path':str(frozen),'sha256':hashlib.file_digest(frozen.open('rb'),'sha256').hexdigest(),'engine_source_sha256':hashlib.sha256((root/'tools/m31/m31_engine.cpp').read_bytes()).hexdigest(),'compile_defines':['M31_SHARED_OVERLAP_TESTING=1','M31_SHARED_OVERLAP_NVTX=1']},indent=2)+'\n')
PY
cmake -S "$M31_SHARED_ROOT/tools/m31/cpu_shared_overlap_build" -B "$M31_SHARED_ROOT/build-cpu-shared-overlap" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda -DM31_SHARED_OVERLAP_NVTX=OFF -DM31_SHARED_OVERLAP_TESTING=OFF
cmake --build "$M31_SHARED_ROOT/build-cpu-shared-overlap" --target m31_generate m31_serve m31_inspect -j24
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python3 "$M31_SHARED_ROOT/tools/m31/tests/test_math.py" --engine "$M31_SHARED_ROOT/build-cpu-shared-overlap/m31_generate" --inspect "$M31_SHARED_ROOT/build-cpu-shared-overlap/m31_inspect"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/mnt/ai/Xolotl-m31');build=root/'build-cpu-shared-overlap'
flags=(build/'CMakeFiles/m31_overlap_engine.dir/flags.make').read_text()
assert 'M31_SHARED_OVERLAP_TESTING' not in flags and 'M31_SHARED_OVERLAP_NVTX' not in flags,flags
with (Path('/mnt/ai/logs/m31-port')/'cpu-shared-overlap-serve-freeze-v1.json').open('x') as out:
 json.dump({'engine_source_sha256':hashlib.sha256((root/'tools/m31/m31_engine.cpp').read_bytes()).hexdigest(),'compile_flags':flags,'binaries':{str(build/name):hashlib.file_digest((build/name).open('rb'),'sha256').hexdigest() for name in ['m31_generate','m31_serve']}},out,indent=2)
 out.write('\n')
PY
