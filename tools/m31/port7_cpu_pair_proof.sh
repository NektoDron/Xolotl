#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:?output directory required}
BUILD=${2:-$ROOT/build-port7-cpu-pair}
mkdir -p "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - "$ROOT" "$OUT" "$BUILD" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys,time
root,out,build=map(pathlib.Path,sys.argv[1:]);exe=build/'port7_cpu_pair_check'
cmd=[str(exe),'/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'];start=time.monotonic()
receipt={'format':'m31-port7-cpu-pair-raw-dot-proof-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'gpu_used':False,'exclusive_build_lock':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/'cpu_batched_dot.hpp',build/'port7_cpu_pair_check.cpp')},'archive_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'build-m31-cpu/ggml/src/libggml-cpu.a',root/'build-m31-cpu/ggml/src/libggml-base.a')},'scope':'Raw IQ3_XXS/IQ4_XS pair2 CPU dot numerical proof only; helper not integrated into model/FFN; no performance claim.'}
with (out/'check.json').open('w') as stdout,(out/'check.stderr').open('w') as stderr:
    try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=54).returncode
    except subprocess.TimeoutExpired:rc=124
receipt|={'rc':rc,'seconds':time.monotonic()-start,'pass':False}
if rc==0:
    proof=json.loads((out/'check.json').read_text());receipt|={'proof':proof,'pass':proof.get('pass') is True and proof.get('max_abs')==0}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if receipt['pass'] else 1)
PY
