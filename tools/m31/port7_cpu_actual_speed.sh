#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:?output directory required}
BUILD=${2:-$ROOT/build-port7-cpu-actual}
INPUT=${3:-/mnt/ai/logs/m31-port/port7-cpu-actual-input-v1}
mkdir -p "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset M31_CPU_AFFINITY M31_CPU_POLL M31_CPU_FUSED_PAIRED_QUANT
python3 - "$OUT" "$BUILD" "$INPUT" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys,time
out,build,inputs=map(pathlib.Path,sys.argv[1:]);exe=build/'port7_cpu_actual_replay';payload=inputs/'routes.bin'
model='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'
cmd=[str(exe),model,'80','0',str(payload)];start=time.monotonic()
receipt={'format':'m31-port7-cpu-actual-route-replay-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'gpu_used':False,'exclusive_build_lock':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/'cpu_batched_verify.hpp',build/'cpu_fused_decode.hpp',build/'port7_cpu_actual_replay.cpp')},'input_manifest':json.loads((inputs/'manifest.json').read_text()),'synthetic_activations':True,'scope':'Actual decode IDs and F32 route weights; component FFN only, no model TPS or full DSpark speed claim.'}
assert hashlib.sha256(payload.read_bytes()).hexdigest()==receipt['input_manifest']['payload_sha256']
with (out/'check.jsonl').open('w') as stdout,(out/'check.stderr').open('w') as stderr:
    try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=54).returncode
    except subprocess.TimeoutExpired:rc=124
receipt|={'rc':rc,'seconds':time.monotonic()-start,'pass':False}
if rc==0:
    rows=[json.loads(line) for line in (out/'check.jsonl').read_text().splitlines()];aggregates=[r for r in rows if r['kind']=='aggregate']
    receipt|={'aggregates':aggregates,'layers':[r for r in rows if r['kind']=='layer'],'pass':len(aggregates)==3 and {r['tokens'] for r in aggregates}=={2,4,8} and all(r['bitwise'] for r in rows)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if receipt['pass'] else 1)
PY
