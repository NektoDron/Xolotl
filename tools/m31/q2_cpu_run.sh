#!/usr/bin/env bash
set -euo pipefail
M31_Q2_ROOT=/mnt/ai/Xolotl-m31
M31_Q2_BUILD=${1:?build directory required}
M31_Q2_OUT=${2:?new output directory required}
mkdir "$M31_Q2_OUT"
exec >"$M31_Q2_OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset M31_CPU_AFFINITY M31_CPU_POLL M31_CPU_FUSED_PAIRED_QUANT M31_CPU_BATCH_PAIR M31_CPU_STOCK_EXPERTS
python3 - "$M31_Q2_ROOT" "$M31_Q2_BUILD" "$M31_Q2_OUT" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys,time
root,build,out=map(pathlib.Path,sys.argv[1:])
model='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf'
cmd=[str(build/'q2_cpu_check'),model,'32']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'format':'m31-q2-cpu-stock-component-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'exclusive_build_lock':True,'gpu_used':False,'source_sha256':{str(p):sha(p) for p in (build/'source').iterdir()},'binary_sha256':sha(build/'q2_cpu_check'),'build_hash_manifest':(build/'build.sha256').read_text(),'scope':'CPU components, real original Q2 weights and synthetic finite activations; no model quality/TPS acceptance.'}
start=time.monotonic()
with (out/'proof.jsonl').open('x') as stdout,(out/'proof.stderr').open('x') as stderr:
    try: rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=180).returncode
    except subprocess.TimeoutExpired: rc=124
receipt|={'seconds':time.monotonic()-start,'returncode':rc,'rows':[json.loads(line) for line in (out/'proof.jsonl').read_text().splitlines() if line.startswith('{')]}
receipt['pass']=rc==0 and any(r.get('kind')=='summary' and r.get('pass') for r in receipt['rows'])
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(0 if receipt['pass'] else 1)
PY
