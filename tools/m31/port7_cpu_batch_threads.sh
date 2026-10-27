#!/usr/bin/env bash
# Prepare-only until validation grants a clean CPU window. No GPU or arena load.
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:?NEW log directory required}
BUILD=${2:-$ROOT/build-port7-cpu-pair-batch}
INPUT=${3:-/mnt/ai/logs/m31-port/port7-cpu-actual-input-v1}
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 M31_CPU_BATCH_PAIR=2
unset M31_CPU_AFFINITY M31_CPU_POLL M31_CPU_FUSED_PAIRED_QUANT
python3 - "$OUT" "$BUILD" "$INPUT" <<'PY'
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
out,build,inputs=map(pathlib.Path,sys.argv[1:]);exe=build/'port7_cpu_actual_replay';payload=inputs/'routes.bin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((inputs/'manifest.json').read_text());assert sha(payload)==manifest['payload_sha256']
start=time.monotonic();receipt={'format':'m31-cpu-batch-thread-sweep-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_used':False,'exclusive_build_lock':True,'timeout_total_s':54,'mode':2,'thread_order':[80,64,96],'input_manifest':manifest,'binary_sha256':sha(exe),'source_sha256':{str(p):sha(p) for p in sorted(build.glob('*.hpp'))+sorted(build.glob('*.cpp'))},'scope':'CPU FFN component: actual routes, synthetic activations, full expert prefault before each arm; original immutable pair-batch artifact, not model speed or live masked performance. No arena allocation, only mapped model weights.','stages':[],'pass':False}
model='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'
try:
 for threads in receipt['thread_order']:
  cmd=[str(exe),model,str(threads),'0',str(payload)];tick=time.monotonic();remaining=54-(tick-start);assert remaining>0
  with (out/f'threads{threads}.jsonl').open('w') as stdout,(out/f'threads{threads}.stderr').open('w') as stderr:
   try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=remaining).returncode
   except subprocess.TimeoutExpired:rc=124
  stage={'threads':threads,'command':cmd,'rc':rc,'seconds':time.monotonic()-tick};receipt['stages'].append(stage)
  assert rc==0,f'arm {threads} rc{rc}'
  rows=[json.loads(s) for s in (out/f'threads{threads}.jsonl').read_text().splitlines()];aggregates=[r for r in rows if r['kind']=='aggregate']
  assert len(aggregates)==3 and {r['tokens'] for r in aggregates}=={2,4,8} and all(r['bitwise'] for r in rows)
  stage['aggregates']=aggregates;stage['layer_results']=[r for r in rows if r['kind']=='layer']
 receipt['by_shape']={}
 for k in (2,4,8):
  arms={s['threads']:next(a for a in s['aggregates'] if a['tokens']==k) for s in receipt['stages']}
  reference=arms[80]
  for arm in arms.values():
   for key in ('windows','token_rows','serial_weight_bytes','unique_weight_bytes','unique_experts'):assert arm[key]==reference[key],f'workload drift {k} {key}'
  best=min(arms,key=lambda t:arms[t]['grouped_s'])
  receipt['by_shape'][str(k)]={'best_threads':best,'grouped_s':{str(t):r['grouped_s'] for t,r in arms.items()},'ratio_to_80':{str(t):reference['grouped_s']/r['grouped_s'] for t,r in arms.items()},'single_sweep_not_confidence_interval':True}
 receipt['pass']=True
except BaseException as exc:receipt['error']=repr(exc)
receipt['seconds']=time.monotonic()-start
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');raise SystemExit(0 if receipt['pass'] else 1)
PY
