#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:?output directory required}
BUILD=${2:-$ROOT/build-port7-cpu-pair-batch}
INPUT=${3:-/mnt/ai/logs/m31-port/port7-cpu-actual-input-v1}
mkdir -p "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset M31_CPU_AFFINITY M31_CPU_POLL M31_CPU_FUSED_PAIRED_QUANT M31_CPU_BATCH_PAIR
python3 - "$ROOT" "$OUT" "$BUILD" "$INPUT" <<'PY'
import datetime,hashlib,json,os,pathlib,platform,subprocess,sys,time
root,out,build,inputs=map(pathlib.Path,sys.argv[1:]);payload=inputs/'routes.bin'
model='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'
cases=[]
for mode in (0,1,2):
    cases.append((f'component-mode{mode}',mode,[str(build/'port7_cpu_batch_check'),model,'80','0']))
    cases.append((f'actual-mode{mode}',mode,[str(build/'port7_cpu_actual_replay'),model,'80','0',str(payload)]))
start=time.monotonic();cpuinfo=pathlib.Path('/proc/cpuinfo').read_text();first=cpuinfo.split('\n\n')[0]
flags=next(line.split(':',1)[1].split() for line in first.splitlines() if line.startswith('flags'))
receipt={'format':'m31-port7-cpu-pair-batch-proof-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_used':False,'exclusive_build_lock':True,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(build.glob('*.hpp'))+list(build.glob('*.cpp'))+list(build.glob('*.sh'))},'input_manifest':json.loads((inputs/'manifest.json').read_text()),'fp_compile_flags':['-O3','-DNDEBUG','-frounding-math','-fno-fast-math'],'pair_target_attribute':['avx2','fma','noinline'],'global_march':False,'architecture':platform.machine(),'runtime_isa':{name:name in flags for name in ('avx2','fma')},'stages':[],'scope':'Exact FFN component only; actual routes/weights with synthetic identical activations; no model TPS/quality claim.'}
assert all(receipt['runtime_isa'].values())
assert hashlib.sha256(payload.read_bytes()).hexdigest()==receipt['input_manifest']['payload_sha256']
for name,mode,cmd in cases:
    t=time.monotonic()
    with (out/f'{name}.jsonl').open('w') as stdout,(out/f'{name}.stderr').open('w') as stderr:
        try:rc=subprocess.run(cmd,env=os.environ|{'M31_CPU_BATCH_PAIR':str(mode)},stdout=stdout,stderr=stderr,timeout=max(1,54-(time.monotonic()-start))).returncode
        except subprocess.TimeoutExpired:rc=124
    stage={'name':name,'mode':mode,'environment':{'M31_CPU_BATCH_PAIR':str(mode)},'command':cmd,'binary_sha256':hashlib.sha256(pathlib.Path(cmd[0]).read_bytes()).hexdigest(),'rc':rc,'seconds':time.monotonic()-t}
    if rc==0:stage['rows']=[json.loads(l) for l in (out/f'{name}.jsonl').read_text().splitlines()]
    receipt['stages'].append(stage)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if rc:break
receipt|={'seconds':time.monotonic()-start,'pass':len(receipt['stages'])==6 and all(s['rc']==0 for s in receipt['stages'])}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if receipt['pass'] else 1)
PY
