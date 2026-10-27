#!/usr/bin/env bash
# Prepared only. Clean CPU window required, no GPU used.
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=/mnt/ai/logs/m31-port/port7-cpu-skew-check-v1
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 M31_CPU_BATCH_PAIR=0
unset M31_CPU_POLL M31_CPU_AFFINITY M31_CPU_FUSED_PAIRED_QUANT
python3 - "$ROOT" "$OUT" <<'PY'
import json,pathlib,subprocess,sys,time,hashlib
r,out=map(pathlib.Path,sys.argv[1:]);b=r/'build-port7-cpu-skew/v2';exe=b/'port7_cpu_skew_replay';cmd=[str(exe),'/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf','80','0','/mnt/ai/logs/m31-port/port7-cpu-actual-input-v1/routes.bin'];start=time.monotonic()
with(out/'records.jsonl').open('w')as stdout,(out/'stderr').open('w')as stderr:
 try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=40).returncode
 except subprocess.TimeoutExpired:rc=124
rows=[json.loads(l)for l in(out/'records.jsonl').read_text().splitlines()];phases=[x for x in rows if x.get('kind')=='cpu_skew_phase'];workers=[x for x in rows if x.get('kind')=='cpu_skew_workers'];agg=[x for x in rows if x.get('kind')=='aggregate']
d={'format':'m31-cpu-private-skew-check-v1','pass':rc==0 and len(phases)==30 and len(workers)==10 and len(agg)==3 and all(x.get('bitwise')for x in agg),'rc':rc,'seconds':time.monotonic()-start,'command':cmd,'elf_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_proposal':json.loads((b/'proposal.json').read_text()),'phase_results':phases,'worker_results':workers,'bitcheck_token_rows':sum(x['token_rows']for x in agg),'scope':'Private frozen-header instrumentation, actualfirst32routes per8cases,syntheticinputs,5layers. Ignore oldserial/groupedtimers: observer aggregation contaminates them. Not model TPS/flat-profile wall attribution.'};(out/'receipt.json').write_text(json.dumps(d,indent=2)+'\n');raise SystemExit(0 if d['pass']else 1)
PY
