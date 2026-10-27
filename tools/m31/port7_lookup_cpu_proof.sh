#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/ai/logs/m31-port/port7-lookup-cpu-proof-v1
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python3 - "$OUT" <<'PY'
import json,pathlib,subprocess,sys,time,hashlib
out=pathlib.Path(sys.argv[1]);build=pathlib.Path('/mnt/ai/Xolotl-m31/build-port7-lookup/v1');exe=build/'port7_msa_sorted_lookup_check';start=time.monotonic()
with(out/'proof.jsonl').open('w')as stdout,(out/'proof.stderr').open('w')as stderr:r=subprocess.run([str(exe),'cpu'],stdout=stdout,stderr=stderr,timeout=20)
rows=[json.loads(l)for l in(out/'proof.jsonl').read_text().splitlines()];end=rows[-1]if rows else{}
d={'format':'m31-lookup-cpu-proof-v1','pass':r.returncode==0 and end.get('pass')is True,'rc':r.returncode,'seconds':time.monotonic()-start,'build_receipt':str(build/'build-receipt.json'),'elf_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'result':end,'geometry':[x for x in rows if x.get('kind')=='geometry'],'scope':'CPU component only: old expansion vs lookup vs closed integer formula. No Engine, GPU init, FA/model quality/speed/backend acceptance.'};(out/'receipt.json').write_text(json.dumps(d,indent=2)+'\n');raise SystemExit(0 if d['pass']else 1)
PY
