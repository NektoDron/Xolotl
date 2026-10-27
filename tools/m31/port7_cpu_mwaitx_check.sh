#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/ai/logs/m31-port/port7-cpu-mwaitx-check-v1
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES=
python3 - "$OUT" <<'PY'
import json,pathlib,subprocess,time,sys,hashlib
out=pathlib.Path(sys.argv[1]);b=pathlib.Path('/mnt/ai/Xolotl-m31/build-port7-cpu-mwaitx/v2');exe=b/'port7_cpu_mwaitx_probe';rows=[];start=time.monotonic()
# core0/core1 share CCD; core0/core8 have distinct L3; no SMT siblings.
# Fixed raw TSC clocks are recorded; no unmeasured ns conversion is claimed.
arms=[(sender,waiter,gap,clocks,mode,0)for sender,waiter in[(0,1),(0,8)]for gap in[0,5000,50000]for clocks in[10000,50000]for mode in[0,1,2]]
arms.extend([(0,1,5000,10000,1,1),(0,8,5000,10000,2,1)])
for i,(sender,waiter,gap,clocks,mode,forced)in enumerate(arms):
 if time.monotonic()-start>25:raise SystemExit('global25s budget exceeded')
 cmd=[str(exe),str(mode),str(sender),str(waiter),'2000',str(gap),str(clocks),str(forced)]
 run=subprocess.run(cmd,capture_output=True,text=True,timeout=3)
 (out/f'arm-{i:02d}.stdout').write_text(run.stdout);(out/f'arm-{i:02d}.stderr').write_text(run.stderr)
 try:r=json.loads(run.stdout)
 except Exception:r={'parse_error':True}
 r.update(command=cmd,rc=run.returncode,sender_cpu=sender,waiter_cpu=waiter,forced_fallback=bool(forced));rows.append(r)
 (out/'arms.json').write_text(json.dumps(rows,indent=2)+'\n')
valid=lambda r:r['rc']==0 and r.get('iterations')==2000 and r.get('migrations')==0 and(r['mode']==0 or r.get('used_mwaitx')or r.get('probe_status')!=0)and(not r['forced_fallback']or not r.get('used_mwaitx'))
r={'pass':len(rows)==38 and all(map(valid,rows)),'seconds':time.monotonic()-start,'elf_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'build_receipt':json.loads((b/'build-receipt.json').read_text()),'arms':rows,'mwaitx_actually_exercised':any(x.get('used_mwaitx')for x in rows),'power_measured':False,'scope':'two-core publish-to-observe component, rawTSCtimerclocks; not pureinstruction latency, not 79worker/modelTG/energy proof'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['pass']else 1)
PY
