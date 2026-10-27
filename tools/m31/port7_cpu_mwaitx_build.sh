#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=$ROOT/build-port7-cpu-mwaitx/v2
LOG=/mnt/ai/logs/m31-port/port7-cpu-mwaitx-build-v2
mkdir -p "$LOG"
exec >"$LOG/build.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$ROOT/build-port7-cpu-mwaitx"
mkdir "$OUT"
cp "$ROOT/tools/m31/port7_cpu_mwaitx_proposal.cpp" "$OUT/"
python3 - "$OUT" <<'PY'
import pathlib,subprocess,json,sys,time,hashlib
out=pathlib.Path(sys.argv[1]);src=out/'port7_cpu_mwaitx_proposal.cpp';elf=out/'port7_cpu_mwaitx_probe'
cmd=['/usr/bin/g++-14','-std=c++20','-O2','-pthread',str(src),'-o',str(elf)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(src);start=time.monotonic();rc=subprocess.run(cmd).returncode
r={'rc':rc,'seconds':time.monotonic()-start,'command':cmd,'source_sha256':before,'source_after_sha256':sha(src),'elf_sha256':sha(elf)if elf.exists()else None,'scope':'One isolated CXX. Baseline no march/native/mmwaitx. Only target(mwaitx) helper has feature instructions. No ggml, Engine, vendor or CUDA links.'}
(out/'build-receipt.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(rc)
PY
