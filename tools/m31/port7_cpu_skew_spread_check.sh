#!/usr/bin/env bash
# Root-approved private component control only; coordinator grants clean X gap.
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=/mnt/ai/logs/m31-port/port7-cpu-skew-spread-check-v1
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 M31_CPU_BATCH_PAIR=0
unset M31_CPU_POLL M31_CPU_AFFINITY M31_CPU_FUSED_PAIRED_QUANT
python3 - "$ROOT" "$OUT" <<'PY'
import collections,hashlib,json,os,pathlib,subprocess,sys,time
root,out=map(pathlib.Path,sys.argv[1:]);b=root/'build-port7-cpu-skew/v2';exe=b/'port7_cpu_skew_replay'
allowed=os.sched_getaffinity(0);physical={};topology=[]
for cpu in sorted(allowed):
 base=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}');sibs=(base/'topology/thread_siblings_list').read_text().strip();l3=int((base/'cache/index3/id').read_text().strip())
 entry={'cpu':cpu,'sibling_list':sibs,'L3_id':l3};topology.append(entry)
 if sibs not in physical:physical[sibs]=entry
domains=collections.defaultdict(list)
for entry in physical.values():domains[entry['L3_id']].append(entry)
for items in domains.values():items.sort(key=lambda x:x['cpu'])
selected=[]
for depth in range(max(map(len,domains.values()))):
 for l3 in sorted(domains):
  if depth<len(domains[l3])and len(selected)<80:selected.append(domains[l3][depth])
assert len(selected)==80 and len({x['sibling_list']for x in selected})==80
mask=','.join(str(x['cpu'])for x in sorted(selected,key=lambda x:x['cpu']));os.environ['M31_CPU_AFFINITY']=mask
top={'allowed_cpus':sorted(allowed),'all_allowed_topology':topology,'selected80':selected,'mask':mask,'unique_physical':80,'selected_L3_counts':dict(collections.Counter(x['L3_id']for x in selected))}
(out/'topology.json').write_text(json.dumps(top,indent=2)+'\n')
cmd=[str(exe),'/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf','80','0','/mnt/ai/logs/m31-port/port7-cpu-actual-input-v1/routes.bin'];start=time.monotonic()
with(out/'records.jsonl').open('w')as stdout,(out/'stderr').open('w')as stderr:
 try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=40).returncode
 except subprocess.TimeoutExpired:rc=124
rows=[json.loads(l)for l in(out/'records.jsonl').read_text().splitlines()];phases=[x for x in rows if x.get('kind')=='cpu_skew_phase'];workers=[x for x in rows if x.get('kind')=='cpu_skew_workers'];agg=[x for x in rows if x.get('kind')=='aggregate']
last=[x for x in workers if x['layer']==59 and x['phase']=='down'];last_physical=collections.defaultdict(list)
if last:
 for w in last[0]['workers']:
  sb=(pathlib.Path(f'/sys/devices/system/cpu/cpu{w["last_cpu"]}')/'topology/thread_siblings_list').read_text().strip();last_physical[sb].append(w['ith'])
valid=rc==0 and len(phases)==30 and len(workers)==10 and len(agg)==3 and all(x.get('bitwise')for x in agg)and len(last_physical)==80 and all(w['last_cpu']in{x['cpu']for x in selected}for v in workers for w in v['workers'])
r={'format':'m31-cpu-private-skew-spread-check-v1','pass':valid,'rc':rc,'seconds':time.monotonic()-start,'command':cmd,'elf_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_proposal':json.loads((b/'proposal.json').read_text()),'environment':{'M31_CPU_AFFINITY':mask,'M31_CPU_BATCH_PAIR':'0'},'topology':top,'last_physical_core_count':len(last_physical),'last_sibling_duplicates':{k:v for k,v in last_physical.items()if len(v)>1},'phase_results':phases,'worker_results':workers,'bitcheck_token_rows':sum(x['token_rows']for x in agg),'scope':'Same frozen private CPU instrumentation/actualroutes/syntheticinput as unpinned control. No scheduler/math/default edits; oldserial/groupedtimers contaminated, not modelTPS.'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if valid else 1)
PY
