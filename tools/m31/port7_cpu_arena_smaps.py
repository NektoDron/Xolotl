#!/usr/bin/env python3
"""Read only a live M31 process owned by this task; never change memory policy."""
import argparse,hashlib,json,os,pathlib,re,time,pwd
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();proc=pathlib.Path('/proc')/str(a.pid)
exe=os.readlink(proc/'exe');assert exe.startswith(('/mnt/ai/logs/m31-port/','/mnt/ai/Xolotl-m31/')),exe
start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
status=(proc/'status').read_text();uids=next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1:];assert len(set(uids))==1 and int(uids[0])==pwd.getpwnam('nekto').pw_uid
maps=[];row=None;raw=(proc/'smaps').read_text()
for line in raw.splitlines():
 m=re.match(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\d+\s*(.*)$',line)
 if m:
  if row:maps.append(row)
  row={'start':m[1],'end':m[2],'perms':m[3],'path':m[4]}
 elif row and ':' in line:
  k,v=line.split(':',1);v=v.strip();row[k]=int(v.split()[0])*1024 if v.endswith(' kB') else v
if row:maps.append(row)
assert start==(proc/'stat').read_text().rsplit(')',1)[1].split()[19],'PID restarted during read'
whitelist={'M31_ARENA_THP','M31_ARENA_THP_FORCE','M31_GPU_REGISTER_ARENA'};env={}
for entry in (proc/'environ').read_bytes().split(b'\0'):
 key,sep,value=entry.partition(b'=')
 if key.decode(errors='replace') in whitelist:env[key.decode()]=value.decode(errors='replace')
anon=[r for r in maps if not r['path'] or r['path']=='[heap]' or r['path'].startswith('[anon:')]
large=[r for r in anon if r.get('Size',0)>=1<<30]
keys=('Size','Rss','Anonymous','AnonHugePages','KernelPageSize','MMUPageSize','VmFlags')
report={'format':'m31-own-process-arena-smaps-v1','pid':a.pid,'exe':exe,'uid':int(uids[0]),'starttime_ticks':start,'timestamp_unix':time.time(),'read_only':True,'environment':env,'sys_thp':{name:pathlib.Path('/sys/kernel/mm/transparent_hugepage'/pathlib.Path(name)).read_text().strip() for name in ('enabled','defrag')},'anonymous_size_bytes':sum(r.get('Size',0) for r in anon),'anonymous_rss_bytes':sum(r.get('Rss',0) for r in anon),'anonymous_huge_bytes':sum(r.get('AnonHugePages',0) for r in anon),'large_anon_vmas':[{k:r.get(k) for k in ('start','end','perms','path',*keys)} for r in large],'expected_arena_bytes':163754029056,'scope':'All anonymous VMAs belong to this M31 process; largest anonymous mapping(s) are inferred arena candidates. AnonHugePages is measured, not inferred from alignment or cudaHostRegister. No foreign process read/modified, no benchmark.'}
out=pathlib.Path(a.out);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('pid','anonymous_size_bytes','anonymous_rss_bytes','anonymous_huge_bytes','environment')},indent=2))
