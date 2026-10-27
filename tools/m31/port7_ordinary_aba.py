#!/usr/bin/env python3
"""Freeze/score a clean same-ELF ordinary cached-decode OFF/ON/OFF diagnostic.

Never builds or reserves a GPU. Uses the existing audited serial-timer freezer;
keeps its two-arm recipe intact and writes a separate three-arm card job.
"""
import argparse, hashlib, json, re, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def score(out):
 m=read(out/'aba-manifest.json'); original=read(out/'manifest.json')
 assert sha(out/'manifest.json')==m['pair_manifest_sha256']
 assert sha(out/'m31_dspark_generate-frozen')==original['engine_sha256']
 assert sha(out/'build-receipt.json')==original['build_receipt_sha256']
 assert sha(out/'cases/corpus_3324.i32')==original['prompt']['sha256']
 assert sha(out/'aba-card-job.json')==m['job_sha256']
 records={}; tokens={}; logs={}; geometry=None
 for index,arm in enumerate(['off','on','off2']):
  assert sha(out/arm/'profile.json')==m['profiles_sha256'][arm]
  rows=[json.loads(s) for s in (out/arm/'native/summary.jsonl').read_text().splitlines() if s.strip()]
  pp=[r for r in rows if r.get('mode')=='prefill_receipt']; serial=[r for r in rows if r.get('mode')=='serial']
  assert len(pp)==len(serial)==1
  pp=pp[0]; serial=serial[0]
  assert pp['draft_resident_bytes']==0 and pp['context']==40000 and pp['max_chunk']==32768
  assert serial['tokens']==128 and serial['decode_ms']>0
  if geometry is None: geometry=pp['chunks']
  assert pp['chunks']==geometry
  data=(out/arm/'native/corpus_3324.generated.i32').read_bytes(); assert len(data)==512; tokens[arm]=data
  log=out/f'aba-card-job-{index:02}.log'; lines=log.read_text().splitlines()
  hybrid=[s for s in lines if 'M31 GPU cached hybrid label=forward ' in s]; assert hybrid
  fields={k:float(v) for k,v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)',hybrid[-1])}
  if arm=='on':
   assert fields['seed_primes']==1 and fields['route_lookups']==original['expected_routes_on'] and fields['resident_routes']>0
   assert fields['seed_ms']>0 and serial['decode_ms']>=fields['seed_ms']
  else: assert fields['seed_primes']==0
  accel_lines=[s for s in lines if 'M31 GPU cached acceleration label=forward ' in s]
  accel={k:float(v) for k,v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)',accel_lines[-1])} if accel_lines else {}
  controls={'M31_GPU_CACHED_IQ3_IMMA':'iq3_imma_calls','M31_GPU_CACHED_RANK_OUTER':'rank_outer_calls','M31_GPU_CACHED_GATE_UP':'joined_gu_jobs'}
  for flag,counter in controls.items():
   enabled=arm=='on' and m['on_environment_overrides'].get(flag)=='1'
   assert accel.get(counter,0)>0 if enabled else accel.get(counter,0)==0
  records[arm]={'decode_ms':serial['decode_ms'],'tps':128000/serial['decode_ms'],'generated_sha256':hashlib.sha256(data).hexdigest(),'profile':fields,'acceleration':accel}
  logs[arm]={'path':str(log),'sha256':sha(log)}
 equal=len(set(tokens.values()))==1
 tps={k:v['tps'] for k,v in records.items()}; conservative=tps['on']/max(tps['off'],tps['off2'])
 result={'format':'m31-port7-ordinary-aba-v1','pass':equal,'generated_ids_equal':equal,'same_ELF_sha256':original['engine_sha256'],'rows':records,'prefill_chunks':geometry,'logs':logs,'on_vs_first_off':tps['on']/tps['off'],'on_vs_second_off':tps['on']/tps['off2'],'conservative_speed_ratio':conservative,'off_drift_ratio':tps['off2']/tps['off'],'clears_10_percent_against_both_off':equal and conservative>=1.1,'timer_audit':original['timer_audit'],'scope':'One clean ordinary serial 128-token OFF/ON/OFF diagnostic; primer included. Not sustained-speed or om/quality acceptance; inspect outer atomic card result separately.'}
 assert not (out/'aba-comparison.json').exists()
 write(out/'aba-comparison.json',result); print(json.dumps(result,indent=2)); return 0 if equal else 2

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['freeze','score']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--candidate',type=Path);p.add_argument('--on-env',action='append',default=[]);a=p.parse_args();out=a.directory.resolve()
 if a.command=='score': return score(out)
 assert a.candidate and not out.exists()
 helper=ROOT/'tools/m31/port7_ordinary_cached_prepare.py'
 subprocess.run([sys.executable,str(helper),'plan','--directory',str(out),'--candidate',str(a.candidate.resolve())],check=True)
 overrides={}
 for item in a.on_env:
  k,v=item.split('=',1);assert k.startswith('M31_GPU_CACHED_') and v in ['0','1'];overrides[k]=v
 for arm in ['off','on']:
  profile=read(out/arm/'profile.json')
  for k,v in overrides.items(): profile['environment'][k]=v if arm=='on' else '0'
  profile['unset_environment']=[k for k in profile['unset_environment'] if k not in profile['environment']]
  write(out/arm/'profile.json',profile)
 subprocess.run([sys.executable,str(helper),'freeze','--directory',str(out),'--candidate',str(a.candidate.resolve())],check=True)
 pair=read(out/(out.name+'-card-job.json'));assert len(pair['commands'])==3
 (out/'off2').mkdir();shutil.copy2(out/'off/profile.json',out/'off2/profile.json')
 third=list(pair['commands'][0]);i=third.index('--out')+1;third[i]=str(out/'off2/native')
 frozen=out/'aba-frozen.py';shutil.copy2(Path(__file__).resolve(),frozen)
 job={'preset':pair['preset'],'measurement':True,'commands':[pair['commands'][0],pair['commands'][1],third,[sys.executable,str(frozen),'score','--directory',str(out)]]}
 write(out/'aba-card-job.json',job)
 write(out/'aba-manifest.json',{'format':'m31-port7-ordinary-aba-plan-v1','pair_manifest_sha256':sha(out/'manifest.json'),'pair_freezer_sha256':sha(helper),'on_environment_overrides':overrides,'profiles_sha256':{k:sha(out/k/'profile.json') for k in ['off','on','off2']},'job_sha256':sha(out/'aba-card-job.json'),'frozen_scorer_sha256':sha(frozen),'arms':['off','on','off2'],'measurement':True,'no_sidecar':True,'scope':'One clean same-ELF 128-token ordinary serial request per arm, retained primer; not quality acceptance.'})
 print(json.dumps({'ready':True,'job':str(out/'aba-card-job.json')}));return 0
if __name__=='__main__':sys.exit(main())
