#!/usr/bin/env python3
"""One native RPC greedy request; lazy draft setup is inside worker TG timing."""
import argparse,array,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser()
 for k in ['engine','model','sidecar','tokenizer','ids','profile','rpc-shim','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--mode',choices=['ordinary','dspark'],required=True);p.add_argument('--depth',type=int,default=1);p.add_argument('--count',type=int,default=128);p.add_argument('--reference',type=Path)
 a=p.parse_args();assert os.environ.get('M31_CARD_SESSION')=='1','aictl/cardlock required';assert 1<=a.depth<=7 and a.count>1
 profile=json.loads(a.profile.read_text())
 for key in profile['unset_environment']:os.environ.pop(key,None)
 os.environ.update(profile['environment']);a.out.mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('frozen_m31_mini_rpc',a.rpc_shim);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ids=array.array('i');ids.frombytes(a.ids.read_bytes())
 if sys.byteorder!='little':ids.byteswap()
 tokens=list(ids);tk=m.RPC([str(a.tokenizer),str(a.model)],read_timeout=30)
 try:info=tk.ready;write(a.out/'tokenizer-ready.json',info)
 finally:tk.close()
 argv=[str(a.engine),'--model',str(a.model),*profile['engine_arguments']]
 if a.mode=='dspark':argv+=['--dspark-model',str(a.sidecar),'--dspark-depth',str(a.depth),'--dspark-p-min','0']
 req={'tokens':tokens,'max_tokens':a.count,'temperature':0,'ignore_eos':True,'seed':42,'stop_ids':info['eog'],'suppress_ids':info.get('suppress',[]),'cache_prompt':False,'interactive':True}
 write(a.out/'request.json',req);engine=m.RPC(argv,read_timeout=600,startup_timeout=600);events=[];generated=[];started=time.monotonic()
 try:
  write(a.out/'engine-ready.json',engine.ready);engine.write(req)
  with (a.out/'events.jsonl').open('x') as f:
   while True:
    event=engine.read();events.append(event);f.write(json.dumps(event,allow_nan=False)+'\n');f.flush()
    if event['event']=='error':raise RuntimeError(event['error'])
    if event['event']=='sample':raise RuntimeError('unexpected grammar sampler RPC')
    if event['event']=='token':
     assert event['index']==len(generated);generated.append(event['token']);engine.write({'continue':True})
    if event['event']=='done':break
  elapsed_ms=1000*(time.monotonic()-started);done=events[-1];tim=done['timings'];n=tim['predicted_n'];fed=n-1
  assert n==a.count==len(generated) and tim['sampling_calls']==n and done['finish_reason']=='length'
  assert done['kv_position']==len(tokens)+fed and done['fed_generated_n']==fed
  pp=next(e for e in events if e['event']=='prefill');assert pp['prompt_n']==len(tokens) and pp.get('cache_hit') is False and pp.get('actual_chunks')
  assert math.isfinite(tim['predicted_ms']) and tim['predicted_ms']>0
  if a.mode=='dspark':
   assert tim['dspark_mode']=='greedy' and tim['dspark_prepare_ms']>0 and tim['predicted_ms']>=tim['dspark_prepare_ms']
   assert tim['dspark_verified_rows']==tim['dspark_rounds']+tim['dspark_proposed']
   assert fed==tim['dspark_rounds']+tim['dspark_accepted'] and tim['dspark_cached_rows']==tim['dspark_accepted']
  raw=array.array('i',generated)
  if sys.byteorder!='little':raw.byteswap()
  dest=a.out/'generated.i32';dest.write_bytes(raw.tobytes());equal=None
  if a.reference:equal=dest.read_bytes()==a.reference.read_bytes();assert equal,'raw greedy differs from same-profile ordinary reference'
  result={'pass':True,'scope':'Single native RPC greedy request, not HTTP/om acceptance; native predicted_ms includes lazy sidecar init/seedappend and ACK waits, excludes finish cleanup measured separately.','mode':a.mode,'depth':a.depth,'argv':argv,'engine_sha256':sha(a.engine),'profile_sha256':sha(a.profile),'rpc_shim_sha256':sha(a.rpc_shim),'tokenizer_sha256':sha(a.tokenizer),'prompt_ids_sha256':sha(a.ids),'generated_ids_sha256':sha(dest),'generated_ids_file':str(dest),'reference':str(a.reference) if a.reference else None,'reference_equal':equal,'actual_chunks':pp['actual_chunks'],'timings':tim,'request_wall_including_finish_ms':elapsed_ms,'native_tps':1000*fed/tim['predicted_ms']}
  if a.mode=='dspark':
   result['acceptance']=tim['dspark_accepted']/max(1,tim['dspark_proposed']);result['fed_per_round']=fed/tim['dspark_rounds'];result['verified_per_fed']=tim['dspark_verified_rows']/fed
  write(a.out/'summary.json',result);print(json.dumps({'pass':True,'mode':a.mode,'native_tps':result['native_tps'],'setup_ms':tim.get('dspark_prepare_ms',0),'raw_equal':equal}))
 finally:engine.close()
if __name__=='__main__':main()
