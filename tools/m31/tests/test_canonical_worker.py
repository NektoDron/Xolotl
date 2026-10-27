#!/usr/bin/env python3
"""Exercise a final compiled worker through the real Python RPC and a tiny model.

CPU backend only, including DSpark's unsupported-backend lazy fallback. This is
not native CUDA, real-sidecar DSpark math, HTTP model quality, or a speed result.
"""
import argparse,importlib.util,json,os,pathlib,tempfile
from make_fixture import make_fixture

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=pathlib.Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('canonical_real_rpc',pathlib.Path(__file__).parents[1]/'serve_m31.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 checks=0
 def check(ok,why):
  nonlocal checks
  if not ok:raise AssertionError(why)
  checks+=1
 with tempfile.TemporaryDirectory(prefix='m31-canonical-worker-') as temp:
  model=pathlib.Path(temp)/'tiny.gguf';make_fixture(model)
  base=[str(a.worker),'--model',str(model),'--backend','cpu','--ctx','32','--chunk','16','--threads','2','--persistent-cpu-pool','1','--reuse-allocations','1','--cpu-residual','1']
  plain=module.RPC(base,read_timeout=20,startup_timeout=30);fast=module.RPC(base+['--dspark-model',str(pathlib.Path(temp)/'MUST-NOT-OPEN.gguf'),'--dspark-depth','3'],read_timeout=20,startup_timeout=30)
  try:
   check('dspark' not in plain.ready,'default ready unexpectedly enables draft');check(fast.ready['dspark']['configured'],'explicit draft policy absent')
   def run(rpc,body,cancel=False):
    rpc.write({'tokens':[1,2,3,4,5,6,7,8],'max_tokens':8,'temperature':0,'seed':42,'ignore_eos':True,**body});events=[]
    while True:
     event=rpc.read();events.append(event)
     if event['event']=='sample':rpc.write({'accepted':True})
     if event['event']=='token' and body.get('interactive'):rpc.write({'continue':not cancel})
     if event['event'] in ('done','error'):return events
   cases=[{'max_tokens':0},{'max_tokens':1},{'max_tokens':8},{'max_tokens':3,'grammar_check':True},{'max_tokens':8,'interactive':True},{'max_tokens':8,'temperature':1,'top_p':1,'top_k':0,'min_p':0}]
   for body in cases:
    x=run(plain,body,cancel=body.get('interactive',False));y=run(fast,body,cancel=body.get('interactive',False))
    check(x[-1]['event']==y[-1]['event']=='done','compiled worker failed');tokens=lambda values:[e['token'] for e in values if e['event']=='token']
    check(tokens(x)==tokens(y),'CPU fallback changed generated IDs');check(x[-1]['finish_reason']==y[-1]['finish_reason'],'fallback finish changed')
    t=y[-1]['timings'];check(t['dspark_mode']=='target_fallback' and t['dspark_fallback_reason']=='backend','CPU did not disclose draft fallback')
    check(y[-1]['kv_position']==8+max(0,t['predicted_n']-1),'unemitted sample counted as fed')
   check(run(plain,{'tokens':[-1]})[-1]['event']=='error','invalid input accepted');check(run(plain,{'max_tokens':2})[-1]['event']=='done','request after error failed')
  finally:
   plain.close();fast.close()
  check(plain.process.returncode==fast.process.returncode==0,'worker did not exit cleanly on EOF')
 print(json.dumps({'pass':True,'checks':checks,'cases':len(cases),'worker':str(a.worker.resolve()),'scope':'Final ELF + actual Python RPC + actual tiny model CPU; DSpark missing-file lazy fallback, grammar ACK, cancel/error recovery. No native CUDA/DSpark quality/speed claim.'}))
if __name__=='__main__':main()
