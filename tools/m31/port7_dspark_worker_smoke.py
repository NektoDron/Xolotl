#!/usr/bin/env python3
"""V5+ actual native worker: ordinary target vs greedy DSpark control/ID proof."""
import argparse
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('engine','model','sidecar','tokenizer','ids','profile','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--depth',type=int,default=3); p.add_argument('--p-min',type=float,default=0)
    args=p.parse_args()
    if os.environ.get('M31_CARD_SESSION')!='1': p.error('aictl/flock session required')
    profile=json.loads(args.profile.read_text())
    for key in profile.get('unset_environment',[]): os.environ.pop(key,None)
    os.environ.update(profile['environment'])
    spec=importlib.util.spec_from_file_location('worker_smoke_rpc',Path(__file__).with_name('serve_m31.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=array.array('i');ids.frombytes(args.ids.read_bytes())
    if sys.byteorder!='little':ids.byteswap()
    ids=list(ids);args.out.mkdir(exist_ok=False,parents=True)
    checks=[];runs={};base_tokens=[]
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    tokenizer=module.RPC([str(args.tokenizer),str(args.model)],read_timeout=30)
    try: token_info=tokenizer.ready
    finally: tokenizer.close()
    eog=list(token_info['eog']);suppressed=list(token_info.get('suppress',[]))
    def run_worker(fast):
        name='dspark' if fast else 'ordinary';log=args.out/(name+'.engine.log')
        argv=[str(args.engine),'--model',str(args.model),*profile['engine_arguments']]
        if fast:argv+=['--dspark-model',str(args.sidecar),'--dspark-depth',str(args.depth),'--dspark-p-min',str(args.p_min)]
        stream=log.open('x',buffering=1)
        original=module.subprocess.Popen
        def launch(*a,**kw):kw['stderr']=stream;return original(*a,**kw)
        module.subprocess.Popen=launch
        try:engine=module.RPC(argv,read_timeout=900,startup_timeout=900)
        finally:module.subprocess.Popen=original
        rows=[]
        try:
            check(name+'-ready-context',engine.ready['context']==40000)
            check(name+'-ready-mode',('dspark' in engine.ready)==fast)
            def request(case,**changes):
                body={'tokens':ids,'max_tokens':16,'temperature':0,'seed':42,'ignore_eos':True,
                      'stop_ids':eog,'suppress_ids':suppressed,'cache_prompt':False,**changes}
                cancel=body.pop('_cancel_after',None)
                if cancel is not None:body['interactive']=True
                stream.write('@@BEGIN '+case+'\n');stream.flush()
                offset=log.stat().st_size
                engine.write(body);events=[]
                while True:
                    event=engine.read();events.append(event)
                    if event['event']=='error':raise RuntimeError(event['error'])
                    if event['event']=='sample':engine.write({'accepted':True})
                    if event['event']=='token' and body.get('interactive'):
                        engine.write({'continue':cancel is None or event['index']<cancel})
                    if event['event']=='done':break
                stream.flush();segment=log.read_bytes()[offset:].decode(errors='replace')
                stream.write('@@END '+case+'\n');stream.flush()
                done=events[-1];timings=done['timings'];generated=[e['token'] for e in events if e['event']=='token']
                prefill=next(e for e in events if e['event']=='prefill')
                item={'case':case,'request':body,'cancel_after':cancel,'events':events,'tokens':generated,
                      'sidecar_loads':segment.count('M31 DSpark memory '),'log_segment_sha256':hashlib.sha256(segment.encode()).hexdigest(),
                      'actual_chunks':[]}
                import re
                item['actual_chunks']=[[int(a),int(b),int(c)] for a,b,c in re.findall(r'M31 prefill-plan pos=(\d+) requested=\d+ planned=(\d+) keys=(\d+)',segment)]
                check(name+'-'+case+'-one-cold-prefill',item['actual_chunks']==[[0,len(ids),((len(ids)+255)//256)*256]])
                check(name+'-'+case+'-sampling-count',timings['sampling_calls']==timings['predicted_n'])
                if fast:
                    expected=len(ids)+max(0,timings['predicted_n']-1)
                    check(name+'-'+case+'-actual-fed',done['kv_position']==expected and done['fed_generated_n']==expected-len(ids))
                    eligible=body['temperature']==0 and not body.get('grammar_check',False) and body['max_tokens']>1
                    check(name+'-'+case+'-mode',timings['dspark_mode']==('greedy' if eligible else 'target_fallback'))
                    expected_load=int(eligible and timings['predicted_n']>1)
                    check(name+'-'+case+'-lazy-load',item['sidecar_loads']==expected_load)
                    if not expected_load:check(name+'-'+case+'-no-prepare',timings.get('dspark_prepare_ms',0)==0)
                generated_file=args.out/(name+'.'+case+'.generated.i32')
                encoded=array.array('i',generated)
                if sys.byteorder!='little':encoded.byteswap()
                generated_file.write_bytes(encoded.tobytes())
                item['generated_ids_file']=str(generated_file);item['generated_ids_sha256']=sha(generated_file)
                rows.append(item)
                (args.out/(name+'.partial.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
                return generated
            greedy=request('greedy16')
            if not fast:base_tokens[:]=greedy
            check(name+'-greedy16-full',len(greedy)==16)
            if fast:check('ordinary-vs-dspark-greedy16',greedy==base_tokens)
            suppress=sorted(set(suppressed+base_tokens[:3]))
            request('suppressed16',suppress_ids=suppress)
            request('max1',max_tokens=1)
            request('temperature1',temperature=1,seed=42)
            request('grammar-fallback',grammar_check=True)
            stopped=request('stop-first',ignore_eos=False,stop_ids=sorted(set(eog+[base_tokens[0]])))
            check(name+'-first-stop-not-emitted',not stopped and rows[-1]['events'][-1]['timings']['predicted_n']==1 and rows[-1]['events'][-1]['finish_reason']=='stop')
            if fast:
                # Force a real model EOG to exercise the native EOS path before draft init.
                check('eog-present',bool(eog))
                stopped=request('real-eos-first',ignore_eos=False,stop_ids=eog,
                        suppress_ids=[token for token in range(int(engine.ready['vocab'])) if token not in set(eog)])
                check('real-eos-not-emitted',not stopped and rows[-1]['events'][-1]['timings']['predicted_n']==1 and rows[-1]['events'][-1]['finish_reason']=='stop')
                cancelled=request('cancel0',_cancel_after=0)
                check('cancel0-greedy-prefix',cancelled==base_tokens[:1])
                cancelled=request('cancel3',_cancel_after=3)
                check('cancel3-greedy-prefix',cancelled==base_tokens[:4])
                request('after-cancel16')
                check('after-cancel-greedy-parity',rows[-1]['tokens']==base_tokens)
            return {'argv':argv,'ready':engine.ready,'requests':rows,'engine_log':str(log)}
        finally:
            engine.close();stream.close()
    try:
        runs['ordinary']=run_worker(False)
        runs['dspark']=run_worker(True)
        ref={row['case']:row for row in runs['ordinary']['requests']}
        for row in runs['dspark']['requests']:
            if row['case'] in ref:
                a=ref[row['case']]
                check(row['case']+'-raw-ids-equal',row['tokens']==a['tokens'])
                check(row['case']+'-sample-count-equal',row['events'][-1]['timings']['predicted_n']==a['events'][-1]['timings']['predicted_n'])
                check(row['case']+'-same-geometry',row['actual_chunks']==a['actual_chunks'])
        receipt={'pass':True,'scope':'same-ELF/profile/prompt actual native RPC ordinary vs greedy DSpark; not HTTP performance or full quality',
                 'checks':checks,'engine_sha256':sha(args.engine),'profile_sha256':sha(args.profile),'ids_sha256':sha(args.ids),
                 'profile_environment':profile['environment'],'all_m31_environment':{k:v for k,v in sorted(os.environ.items()) if k.startswith(('M31_','GGML_'))},
                 'tokenizer_ready':token_info,'runs':runs}
        (args.out/'result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'pass':True,'checks':len(checks),'out':str(args.out)}))
    except BaseException as error:
        (args.out/'failure.json').write_text(json.dumps({'pass':False,'error':repr(error),'checks':checks,'runs':runs},ensure_ascii=False,indent=2)+'\n')
        raise

if __name__=='__main__':main()
