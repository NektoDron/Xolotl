#!/usr/bin/env python3
"""Actual HTTP facade checks against a completed same-V5 native raw-ID proof."""
import argparse
import array
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,required=True);p.add_argument('--model-alias',required=True)
    for name in ('native-result','ids','tokenizer','model','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('M31_CARD_SESSION')!='1':p.error('authorized card session required')
    native=json.loads(args.native_result.read_text())
    if not native.get('pass'):raise RuntimeError('native proof must pass before HTTP smoke')
    ids=array.array('i');ids.frombytes(args.ids.read_bytes())
    if sys.byteorder!='little':ids.byteswap()
    ids=list(ids)
    if hashlib.sha256(args.ids.read_bytes()).hexdigest()!=native['ids_sha256']:raise RuntimeError('HTTP prompt differs from native proof')
    spec=importlib.util.spec_from_file_location('http_smoke_rpc',Path(__file__).with_name('serve_m31.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokenizer=module.RPC([str(args.tokenizer),str(args.model)],read_timeout=30)
    pieces={}
    def text_for(tokens):
        values=[]
        for token in tokens:
            if token not in pieces:pieces[token]=bytes(tokenizer.call({'op':'piece','token':token})['bytes'])
            values.append(pieces[token])
        return b''.join(values).decode('utf-8',errors='replace')
    try:
        reference={row['case']:row for row in native['runs']['ordinary']['requests']}
        expected=text_for(reference['greedy16']['tokens'])
        stochastic=text_for(reference['temperature1']['tokens'])
    finally:tokenizer.close()
    base=f'http://127.0.0.1:{args.port}'
    records=[];checks=[]
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    def call(path,body=None):
        data=None if body is None else json.dumps(body).encode()
        req=urllib.request.Request(base+path,data,{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=900) as response:return json.load(response)
    def generate(name,**overrides):
        body={'prompt':ids,'max_tokens':16,'temperature':0,'seed':42,'ignore_eos':True,'cache_prompt':False,**overrides}
        started=time.monotonic();response=call('/v1/completions',body)
        records.append({'name':name,'request':body,'response':response,'wall_seconds':time.monotonic()-started})
        check(name+'-model',response['model']==args.model_alias)
        check(name+'-prompt-count',response['usage']['prompt_tokens']==len(ids))
        check(name+'-sample-count',response['timings']['sampling_calls']==response['usage']['completion_tokens'])
        return response
    try:
        models=call('/v1/models');check('owned-model-alias',any(row['id']==args.model_alias for row in models['data']))
        result=generate('greedy16')
        check('http-native-greedy-text',result['choices'][0]['text']==expected)
        check('http-greedy-full-length',result['usage']['completion_tokens']==16)
        check('http-greedy-mode',result['timings']['dspark_mode']=='greedy' and result['timings']['dspark_rounds']>0)
        check('draft-init-charged-to-decode',result['timings']['dspark_prepare_ms']>0 and result['timings']['predicted_ms']>=result['timings']['dspark_prepare_ms'])
        result=generate('max1',max_tokens=1)
        check('max1-target-fallback',result['timings']['dspark_mode']=='target_fallback' and result['timings']['dspark_fallback_reason']=='max_tokens_le_one')
        check('max1-no-draft-load',result['timings'].get('dspark_prepare_ms',0)==0 and result['timings'].get('dspark_rounds',0)==0)
        result=generate('temperature1',temperature=1)
        check('sampling-target-fallback',result['timings']['dspark_mode']=='target_fallback' and result['timings']['dspark_fallback_reason']=='sampling_temperature')
        check('sampling-no-draft-load',result['timings'].get('dspark_prepare_ms',0)==0)
        check('http-native-stochastic-text',result['choices'][0]['text']==stochastic)
        if expected:
            result=generate('string-stop',stop=expected[:1])
            check('stop-string-hidden',result['choices'][0]['text']=='' and result['choices'][0]['finish_reason']=='stop')
        # Client cancellation may leave one already-ACKed native token in flight.
        # The next request must be cold and healthy; native proof checks exact KV boundaries.
        body={'prompt':ids,'max_tokens':64,'temperature':0,'ignore_eos':True,'cache_prompt':False,'stream':True}
        connection=http.client.HTTPConnection('127.0.0.1',args.port,timeout=900)
        connection.request('POST','/v1/completions',json.dumps(body),{'Content-Type':'application/json'})
        response=connection.getresponse();check('stream-http200',response.status==200)
        seen=False
        for _ in range(200):
            line=response.readline()
            if not line:break
            if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                event=json.loads(line[6:])
                if event.get('error'):raise RuntimeError(event['error'])
                if event.get('choices',[{}])[0].get('text'):
                    seen=True;break
        response.close();connection.close();check('cancel-after-visible-content',seen)
        for attempt in range(20):
            try:
                result=generate('after-cancel-max1',max_tokens=1);break
            except urllib.error.HTTPError as error:
                if error.code!=503:raise
                time.sleep(.25)
        else:raise RuntimeError('server did not drain cancellation')
        check('next-request-cold-fallback',result['timings']['dspark_mode']=='target_fallback' and result['timings']['dspark_fallback_reason']=='max_tokens_le_one')
        check('next-request-first-token',result['choices'][0]['text']==text_for_cached_first(reference,pieces))
        chat={'messages':[{'role':'user','content':'Reply with the word HTTP_OK.'}],
              'temperature':0,'max_tokens':16,'ignore_eos':True,'cache_prompt':False,'reasoning_effort':'none'}
        response=call('/v1/chat/completions',chat)
        records.append({'name':'chat-facade','request':chat,'response':response})
        check('chat-role',response['choices'][0]['message']['role']=='assistant')
        check('chat-greedy-mode',response['timings']['dspark_mode']=='greedy')
        receipt={'pass':True,'scope':'actual standalone HTTP M3 facade; native raw-ID proof remains separate; not om/GSM/tools acceptance',
                 'checks':checks,'records':records,'native_receipt_sha256':hashlib.sha256(args.native_result.read_bytes()).hexdigest(),
                 'ids_sha256':hashlib.sha256(args.ids.read_bytes()).hexdigest(),'model_alias':args.model_alias}
        args.out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'pass':True,'checks':len(checks),'out':str(args.out)}))
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'pass':False,'error':repr(error),'checks':checks,'records':records},ensure_ascii=False,indent=2)+'\n')
        raise

def text_for_cached_first(reference,pieces):
    return pieces[reference['greedy16']['tokens'][0]].decode('utf-8',errors='replace')

if __name__=='__main__':main()
