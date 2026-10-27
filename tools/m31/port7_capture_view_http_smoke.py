#!/usr/bin/env python3
"""One loaded HTTP worker: bounded lease, real cancel/drain and next cold request."""
import argparse
import array
import hashlib
import http.client
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--alias',required=True)
    parser.add_argument('--ids',type=Path,required=True)
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();packed=array.array('i');packed.frombytes(args.ids.read_bytes());ids=list(packed)
    base=f'http://127.0.0.1:{args.port}';checks=[];records=[]
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    def events():return [json.loads(line) for line in args.events.read_text().splitlines()] if args.events.exists() else []
    def post(body):
        request=urllib.request.Request(base+'/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=900) as response:return json.load(response)
    def generate(name,**changes):
        body={'prompt':ids,'max_tokens':8,'temperature':0,'ignore_eos':True,'cache_prompt':False,**changes}
        for _ in range(50):
            try:response=post(body);break
            except urllib.error.HTTPError as error:
                if error.code!=503:raise
                time.sleep(.2)
        else:raise RuntimeError('HTTP cancellation did not drain')
        native=events()[-1];done=native['native_done'];timings=done['timings'];tokens=native['tokens']
        check(name+'-owned-model',response['model']==args.alias)
        check(name+'-cold-prompt',timings['prompt_n']==len(ids) and response['usage']['prompt_tokens']==len(ids))
        check(name+'-full-native-ID-count',len(tokens)==timings['predicted_n'] and native['native_terminal_unemitted_samples']==0)
        check(name+'-sampler-count',timings['sampling_calls']==len(tokens))
        check(name+'-ACK-count',len(native['acks'])==len(tokens)==timings['interactive_waits'])
        check(name+'-exact-fed-boundary',done['kv_position']==len(ids)+max(len(tokens)-1,0))
        records.append({'name':name,'body':body,'response':response,'native':native})
        return response,native
    def cancel(name,count):
        first_event=len(events())
        body={'prompt':ids,'max_tokens':64,'temperature':0,'ignore_eos':True,'cache_prompt':False,'stream':True}
        connection=http.client.HTTPConnection('127.0.0.1',args.port,timeout=900)
        connection.request('POST','/v1/completions',json.dumps(body),{'Content-Type':'application/json'})
        response=connection.getresponse();check(name+'-HTTP200',response.status==200)
        seen=0
        for _ in range(2000):
            line=response.readline()
            if not line:break
            if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                frame=json.loads(line[6:])
                if frame.get('error'):raise RuntimeError(frame['error'])
                if frame.get('choices',[{}])[0].get('text'):
                    seen+=1
                    if seen==count:break
        check(name+'-visible-boundary',seen==count)
        owned_socket=connection.sock or getattr(getattr(response.fp,'raw',None),'_sock',None)
        if owned_socket:
            try:owned_socket.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        response.close();connection.close()
        records.append({'name':name,'visible_chunks_before_disconnect':seen})
        return first_event
    try:
        response,native=generate('max1-before',max_tokens=1);first=native['tokens'][0]
        check('max1-no-draft-load',native['native_done']['timings']['dspark_mode']=='target_fallback' and native['native_done']['timings'].get('dspark_prepare_ms',0)==0)
        reference,native=generate('greedy8-before');expected=native['tokens']
        check('greedy8-same-first-native-ID',expected[0]==first)
        check('greedy8-view-prepared',native['native_done']['timings']['dspark_mode']=='greedy' and native['native_done']['timings']['dspark_prepare_ms']>0)
        for count in (1,3):
            first_event=cancel(f'cancel{count}',count)
            _,native=generate(f'next-max1-after-cancel{count}',max_tokens=1)
            check(f'cancel{count}-actual-native-cancel-ACK',any(row['event']=='done' and row['acks'] and not row['acks'][-1] for row in events()[first_event:]))
            check(f'cancel{count}-next-cold-first-ID',native['tokens']==[first])
            check(f'cancel{count}-next-no-draft-load',native['native_done']['timings']['dspark_mode']=='target_fallback' and native['native_done']['timings'].get('dspark_prepare_ms',0)==0)
            _,native=generate(f'next-greedy8-after-cancel{count}')
            check(f'cancel{count}-next-native-greedy8-identical',native['tokens']==expected)
        if reference['choices'][0]['text']:
            response,native=generate('string-stop',stop=reference['choices'][0]['text'][:1])
            check('text-stop-hidden',response['choices'][0]['text']=='' and response['choices'][0]['finish_reason']=='stop')
        _,native=generate('max0',max_tokens=0)
        check('max0-zero-ID-no-draft',native['tokens']==[] and native['native_done']['timings'].get('dspark_prepare_ms',0)==0)
        cancelled=[]
        for native in events():
            if native['event']=='done' and native['acks'] and not native['acks'][-1]:
                done=native['native_done'];timings=done['timings']
                check('cancel-native-sampler-ID-count',len(native['tokens'])==timings['predicted_n']==timings['sampling_calls'])
                check('cancel-native-ACK-boundary',len(native['acks'])==len(native['tokens'])==timings['interactive_waits'])
                check('cancel-native-KV-rollback',done['kv_position']==len(ids)+max(timings['predicted_n']-1,0))
                cancelled.append({'predicted_n':timings['predicted_n'],'kv_position':done['kv_position'],'prepared':timings.get('dspark_prepare_ms',0)>0})
        check('actual-cancel-controls-observed',len(cancelled)>=2)
        result={'pass':True,'scope':'actual single-loaded bounded feature-view HTTP worker: native IDs/ACK/fed rollback, cancel1/3 then cold healthy requests, max0/1 lazy and text stop; not throughput/GSM/tools acceptance','checks':checks,'records':records,'cancelled':cancelled,'ids_sha256':hashlib.sha256(args.ids.read_bytes()).hexdigest()}
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'cancelled':cancelled}))
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'pass':False,'error':repr(error),'checks':checks,'records':records},ensure_ascii=False,indent=2)+'\n');raise


if __name__=='__main__':main()
