#!/usr/bin/env python3
"""Owned GPU-card worker job: native HTTP vision acceptance and tool smoke.

Launch only from an aictl-owned vision card worker under the coordinator's
exclusive speed lock. This script neither reserves/releases a card nor changes aictl.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from profile_plan import load_profile
import vision_validation as V

ROOT = Path(__file__).resolve().parents[2]
BOOL_FLAGS = {'--reuse-allocations','--persistent-cpu-pool','--cpu-residual',
              '--cpu-decode-cache','--dense-decode-cache','--profile'}


def standalone_arguments(arguments, context, experts):
    """Native profile uses flag/value pairs; standalone launcher uses bool flags."""
    if len(arguments) % 2:
        raise ValueError('native profile needs flag/value pairs')
    result = []
    for flag, value in zip(arguments[::2], arguments[1::2]):
        if flag == '--ctx':
            value = str(context)
        if flag == '--gpu-experts-gib' and experts is not None:
            value = str(experts)
        if flag in BOOL_FLAGS:
            if value not in ('0','1'):
                raise ValueError('invalid boolean profile flag '+flag)
            if value == '1':
                result.append(flag)
        else:
            result.extend((flag,value))
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            digest.update(block)
    return {'path':str(Path(path).resolve()), 'sha256':digest.hexdigest(), 'bytes':Path(path).stat().st_size}


def json_request(endpoint, body, timeout):
    from urllib.parse import urlsplit
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        connection.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {result}')
        return result
    finally:
        connection.close()


def tools_smoke(endpoint, output, timeout):
    cases = [
        ('get_weather', {'city':'Paris'}, {'city':{'type':'string'}},
         'Call get_weather with city exactly Paris. Do not answer without the tool call.'),
        ('add', {'a':7,'b':5}, {'a':{'type':'integer'},'b':{'type':'integer'}},
         'Call add once with a=7 and b=5. Do not compute the answer yourself.'),
        ('set_config', {'config':{'enabled':True,'labels':['red','blue']}},
         {'config':{'type':'object','properties':{'enabled':{'type':'boolean'},'labels':{'type':'array','items':{'type':'string'}}},'required':['enabled','labels'],'additionalProperties':False}},
         'Call set_config exactly once with config={"enabled":true,"labels":["red","blue"]}.'),
    ]
    results = []
    for name, expected, properties, prompt in cases:
        body = {'messages':[{'role':'user','content':prompt}], 'temperature':0, 'seed':0,
                'max_tokens':512,'tool_choice':'auto','parallel_tool_calls':True,
                'chat_template_kwargs':{'enable_thinking':False},
                'tools':[{'type':'function','function':{'name':name,'description':'Requested test function',
                          'parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}]}
        result = {'case':name,'expected':expected}
        try:
            result['response'] = json_request(endpoint,body,timeout)
            message = result['response']['choices'][0]['message']
            calls = message.get('tool_calls',[])
            if len(calls) != 1:
                result.update({'pass':False, 'error':f'expected one tool call, got {len(calls)}'})
            else:
                function = calls[0]['function']
                arguments = function['arguments']
                arguments = json.loads(arguments) if isinstance(arguments,str) else arguments
                result.update({'pass':function['name']==name and arguments==expected,'arguments':arguments})
        except Exception as exc:
            result.update({'pass':False,'error':f'{type(exc).__name__}: {exc}'})
        results.append(result)
        output.write_text(json.dumps({'pass':all(r['pass'] for r in results),'cases':results},ensure_ascii=False,indent=2)+'\n')
    return all(r['pass'] for r in results)


def low_reference(endpoint, output, timeout, phase="warm"):
    """Same user prompts, ordering, low/temp0/max1200/fps1 as the saved ref client."""
    cases = [
        ('image', V.REFERENCE/'0.png', 'Describe the visible shape, its color and its position. Be concise.',5.238),
        ('video', V.REFERENCE/'shapes-12s.mp4', 'Describe the sequence of changes in this video in chronological order: the shape, color and position in each stage. Be concise.',36.171),
    ]
    results = []
    for name,path,prompt,reference in cases:
        media = V.image(path) if name=='image' else {'type':'video_url','video_url':{'url':'file://'+str(path)}}
        body = {'model':'m31-vision','messages':[{'role':'user','content':[V.text(prompt),media]}],
                'stream':True,'stream_options':{'include_usage':True},'max_tokens':1200,
                'temperature':0,'reasoning_effort':'low','cache_prompt':False,'video_fps':1}
        result = {'case':name, 'reference_ttft_s':reference,'prompt':prompt,
                  'effort':'low','temperature':0,'max_tokens':1200,'fps':1,
                  'encoder_state':('warm: quality probes already encoded these media; no LLM prefix reuse' if phase=='warm' else
                                   'cold media: image is first encoder request; video follows image, before quality probes; no LLM prefix reuse')}
        try:
            result.update(V.request(endpoint,body,timeout))
            result.update({'pass':True,'speedup':reference/result['ttft_s']})
        except Exception as exc:
            result.update({'pass':False,'error':f'{type(exc).__name__}: {exc}'})
        results.append(result)
        output.write_text(json.dumps({'pass':all(r['pass'] for r in results),'cases':results},ensure_ascii=False,indent=2)+'\n')
    return all(r['pass'] for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda','auto'],default='cpu',help='vision encoder backend; LLM uses native CUDA')
    parser.add_argument('--ctx',type=int)
    parser.add_argument('--gpu-min-free-mib',type=int,help='auto encoder CUDA headroom threshold')
    parser.add_argument('--threads',type=int,default=16,help='threads per encoder CPU context')
    parser.add_argument('--cpu-workers',type=int,default=6)
    parser.add_argument('--engine',type=Path,default=ROOT/'build-m31-cuda/m31_serve')
    parser.add_argument('--tokenizer',type=Path,default=Path('/mnt/ai/Xolotl-m31/build-m31-oracle/m31_tokenizer'))
    parser.add_argument('--adapter',type=Path,default=ROOT/'build-vision/m31_vision_adapter')
    parser.add_argument('--model',type=Path,default=Path(V.MODEL))
    parser.add_argument('--mmproj',type=Path,default=Path(V.MMPROJ))
    parser.add_argument('--profile',type=Path,default=ROOT/'tools/m31/profiles/native-gpu.json')
    parser.add_argument('--gpu-experts-gib',type=float)
    parser.add_argument('--port',type=int,default=8191)
    parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--skip-low',action='store_true')
    args = parser.parse_args()
    if os.environ.get('M31_CARD_SESSION')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        parser.error('requires an aictl-owned GPU0 card session')
    context = args.ctx or (8192 if args.backend=='cuda' else 40000)
    if context < 8192:
        parser.error('vision suite needs context >=8192; context shifting is not supported')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    profile = load_profile(args.profile)
    env = os.environ.copy()
    for key in profile['unset_environment']:
        env.pop(key,None)
    env.update(profile['environment'])
    vision = {'adapter':str(args.adapter.resolve()),'model':str(args.model.resolve()),
              'mmproj':str(args.mmproj.resolve()),'cache_dir':str(output/'cache'),
              'backend':args.backend,'threads':args.threads,'cpu_workers':args.cpu_workers,
              'limits':{'video_fps':1.0,'max_video_frames':120,'max_media_bytes':32<<20,
                        'max_total_bytes':128<<20,'allow_http':False,'allow_local':True}}
    if args.gpu_min_free_mib is not None:
        vision['gpu_min_free_mib'] = args.gpu_min_free_mib
    config = output/'vision-config.json'
    config.write_text(json.dumps(vision,indent=2)+'\n')
    argv = [sys.executable,str(ROOT/'tools/m31/serve_m31.py'),'--engine',str(args.engine.resolve()),
            '--tokenizer',str(args.tokenizer.resolve()),'--model',str(args.model.resolve()),
            '--alias','m31-vision','--port',str(args.port),'--vision-config',str(config),
            *standalone_arguments(profile['engine_arguments'],context,args.gpu_experts_gib)]
    receipt = {'format':'m31-vision-http-session-v1','context':context,'context_shifting':False,
               'encoder':vision,'server_argv':argv,'environment':profile['environment'],
               'unset_environment':profile['unset_environment'],'files':[],
               'quality_config':{'reasoning_effort':'low','temperature':0,'max_tokens':1200,'fps':1},
               'reference_comparison_scope':'low prompts match saved ref client; low runs have warm encoder cache; placement/KV differ from ref'}
    for path in (args.engine,args.tokenizer,args.adapter,args.profile,ROOT/'tools/m31/serve_m31.py',ROOT/'tools/m31/serve/vision_http.py',
                 ROOT/'tools/m31/vision_encoder.py',Path(__file__),ROOT/'tools/m31/vision_validation.py'):
        receipt['files'].append(fingerprint(path))
    receipt_path = output/'receipt.json'
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    endpoint = f'http://127.0.0.1:{args.port}'
    # Refuse an occupied port before loading a second model or probing another service.
    with socket.socket() as check_port:
        check_port.bind(('127.0.0.1', args.port))
    proc = None
    telemetry, telemetry_errors = [], []
    telemetry_stop = threading.Event()
    telemetry_start = time.monotonic()
    def sample_memory():
        while not telemetry_stop.is_set():
            try:
                row = subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total,utilization.gpu',
                                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                used,total,util = [float(x.strip()) for x in row.stdout.strip().split(',')]
                telemetry.append({'elapsed_s':time.monotonic()-telemetry_start,'used_mib':used,'total_mib':total,'utilization_pct':util})
            except Exception as exc:
                telemetry_errors.append(f'{type(exc).__name__}: {exc}')
            telemetry_stop.wait(.3)
    monitor = threading.Thread(target=sample_memory,daemon=True,name='vision-gpu0-memory')
    monitor.start()
    # First sample precedes model startup and is the owned-card baseline.
    deadline = time.monotonic()+6
    while not telemetry and not telemetry_errors and time.monotonic()<deadline:
        time.sleep(.02)
    def stopping(signum,_frame):
        raise InterruptedError(f'HTTP session interrupted ({signum})')
    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGINT,stopping)
    try:
        with (output/'server.log').open('x') as log:
            proc = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            started = time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited ({proc.returncode}); see server.log')
                if time.monotonic()-started > args.timeout:
                    raise TimeoutError('HTTP server readiness timeout')
                conn = http.client.HTTPConnection('127.0.0.1',args.port,timeout=2)
                try:
                    conn.request('GET','/health')
                    response = conn.getresponse()
                    if response.status==200:
                        response.read()
                        break
                except OSError:
                    pass
                finally:
                    conn.close()
                time.sleep(.5)
            receipt['startup_s'] = time.monotonic()-started
            V.prepare(output/'fixtures')
            cold = True if args.skip_low else low_reference(endpoint,output/'low-reference-cold.json',args.timeout,phase='cold')
            quality = V.run_http(output/'fixtures/cases.json',endpoint,'m31-vision',output/'quality.json',args.timeout,None)
            tools = tools_smoke(endpoint,output/'tools.json',args.timeout)
            low = True if args.skip_low else low_reference(endpoint,output/'low-reference.json',args.timeout)
            receipt.update({'pass':cold and quality['pass'] and tools and low,'quality':quality,'tools_pass':tools,'low_pass':low,'low_cold_pass':cold})
    except Exception as exc:
        receipt.update({'pass':False,'error':f'{type(exc).__name__}: {exc}'})
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        telemetry_stop.set()
        monitor.join(timeout=6)
        (output/'gpu0-memory.json').write_text(json.dumps({'samples':telemetry,'errors':telemetry_errors},indent=2)+'\n')
        receipt['gpu0_memory'] = {'baseline_mib':telemetry[0]['used_mib'] if telemetry else None,
                                  'peak_mib':max((r['used_mib'] for r in telemetry),default=None),
                                  'sample_interval_s':.3,'samples':len(telemetry),'errors':telemetry_errors,
                                  'path':str(output/'gpu0-memory.json')}
        receipt['server_returncode'] = None if proc is None else proc.returncode
        receipt['encoder_artifacts'] = [str(p) for p in (output/'cache').glob('*/ipc/manifest.json')]
        receipt['encoder_runtime'] = []
        for path in receipt['encoder_artifacts']:
            try:
                metadata = json.loads(Path(path).read_text())
            except (OSError, ValueError) as exc:
                receipt['encoder_runtime'].append({'manifest':path,'error':str(exc)})
                continue
            receipt['encoder_runtime'].append({'manifest':path, **{key:metadata.get(key) for key in
                                               ('backend','encoder_threads','rows','visual_rows','encode_seconds','cached_image_chunks')}})
        receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pass':receipt['pass'],'receipt':str(receipt_path)},ensure_ascii=False),flush=True)
    return 0 if receipt['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
