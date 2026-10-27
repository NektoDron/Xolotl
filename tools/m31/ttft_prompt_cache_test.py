#!/usr/bin/env python3
"""CPU-only checkpoint regression. Coordinator runs under shared build lock/systemd."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,default=root/'build-m31-cuda')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'tools/m31/tests'))
from make_fixture import make_fixture
fixture=a.output/'tiny.gguf';make_fixture(fixture)
obj=a.output/'test.o';binary=a.output/'test'
subprocess.run(['/usr/bin/g++-14','-std=c++20','-O2','-I'+str(root/'tools/m31'),'-I'+str(root/'include'),'-I'+str(root/'build-vision-vendor/ggml/include'),'-c',str(root/'tools/m31/ttft_prompt_cache_test.cpp'),'-o',str(obj)],check=True)
link=shlex.split((a.build/'CMakeFiles/m31_generate.dir/link.txt').read_text())
for i,arg in enumerate(link):
    if arg.endswith('m31_generate.cpp.o'):link[i]=str(obj)
    if arg=='-o':link[i+1]=str(binary)
subprocess.run(link,cwd=a.build,check=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
subprocess.run([str(binary),str(fixture)],env=env,check=True)

# Native JSONL regression: an explicit HTTP/JSON optout must remain a fresh
# computation even if an identical completed prompt is already in the cache.
request={"tokens":[5,7,11,17],"max_tokens":3,"temperature":0,"top_p":1,"top_k":0,
         "min_p":0,"seed":0,"ignore_eos":True,"interactive":False,"ttft_trace":True}
requests=[{**request,"cache_prompt":value} for value in (True,True,False,True)]
def native_requests(chunk, requests):
    command=[str(a.build/'m31_serve'),'-m',str(fixture),'--backend','cpu','--ctx','32','--chunk',str(chunk),'--threads','2']
    result=subprocess.run(command,input=''.join(json.dumps(req)+'\n' for req in requests),
                          capture_output=True,text=True,timeout=60,
                          env={**env,'M31_PROMPT_CACHE_MIB':'1'})
    if result.returncode:
        raise RuntimeError(result.stderr)
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events.pop(0)['event']=='ready'
    groups=[]; current=[]
    for event in events:
        assert event['event']!='error', event
        current.append(event)
        if event['event']=='done':groups.append(current);current=[]
    assert len(groups)==len(requests) and not current
    return groups
normal=native_requests(32,requests)
first_tokens=[e['token'] for e in normal[0] if e['event']=='token']
for index,group in enumerate(normal):
    prefill=next(e for e in group if e['event']=='prefill')
    assert [e['token'] for e in group if e['event']=='token']==first_tokens
    expected_hit=index in (1,3)
    assert prefill['prompt_cache']['hit'] is expected_hit,prefill
    assert bool(prefill['actual_chunks']) is (not expected_hit),prefill
    if expected_hit:
        assert prefill['cached_prefill_recipe']==[{"position":0,"rows":4}]
    if index==2:
        assert prefill['prompt_cache']['enabled'] is False
        assert prefill['prompt_cache']['skip']=='cache_prompt:false'
        assert 'key_ms' not in prefill['prompt_cache']
for group in native_requests(2,requests[:2]):
    prefill=next(e for e in group if e['event']=='prefill')
    assert prefill['prompt_cache']['hit'] is False
    assert len(prefill['actual_chunks'])==2
    assert prefill['prompt_cache']['entries']==0
print('PASS: native exact-repeat cache hit, explicit false bypass, identical greedy continuation, multichunk bypass')
