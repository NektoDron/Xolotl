#!/usr/bin/env python3
"""Prepare/replay/score selected cached-E3 gates with fresh resident-E3 oracle."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from compare_logits import compare_case, execution_chunks
from q2_compat_prepare import completed, fp, freeze, write
from q2_cached_e3_compare import validate_seed, GEN_SHA, STRIDE, SLOTS, HOT, SEED_BYTES, fields

ROOT=Path('/mnt/ai/Xolotl-m31'); LOG=Path('/mnt/ai/logs/m31-port')
MODEL='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf'
GOLD_SHA='0e634ebe108c8e35d7d4e69792ea3e27c6e156a4f87a3b01f4ed634378523d72'


def need(ok,message):
    if not ok: raise ValueError(message)


def prepare(a):
    d=a.directory.resolve();d.mkdir(parents=True,exist_ok=False)
    profile=json.loads(a.profile.read_text()); env=profile['environment']; argv=profile['engine_arguments']; opts=dict(zip(argv[::2],argv[1::2]))
    need(opts['--ctx']=='40000' and opts['--chunk']=='6656' and float(opts['--arena-gib'])==140 and float(opts['--gpu-experts-gib'])==9.88 and opts['--threads']=='80','requires final6656 profile')
    need(env['M31_GPU_CACHED_DECODE']==env['M31_GPU_CACHED_DECODE_NATIVE']=='1' and env['M31_GPU_CACHED_DECODE_EXACT']=='0' and env['M31_GPU_DECODE_SEED_HOT']=='384' and env['M31_GPU_SEED_HOT']=='0','wrong cached seed policy')
    need(env['GGML_CUDA_DISABLE_FUSION']=='1' and all(env[k]=='0' for k in ('M31_GPU_DECODE_FUSION','M31_GPU_RMS_FUSION','M31_DECODE_FUSION_UNFENCED','M31_GPU_EXACT','M31_GPU_FAST_FFN','M31_GPU_CPU_QUANTS')),'wrong math/fusion policy')
    seed=Path(env['M31_GPU_SEED_PATH']).parent; pre=validate_seed(seed)
    need(env['M31_GPU_SEED_IDENTITY']==pre['identity_sha256'] and env['M31_GPU_SEED_SHA256']==pre['rank_binary_sha256'] and fp(Path(env['M31_GPU_SEED_RECEIPT']))['sha256']==env['M31_GPU_SEED_RECEIPT_SHA256'],'seed environment identity mismatch')
    native=fp(a.engine);oracle=fp(a.oracle);need(native['sha256']==GEN_SHA and oracle['sha256']==GOLD_SHA,'wrong immutable ELF')
    freeze(a.profile,d/'runtime-profile.json'); (d/'ids').mkdir();(d/'comparison').mkdir()
    for name in ('q2_cached_gates.py','q2_cached_e3_compare.py','q2_compat_prepare.py','compare_logits.py','cache_seed_preflight.py'):
        freeze(ROOT/'tools/m31'/name,d/'comparison'/name)
    case_specs=[('code128',LOG/'ids-v1/code_02_128.i32',128,32,'all',0),('code3000',LOG/'ids-v1/code_18_3000.i32',3000,0,'last:32',32),('long16384',LOG/'port6r4-final-v8-acceptance-v1/long/ids/corpus_16384.i32',16384,32,'last:32',32),('long32768',LOG/'port6r4-final-v8-acceptance-v1/long/ids/corpus_32768.i32',32768,32,'last:32',32)]
    cases=[];launch=['/usr/bin/env',*[v for k in profile['unset_environment'] for v in ('-u',k)],*[k+'='+v for k,v in sorted(env.items())]]
    for name,source,count,tail,rows,head in case_specs:
        need(source.stat().st_size==count*4,'wrong IDs length');ids=freeze(source,d/'ids'/source.name);c=d/name;c.mkdir()
        common=['-m',MODEL,'--backend','cuda','--ctx','40000','--chunk','6656','--threads','80','--ids-file',ids['path'],'--rows',rows,'--taps','none']
        if tail:common+=['--decode-tail',str(tail)]
        if head:common+=['--head-tail',str(head)]
        native_cli=[native['path'],'-m',MODEL,*argv,'--ids-file',ids['path'],'--rows',rows,'--taps','none','--out',str(c/'native')]
        if tail:native_cli+=['--decode-tail',str(tail)]
        if head:native_cli+=['--head-tail',str(head)]
        job=c/(d.name+'-'+name+'-native.json');write(job,dict(preset='m31-xolotl-check',measurement=False,command_timeout_seconds=1800,commands=[launch+native_cli]))
        item=dict(name=name,stem=source.stem,ids=ids,tokens=count,tail=tail,rows=rows,head_tail=head,native_job=str(job),oracle_common=common);cases.append(item);write(c/'plan.json',item)
    canonical=ROOT/'build-port7-canonical/q2-v1';snapshot=json.loads((canonical/'source-before.json').read_text());proof={}
    for relative in ('tools/m31/m31_engine.cpp','tools/m31/m31_gpu_experts.cpp','tools/m31/gpu_cached_decode_impl.hpp'):
        descriptor=fp(canonical/'source-tree'/relative);need(descriptor['sha256']==snapshot[relative]['sha256'],'source proof changed');proof[relative]=descriptor
    write(d/'manifest.json',dict(format='m31-q2-cached-e3-6656-selected-plan-v1',acceptance_eligible=False,model=MODEL,native=native,oracle=oracle,profile=fp(d/'runtime-profile.json'),seed_files={name:fp(seed/name) for name in ('rank.json','rank.bin','preflight.json','seed.receipt','tensors.json','integrity.json')},source_proof=proof,cases=cases,thresholds=dict(top1=.99,mean_kl=.001,max_kl=.05),phase_criterion='Aggregate selected rows and, when present, actual32singleton tail separately meet thresholds. No retrospective change to baseline criteria.',authorization='New root authorization after bounded shortPASS: four fresh selected6656 gates against resident E3 F16 oracle; fullcontext40000 first, explicit allowed capacity fallback only after actual OOM.',quality='No inherited baseline97/tools17 or bounded32768 acceptance.'))
    print(json.dumps([c['native_job']for c in cases],indent=2))


def case(a):
    d=a.directory.resolve();p=json.loads((d/'manifest.json').read_text());c=next(v for v in p['cases']if v['name']==a.case);return d,p,c,d/a.case


def replay(a):
    d,p,c,r=case(a);completed(Path(c['native_job']));m=json.loads((r/'native'/c['stem']/'manifest.json').read_text());chunks=m['actual_chunks'];prefix=c['tokens']-c['tail']
    cursor=0
    for row in chunks:
        need(row['position']==cursor and row['tokens']>0,'noncontiguous native chunks');cursor+=row['tokens']
        need(row['tokens']==1 if row['position']>=prefix and c['tail'] else row['tokens']>=32,'unexpected prefix singleton/smallchunk; coordinator review required')
    need(cursor==c['tokens'],'native chunks incomplete');tsv=r/'oracle-chunk-plan.tsv'
    with tsv.open('x')as f:
        for row in chunks:f.write(str(row['position'])+'\t'+str(row['tokens'])+'\n')
    job=r/(d.name+'-'+c['name']+'-oracle.json');command=['/usr/bin/env','-u','CUDA_DEVICE_MAX_CONNECTIONS','-u','CUDA_DEVICE_MAX_COPY_CONNECTIONS','-u','M31_ORACLE_CAPACITY_EQUIVALENCE','GGML_CUDA_DISABLE_FUSION=1','M31_ORACLE_GPU_EXPERT_LAYERS=10,11,12','M31_ORACLE_CHUNK_PLAN='+str(tsv),p['oracle']['path'],*c['oracle_common'],'--oracle-op-offload','1','--out',str(r/'oracle')]
    write(job,dict(preset='m31-xolotl-check',measurement=False,command_timeout_seconds=1800,commands=[command]));write(r/'replay-receipt.json',dict(job=str(job),actual_chunks=chunks,chunk_plan=fp(tsv),native_result=fp(Path(c['native_job']).with_suffix('.result.json'))));print(job)


def capacity(a):
    d,p,c,r=case(a);need(c['tokens']in(16384,32768) and a.authorization,'authorized selected long case required');replay_doc=json.loads((r/'replay-receipt.json').read_text());original=Path(replay_doc['job']);failed=json.loads(original.with_suffix('.result.json').read_text());need(failed['returncode']!=0,'original fullcontext must fail first');cmd=json.loads(original.read_text())['commands'][0];entries=[e for e in failed['commands']if e['argv']==cmd and e['returncode']!=0];need(len(entries)==1,'missing actual failed command');log=Path(entries[0]['log']);need(re.search(r'out of memory|cannot initialize PR context|failed to allocate',log.read_text(),re.I),'not a capacity failure')
    cmd=cmd.copy();i=cmd.index('M31_ORACLE_CAPACITY_EQUIVALENCE');need(cmd[i-1]=='-u','capacity flag not initially unset');cmd[i-1:i+1]=['M31_ORACLE_CAPACITY_EQUIVALENCE=1'];cmd[cmd.index('--ctx')+1]=str(c['tokens']);cmd[cmd.index('--out')+1]=str(r/'oracle-capacity-v1');job=r/(d.name+'-'+c['name']+'-oracle-capacity-v1.json')
    write(job,dict(preset='m31-xolotl-check',measurement=False,command_timeout_seconds=1800,commands=[cmd]));write(r/'capacity-equivalence-v1.json',dict(authorization=a.authorization,job=str(job),original_failed_result=fp(original.with_suffix('.result.json')),original_failed_log=fp(log),native_requested_context=40000,oracle_requested_context=c['tokens'],oracle_actual_expected=((c['tokens']+255)//256)*256));print(job)


def cached_trace(p,c,r,text,chunks):
    init=re.findall(r'M31 GPU experts: fixed_slots=(\d+) slot_stride=(\d+) pool_bytes=(\d+) budget=(\d+)',text);need(len(init)==1,'missing native pool receipt');slots,stride,pool,budget=map(int,init[0]);need((slots,stride,pool)==(SLOTS,STRIDE,SLOTS*STRIDE)and budget//stride==SLOTS,'wrong full-pool capacity')
    seeds=re.findall(r'M31 decode-only seed: hot=(\d+) upload_bytes=(\d+) seconds=([0-9.]+)',text);need(len(seeds)==int(c['tail']>0),'unexpected primer count');seed_bytes=0
    if seeds:
        hot,raw=map(int,seeds[0][:2]);need(hot==384 and 0<raw<=SEED_BYTES and raw%(SEED_BYTES//384)==0,'invalid actual seed geometry/bytes');seed_bytes=raw
    hybrid=[fields(line)for line in text.splitlines()if line.startswith('M31 GPU cached hybrid label=forward ')];expert=[fields(line)for line in text.splitlines()if line.startswith('M31 GPU expert profile label=forward ')];need(len(hybrid)==len(expert)==len(chunks),'profile snapshot count differs from actual chunks')
    gpu_pp=n1=0;previous=None
    for row,h,e in zip(chunks,hybrid,expert):
        is_n1=row['tokens']==1
        if is_n1:n1+=1
        else:gpu_pp+=1
        need(h['forwards']==3*n1 and h['route_lookups']==228*n1 and h['resident_routes']==12*n1 and h['seed_primes']==int(n1>0)and h['seed_bytes']==(seed_bytes if n1 else 0),'actual E3 resident coverage counters differ')
        need(e['forwards']==57*gpu_pp+3*n1 and e['cached']==0 and e['fast']==0,'wrong GPU FFN path')
        if is_n1:need(previous is not None and e['bytes']==previous['bytes']+(seed_bytes if n1==1 else 0),'unexpected uploads/admissions during cached-only decode')
        previous=e
    need(n1==c['tail'],'actual singleton count differs from intended tail');need(text.count('M31 fused CPU MoE: enabled=1')==1,'fused CPU path inactive')
    return dict(slots=slots,stride=stride,pool_bytes=pool,seed_upload_bytes=seed_bytes,seed_primes=len(seeds),actual_gpu_PP_chunks=gpu_pp,actual_n1_inputs=n1,last_hybrid=hybrid[-1],residency_policy='Observed full384pool+successful384keyprime and sealed first384 keys exactlyallE3 experts, with compiled cached-only no-admission-on-miss code, imply all4hits on10/11/12 and none elsewhere. Per-layer coverage is source/full-pool inference, not direct per-layer telemetry; not applicable to PP-only case.')


def score(a):
    d,p,c,r=case(a);native_result=completed(Path(c['native_job']));replay_doc=json.loads((r/'replay-receipt.json').read_text());cap=json.loads((r/'capacity-equivalence-v1.json').read_text())if a.capacity else None;oracle_job=Path(cap['job']if cap else replay_doc['job']);oracle_result=completed(oracle_job);native=r/'native'/c['stem'];oracle=r/('oracle-capacity-v1'if cap else'oracle')/c['stem'];nm=json.loads((native/'manifest.json').read_text());om=json.loads((oracle/'manifest.json').read_text())
    for role in ('native','oracle'):need(fp(Path(p[role]['path']))==p[role],role+' ELF changed')
    for descriptor in p['seed_files'].values():need(fp(Path(descriptor['path']))==descriptor,'seed metadata changed')
    validate_seed(Path(json.loads((d/'runtime-profile.json').read_text())['environment']['M31_GPU_SEED_PATH']).parent)
    need(fp(d/'runtime-profile.json')==p['profile'],'profile changed')
    for descriptor in p['source_proof'].values():need(fp(Path(descriptor['path']))==descriptor,'compiled source proof changed')
    chunks=nm['actual_chunks'];need(execution_chunks(nm)==execution_chunks(om),'actual oracle replay boundaries differ')
    need(nm['context']==40000 and om['context']==(c['tokens']if cap else 40000),'requested physical context differs')
    for m in(nm,om):need(m['kv']=='f16'and m['cuda_fusion_disabled']and m.get('cuda_mmq_precision')=='auto','cache/fusion/MMQ policy mismatch')
    need(not nm.get('m31_decode_fusion_override_requested')and not nm.get('m31_decode_fusion_unfenced_requested')and om['oracle_op_offload'],'unexpected fusion/placement override')
    nlog=Path(native_result['commands'][0]['log']);olog=Path(oracle_result['commands'][0]['log']);nt=nlog.read_text();ot=olog.read_text();overrides=re.findall(r'^tensor (blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight) .*?buffer type overridden to (\S+)',ot,re.M)
    need(len(overrides)==len({v[0]for v in overrides})==171,'missing actual oracle expert placement evidence')
    need(all(buf==('CUDA0'if int(layer)in(10,11,12)else'CUDA_Host')for _,layer,_,buf in overrides),'oracle does not have exactlyresidentE3')
    placement=json.loads((oracle.parent/'oracle-placement.json').read_text());need(placement['gpu_expert_layers']=='10,11,12'and placement['oracle_op_offload']and placement['context_requested']==om['context'],'oracle placement sidecar differs')
    actual_contexts=re.findall(r'^llama_context: n_ctx\s*=\s*(\d+)\s*$',ot,re.M);expected_context=((om['context']+255)//256)*256;need(actual_contexts==[str(expected_context)],'oracle physical allocation unexpected')
    pp_observed=[tuple(map(int,v))for v in re.findall(r'M31 prefill-plan pos=(\d+) requested=\d+ planned=(\d+) keys=(\d+)',nt)];need([(v[0],v[1])for v in pp_observed]==[(v['position'],v['tokens'])for v in chunks if v['tokens']>1],'native PP n/nk observation differs')
    need(all(nk==((pos+n+255)//256)*256 for pos,n,nk in pp_observed),'native observed active nk differs')
    if cap:
        traces=[json.loads(line)for line in(oracle/'capacity-equivalence.jsonl').read_text().splitlines()];need(len(traces)==len(chunks),'missing capacity traces')
        for chunk,t in zip(chunks,traces):
            end=chunk['position']+chunk['tokens'];expected=dict(position=chunk['position'],tokens=chunk['tokens'],seq_min=0,seq_max=end-1,active_keys_padded_from_verified_positions=((end+255)//256)*256,physical_context_requested=c['tokens'],physical_context_actual=expected_context,rope_ext_factor=0,rope_orig_context=1048576,rope_frequency_scale=1)
            need(all(t.get(k)==v for k,v in expected.items()),'contiguous KV/padding/RoPE capacity trace differs')
        source=Path(p['source_proof']['tools/m31/m31_engine.cpp']['path']).read_text();need('int64_t kv_length(int64_t n, const Geometry & g) { return padded(n, std::max<int64_t>(256, g.block)); }'in source,'compiled singleton KV padding invariant missing')
    coverage=cached_trace(p,c,r,nt,chunks);metrics=compare_case(oracle,native);phase=None
    if c['tail']:
        ah=np.memmap(oracle/'logits.f32',dtype='<f4',mode='r',shape=(len(om['positions']),om['vocab']));bh=np.memmap(native/'logits.f32',dtype='<f4',mode='r',shape=ah.shape);selected=[i for i,pos in enumerate(nm['positions'])if pos>=c['tokens']-32];need(len(selected)==32,'missing selected actual32n1 rows');x=np.asarray(ah[selected],dtype=np.float64);y=np.asarray(bh[selected],dtype=np.float64);same=int((x.argmax(1)==y.argmax(1)).sum());maxabs=float(abs(x-y).max());x-=x.max(1,keepdims=True);y-=y.max(1,keepdims=True);x-=np.log(np.exp(x).sum(1,keepdims=True));y-=np.log(np.exp(y).sum(1,keepdims=True));kl=np.maximum((np.exp(x)*(x-y)).sum(1),0);phase=dict(rows=32,top1=same/32,mean_kl=float(kl.mean()),max_kl=float(kl.max()),max_abs=maxabs);phase['pass']=phase['top1']>=.99 and phase['mean_kl']<=.001 and phase['max_kl']<=.05
    aggregate=metrics['top1']>=.99 and metrics['mean_kl']<=.001 and metrics['max_kl']<=.05
    result=dict(format='m31-q2-cached-e3-6656-selected-result-v1',acceptance_eligible=False,aggregate=metrics,aggregate_pass=aggregate,actual_decode32=phase,pass_=aggregate and(phase is None or phase['pass']),coverage=coverage,native_context_requested=40000,oracle_context_requested=om['context'],oracle_context_actual=expected_context,capacity_equivalence=cap,oracle_resident_roles=9,native_log=fp(nlog),oracle_log=fp(olog),native_header=fp(native/'manifest.json'),oracle_header=fp(oracle/'manifest.json'),native_logits=fp(native/'logits.f32'),oracle_logits=fp(oracle/'logits.f32'),limits='Four selected teacher-forced cases; no fullgate20/greedy/quality acceptance. Native singleton nk comes from compiled padding invariant; PPnk and alloraclecapacity traces are observed. Newmaxchunk6656 profile never inherits baseline97/tools17 or earlier shortPASS.')
    result['pass']=result.pop('pass_');write(r/('compare-capacity-v1.json'if cap else'compare.json'),result);print(json.dumps(result,indent=2));return 0 if result['pass']else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);prep=sub.add_parser('prepare');prep.add_argument('--directory',type=Path,required=True);prep.add_argument('--profile',type=Path,default=LOG/'q2-cached-e3-final-profile-v1/runtime-profile.json');prep.add_argument('--engine',type=Path,default=LOG/'q2-cached-e3-code128-v1/m31_generate-frozen');prep.add_argument('--oracle',type=Path,default=LOG/'q2-best-placement-code128-e3-v1/m31_oracle-placement-frozen')
    for action in('replay','capacity','score'):
        q=sub.add_parser(action);q.add_argument('--directory',type=Path,required=True);q.add_argument('--case',choices=('code128','code3000','long16384','long32768'),required=True)
        if action=='capacity':q.add_argument('--authorization',required=True)
        if action=='score':q.add_argument('--capacity',action='store_true')
    a=p.parse_args();return globals()[a.action](a)


if __name__=='__main__':raise SystemExit(main())
