#!/usr/bin/env python3
"""Ordinary E3 oracle2048 versus unchanged native6656: semantic cross-batch gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from compare_logits import load_case, execution_chunks
from q2_compat_prepare import completed, fp, freeze, write
from q2_cached_gates import cached_trace

ROOT=Path('/mnt/ai/Xolotl-m31'); LOG=Path('/mnt/ai/logs/m31-port')


def need(ok,message):
    if not ok:raise ValueError(message)


def policy(total,tail):
    chunks=[];pos=0;prefix=total-tail
    while pos<total:
        n=1 if tail and pos>=prefix else min(2048,(prefix if tail else total)-pos)
        chunks.append((pos,n));pos+=n
    return chunks


def prepare(a):
    d=a.directory.resolve();d.mkdir(parents=True,exist_ok=False);native_root=a.native_directory.resolve(strict=True);base=json.loads((native_root/'manifest.json').read_text());oracle=fp(a.oracle);receipt=json.loads(a.oracle_receipt.read_text());need(receipt['elf']['m31_oracle']==oracle['sha256'],'ordinary oracle receipt/ELF mismatch')
    freeze(a.oracle_receipt,d/'oracle-build-receipt.json');(d/'comparison').mkdir();(d/'ids').mkdir()
    for name in('q2_crossbatch_e3.py','q2_cached_gates.py','q2_cached_e3_compare.py','q2_compat_prepare.py','compare_logits.py','cache_seed_preflight.py'):
        freeze(ROOT/'tools/m31'/name,d/'comparison'/name)
    cases=[]
    for c in base['cases']:
        completed(Path(c['native_job']));r=d/c['name'];r.mkdir();ids=freeze(Path(c['ids']['path']),d/'ids'/Path(c['ids']['path']).name);native_case=native_root/c['name']/'native'/c['stem'];head=json.loads((native_case/'manifest.json').read_text());need(head['context']==40000 and head['chunk']==6656 and head['tokens']==c['tokens'],'wrong unchanged native case')
        cli=[oracle['path'],'-m',base['model'],'--backend','cuda','--ctx','40000','--chunk','2048','--threads','80','--ids-file',ids['path'],'--rows',c['rows'],'--taps','none','--oracle-op-offload','1','--out',str(r/'oracle')]
        if c['tail']:cli+=['--decode-tail',str(c['tail'])]
        if c['head_tail']:cli+=['--head-tail',str(c['head_tail'])]
        env=['/usr/bin/env',*[v for k in('CUDA_DEVICE_MAX_CONNECTIONS','CUDA_DEVICE_MAX_COPY_CONNECTIONS','M31_ORACLE_CHUNK_PLAN','M31_ORACLE_CAPACITY_EQUIVALENCE','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_MMQ_PREC')for v in('-u',k)],'GGML_CUDA_DISABLE_FUSION=1','M31_ORACLE_GPU_EXPERT_LAYERS=10,11,12','M31_ORACLE_STOCK_HEAD_PRUNING=1']
        job=r/(d.name+'-'+c['name']+'-oracle.json');write(job,dict(preset='m31-xolotl-check',measurement=False,command_timeout_seconds=1800,commands=[env+cli]));item={**c,'ids':ids,'oracle_job':str(job),'native_case':str(native_case),'expected_oracle_chunks':policy(c['tokens'],c['tail'])};write(r/'plan.json',item);cases.append(item)
    pr=Path('/mnt/ai/llama.cpp-m31pr-1026');sources={relative:fp(pr/relative)for relative in('src/models/minimax-m3.cpp','src/llama-kv-cache.cpp','src/llama-kv-cache-msa.cpp')}
    old=LOG/'q2-best-placement-code128-e3-v1/oracle/code_02_128';seal=json.loads((LOG/'q2-final-selected-byte-seal-v3.json').read_text());segments=next(v for v in seal['resident_E3_phase_review']['segments']if v['name']=='resident_e3_oracle')
    write(d/'manifest.json',dict(format='m31-q2-ordinary-e3-crossbatch2048-plan-v1',acceptance_eligible=False,pp_geometry_different=True,comparison_contract='Same full input IDs and selected causal positions, F16KV, E3 resident storage, trained linear RoPE and n1 teacher inputs. Oracle PP<=2048 and stock lastFFN pruning; native max6656/fullFFN unchanged. No matched-shape/internal-KV bitwise claim.',native_manifest=fp(native_root/'manifest.json'),native_profile=base['profile'],native=base['native'],oracle=oracle,oracle_receipt=fp(d/'oracle-build-receipt.json'),cases=cases,causal_source_proof=sources,control_old_gold=str(old),control_old_gold_segments=segments,thresholds=dict(top1=.99,mean_kl=.001,max_kl=.05),additional_phase_criterion='Aggregate rows and actual32n1 tail independently pass; code128 additionally byte-equal old E3 gold before othercase GPU dispatch.',stop_policy='One authorized ordinary2048 oracle variant. OOM/numeric failure reported to root; no native change, smaller chunk, alternate oracle or automatic qualification/quality. Old6656 oracle OOMs preserved.'))
    print(json.dumps({'control_job':cases[0]['oracle_job'],'conditional_followups':[c['oracle_job']for c in cases[1:]]},indent=2))


def score(a):
    d=a.directory.resolve();p=json.loads((d/'manifest.json').read_text());c=next(v for v in p['cases']if v['name']==a.case);r=d/a.case
    if a.case!='code128':
        control=json.loads((d/'code128/compare.json').read_text());need(control['pass']and control['old_E3_control_byte_equal'],'new ordinary control must pass first')
    oracle_result=completed(Path(c['oracle_job']));native_result=completed(Path(c['native_job']));need(fp(Path(p['oracle']['path']))==p['oracle'],'ordinary oracle ELF changed');need(fp(Path(p['native']['path']))==p['native'],'native ELF changed');need(fp(Path(p['native_manifest']['path']))==p['native_manifest'],'unchanged native manifest changed')
    for descriptor in p['causal_source_proof'].values():need(fp(Path(descriptor['path']))==descriptor,'causal model source proof changed')
    oracle=r/'oracle'/c['stem'];native=Path(c['native_case']);om,av=load_case(oracle);nm,bv=load_case(native)
    for key in('model','tokens','vocab','positions','kv'):need(om[key]==nm[key],'semantic identity differs: '+key)
    need((oracle/'tokens.i32').read_bytes()==(native/'tokens.i32').read_bytes()==Path(c['ids']['path']).read_bytes(),'full input token bytes differ')
    need(nm['engine']=='xolotl-m31-correctness'and om['engine']=='llama.cpp-PR31192'and om['oracle_op_offload'],'wrong source engines/placement')
    need(nm['context']==om['context']==40000 and nm['kv']=='f16'and nm['chunk']==6656 and om['chunk']==2048,'wrong requested contexts/dtype/batchcap')
    need(nm['cuda_fusion_disabled']and om['cuda_fusion_disabled']and nm.get('cuda_mmq_precision')==om.get('cuda_mmq_precision')=='auto','wrong math precision/fusion')
    need(not nm.get('m31_decode_fusion_override_requested')and not nm.get('m31_decode_fusion_unfenced_requested'),'native fusion override changed')
    need(nm['head_tail']==om['head_tail']==c['head_tail'],'selected head width differs')
    need(nm.get('full_last_layer_before_crop')is True and om.get('full_last_layer_before_crop')is (not bool(c['head_tail'])),'stock-vs-full lastFFN metadata incorrect')
    need(execution_chunks(om)==[tuple(v)for v in c['expected_oracle_chunks']],'oracle is not ordinary uniform2048 PP plus singleton tail')
    olog=Path(oracle_result['commands'][0]['log']);nlog=Path(native_result['commands'][0]['log']);ot=olog.read_text();nt=nlog.read_text();need(re.findall(r'^llama_context: n_ctx\s*=\s*(\d+)\s*$',ot,re.M)==['40192'],'oracle physical context is not PAD40000');need(re.findall(r'^llama_context: n_batch\s*=\s*(\d+)\s*$',ot,re.M)==['2048']and re.findall(r'^llama_context: n_ubatch\s*=\s*(\d+)\s*$',ot,re.M)==['2048'],'oracle actual PP capacities not2048')
    overrides=re.findall(r'^tensor (blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight) .*?buffer type overridden to (\S+)',ot,re.M);need(len(overrides)==len({v[0]for v in overrides})==171 and all(buf==('CUDA0'if int(layer)in(10,11,12)else'CUDA_Host')for _,layer,_,buf in overrides),'not exactly E3 resident expert storage')
    trace_path=oracle/'ordinary-trace.jsonl';traces=[json.loads(line)for line in trace_path.read_text().splitlines()];chunks=execution_chunks(om);need(len(traces)==len(chunks),'ordinary causal trace incomplete')
    for (pos,n),t in zip(chunks,traces):
        expected=dict(position=pos,tokens=n,seq_min=0,seq_max=pos+n-1,active_keys_padded_from_verified_positions=((pos+n+255)//256)*256,physical_context_requested=40000,physical_context_actual=40192,rope_ext_factor=0,rope_orig_context=1048576,rope_frequency_scale=1,causal_attention=True)
        need(all(t.get(k)==v for k,v in expected.items()),'ordinary contiguous KV/causal/RoPE trace differs')
    native_base=json.loads(Path(p['native_manifest']['path']).read_text());coverage=cached_trace(native_base,c,Path(c['native_case']).parents[1],nt,nm['actual_chunks'])
    per_rows=[];total_kl=max_kl=max_abs=total_sq=0.;same=0
    for start in range(0,len(om['positions']),8):
        x=np.asarray(av[start:start+8],dtype=np.float64);y=np.asarray(bv[start:start+8],dtype=np.float64);need(np.isfinite(x).all()and np.isfinite(y).all(),'nonfinite logits');same_mask=x.argmax(1)==y.argmax(1);diff=x-y;max_abs=max(max_abs,float(abs(diff).max()));total_sq+=float((diff*diff).sum());same+=int(same_mask.sum());x-=x.max(1,keepdims=True);y-=y.max(1,keepdims=True);x-=np.log(np.exp(x).sum(1,keepdims=True));y-=np.log(np.exp(y).sum(1,keepdims=True));raw_kl=(np.exp(x)*(x-y)).sum(1);need(np.isfinite(raw_kl).all()and float(raw_kl.min())>=-1e-10,'invalid raw KL');kl=np.maximum(raw_kl,0);total_kl+=float(kl.sum());max_kl=max(max_kl,float(kl.max()));per_rows.extend(dict(position=om['positions'][start+i],top1=bool(same_mask[i]),kl=float(k))for i,k in enumerate(kl))
    count=len(per_rows);aggregate=dict(rows=count,top1=same/count,mean_kl=total_kl/count,max_kl=max_kl,max_abs=max_abs,logits_rms=(total_sq/(count*om['vocab']))**.5);aggregate['pass']=aggregate['top1']>=.99 and aggregate['mean_kl']<=.001 and aggregate['max_kl']<=.05;tail=None
    if c['tail']:
        selected=[row for row in per_rows if row['position']>=c['tokens']-32];need(len(selected)==32,'missing32 actual singleton tail outputs');tail=dict(rows=32,top1=sum(row['top1']for row in selected)/32,mean_kl=sum(row['kl']for row in selected)/32,max_kl=max(row['kl']for row in selected));tail['pass']=tail['top1']>=.99 and tail['mean_kl']<=.001 and tail['max_kl']<=.05
    control_equal=None
    if a.case=='code128':
        gold=Path(p['control_old_gold']);raw=(gold/'logits.f32').read_bytes();segment=p['control_old_gold_segments'];cut=segment['PP96_bytes'];need(len(raw)==cut+segment['tail32_bytes']and hashlib.sha256(raw[:cut]).hexdigest()==segment['PP96_sha256']and hashlib.sha256(raw[cut:]).hexdigest()==segment['tail32_sha256'],'oldE3 gold changed');control_equal=raw==(oracle/'logits.f32').read_bytes();need((gold/'tokens.i32').read_bytes()==(oracle/'tokens.i32').read_bytes(),'control full input differs')
    result=dict(format='m31-q2-ordinary-e3-crossbatch2048-result-v1',acceptance_eligible=False,cross_batch_contract=True,pp_geometry_different=(execution_chunks(nm)!=chunks),comparison_contract=p['comparison_contract'],aggregate=aggregate,actual_decode32=tail,old_E3_control_byte_equal=control_equal,pass_=aggregate['pass']and(tail is None or tail['pass'])and(control_equal is not False),native_actual_chunks=execution_chunks(nm),oracle_actual_chunks=chunks,native_lastFFN_full=nm['full_last_layer_before_crop'],oracle_lastFFN_full=om['full_last_layer_before_crop'],causal_runtime_trace=fp(trace_path),native_coverage=coverage,native_logits=fp(native/'logits.f32'),oracle_logits=fp(oracle/'logits.f32'),native_header=fp(native/'manifest.json'),oracle_header=fp(oracle/'manifest.json'),oracle_log=fp(olog),limits='Semantic comparison across different PP microbatches and last-layer pruning; no internalKV/identicalGEMM/bitwise matched-shape claim. Same selected causal outputs and full inputs; numerical/user thresholds unchanged. No quality inherited.')
    result['pass']=result.pop('pass_');write(r/'compare.json',result);print(json.dumps(result,indent=2));return 0 if result['pass']else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);q=sub.add_parser('prepare');q.add_argument('--directory',required=True,type=Path);q.add_argument('--oracle',required=True,type=Path);q.add_argument('--oracle-receipt',required=True,type=Path);q.add_argument('--native-directory',type=Path,default=LOG/'q2-cached-e3-6656-gates-v1');q=sub.add_parser('score');q.add_argument('--directory',required=True,type=Path);q.add_argument('--case',choices=('code128','code3000','long16384','long32768'),required=True);a=p.parse_args();return globals()[a.action](a)


if __name__=='__main__':raise SystemExit(main())
