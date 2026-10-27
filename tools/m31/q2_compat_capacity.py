#!/usr/bin/env python3
"""Explicit selected-long physical KV capacity diagnostic; no GPU launch."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


def module(path):
    spec=importlib.util.spec_from_file_location('q2helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','compare'))
    p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--case',required=True,choices=('long16384','long32768'))
    p.add_argument('--authorization',default='')
    a=p.parse_args();r=a.directory.resolve();h=module(r/'q2_compat_prepare.py')
    plan=json.loads((r/'manifest.json').read_text());case=next(c for c in plan['cases'] if c['name']==a.case);d=r/a.case
    for role in ('native','oracle'):
        if h.fp(Path(plan[role]['path']))!=plan[role]:
            raise ValueError(role+' frozen ELF changed')
    replay=json.loads((d/'replay-receipt.json').read_text());original=Path(replay['job']);originaljob=json.loads(original.read_text())
    native_result=h.completed(Path(case['native_job']))
    native_log=Path(native_result['commands'][0]['log'])
    chunks=json.loads((d/'native'/case['stem']/'manifest.json').read_text())['actual_chunks']
    job=d/(r.name+'-'+a.case+'-oracle-capacity-v1.json')
    descriptor=d/'capacity-equivalence-v1.json'
    if a.action=='prepare':
        if not a.authorization:
            raise ValueError('explicit coordinator authorization text required')
        failed=json.loads(original.with_suffix('.result.json').read_text())
        if failed.get('returncode')==0:
            raise ValueError('original oracle succeeded; no capacity fallback justified')
        selected=[x for x in failed['commands'] if x['argv']==originaljob['commands'][0] and x['returncode']!=0]
        if len(selected)!=1:
            raise ValueError('failed original command provenance absent')
        errorlog=Path(selected[0]['log']).read_text(errors='replace')
        if not re.search(r'out of memory|cannot initialize PR context|failed to allocate|unable to allocate',errorlog,re.I):
            raise ValueError('failure is not recognized capacity allocation failure')
        command=originaljob['commands'][0].copy();command[command.index('--ctx')+1]=str(case['tokens']);command[command.index('--out')+1]=str(d/'oracle-capacity-v1')
        key='M31_ORACLE_CAPACITY_EQUIVALENCE'
        i=command.index(key)
        if command[i-1]!='-u':raise ValueError('expected explicit initial capacity unset')
        command[i-1:i+1]=[key+'=1']
        h.write(job,dict(preset='m31-xolotl-check',measurement=False,command_timeout_seconds=1800,commands=[command]))
        h.write(descriptor,dict(authorization=a.authorization,original_failed_result=h.fp(original.with_suffix('.result.json')),original_failure_log=h.fp(Path(selected[0]['log'])),job=str(job),native_context_requested=case['context'],oracle_context_requested=case['tokens'],oracle_context_expected_actual=((case['tokens']+255)//256)*256,scope='Selected diagnostic physical KV reduction only; actual chunks, contiguous KV, padded active keys and trained linear RoPE checked after completion.'))
        print(job);return
    desc=json.loads(descriptor.read_text());h.completed(job)
    from compare_logits import compare_case,execution_chunks
    native=d/'native'/case['stem'];oracle=d/'oracle-capacity-v1'/case['stem']
    nm=json.loads((native/'manifest.json').read_text());om=json.loads((oracle/'manifest.json').read_text())
    if om['context']!=desc['oracle_context_requested'] or nm['context']!=desc['native_context_requested']:
        raise ValueError('unexpected physical context')
    if om['model']!=plan['model'] or nm['model']!=plan['model'] or om['engine']!='llama.cpp-PR31192' or not om['oracle_op_offload']:
        raise ValueError('wrong model/engine/placement')
    if not all(x['cuda_fusion_disabled'] for x in (nm,om)) or nm.get('m31_decode_fusion_override_requested') or nm.get('m31_decode_fusion_unfenced_requested'):
        raise ValueError('fusion policies differ')
    if execution_chunks(nm)!=execution_chunks(om):raise ValueError('actual chunks differ')
    traces=[json.loads(line) for line in (oracle/'capacity-equivalence.jsonl').read_text().splitlines()]
    if len(traces)!=len(chunks):raise ValueError('missing capacity runtime traces')
    requested_trace=[tuple(map(int,m)) for m in re.findall(r'M31 prefill-plan pos=(\d+) requested=\d+ planned=(\d+) keys=(\d+)',native_log.read_text())]
    if [(c['position'],c['tokens']) for c in chunks if c['tokens']>1]!=[(x[0],x[1]) for x in requested_trace]:
        raise ValueError('native PP planner trace/chunks mismatch')
    flat=json.loads(Path(plan['native_receipt']['path']).read_text())
    build_root=Path(flat['canonical_build_receipt']['path']).parent
    native_source=build_root/'source-tree/tools/m31/m31_engine.cpp'
    source_snapshot=json.loads((build_root/'source-before.json').read_text())
    source_text=native_source.read_text()
    if h.fp(native_source)['sha256']!=source_snapshot['tools/m31/m31_engine.cpp']['sha256'] or 'int64_t kv_length(int64_t n, const Geometry & g) { return padded(n, std::max<int64_t>(256, g.block)); }' not in source_text:
        raise ValueError('compiled native KV padding source invariant absent')
    observed_pp={(x[0],x[1]):x[2] for x in requested_trace}
    for c,t in zip(chunks,traces):
        end=c['position']+c['tokens'];nk=((end+255)//256)*256
        expected=dict(position=c['position'],tokens=c['tokens'],seq_min=0,seq_max=end-1,active_keys_padded_from_verified_positions=nk,physical_context_requested=desc['oracle_context_requested'],physical_context_actual=desc['oracle_context_expected_actual'],rope_ext_factor=0,rope_orig_context=1048576,rope_frequency_scale=1)
        if any(t.get(k)!=v for k,v in expected.items()) or (c['tokens']>1 and observed_pp[(c['position'],c['tokens'])]!=nk):
            raise ValueError('physical capacity/contiguous KV/padded active keys/trained RoPE trace mismatch')
    result=compare_case(oracle,native)
    result.update(acceptance_eligible=False,capacity_equivalence=desc,capacity_trace=h.fp(oracle/'capacity-equivalence.jsonl'),native_planner_trace=h.fp(native_log),native_kv_padding_source=h.fp(native_source),native_nk_observation_policy='PP n>1 nk observed in runtime planner log. Singleton nk derives from compiled kv_length(P+1,pad256) invariant and contiguous actual chunk positions; singleton prefill_chunk fast return emits no planner line.',thresholds=dict(top1=.99,mean_kl=.001,max_kl=.05))
    result['pass']=result['top1']>=.99 and result['mean_kl']<=.001 and result['max_kl']<=.05
    h.write(d/'compare-capacity-v1.json',result);print(json.dumps(result,indent=2));return 0 if result['pass'] else 1


if __name__=='__main__':
    args=sys.argv
    if '--directory' in args:sys.path.insert(0,str(Path(args[args.index('--directory')+1]).resolve()))
    raise SystemExit(main())
