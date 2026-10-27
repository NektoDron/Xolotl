#!/usr/bin/env python3
"""History-only lookup draft on frozen ordinary traces; never reads om_bench."""
import argparse
import array
import hashlib
import json
from pathlib import Path


def ids(path):
    values=array.array('i');values.frombytes(path.read_bytes())
    return list(values)


def propose(history, max_order, depth, fallback):
    orders=range(max_order,2,-1) if fallback else (max_order,)
    for order in orders:
        if len(history)<=order:continue
        suffix=history[-order:]
        # Start excludes the current suffix and requires a known next token.
        # Latest past occurrence is selected without scoring future trace data.
        for start in range(len(history)-order-1,-1,-1):
            if history[start:start+order]==suffix:
                stop=min(start+order+depth,len(history))
                assert stop<=len(history)
                return history[start+order:stop],order,start
    return [],0,None


def simulate(prompt, generated, order, depth, fallback):
    anchor=0;rounds=proposed=accepted=verified=found=0;matched_context={};seen=0
    while anchor<len(generated)-1:
        history=prompt+generated[:anchor+1] # already sampled/emitted anchor is visible
        remaining=len(generated)-1-anchor
        candidates,n,start=propose(history,order,min(depth,remaining-1),fallback) if remaining>1 else ([],0,None)
        # Proposal is complete before reading any future reference token.
        if candidates:
            assert start+n+len(candidates)<=len(history)
            found+=1;matched_context[str(n)]=matched_context.get(str(n),0)+1
        matched=0
        for candidate in candidates:
            if candidate!=generated[anchor+matched+1]:break
            matched+=1
        # m accepted drafts plus a target bonus; last sampled token stays unfed.
        committed=matched+1
        rounds+=1;proposed+=len(candidates);accepted+=matched;verified+=len(candidates)+1
        anchor+=committed;seen+=committed
    assert anchor==len(generated)-1 and seen==len(generated)-1
    return {'rounds':rounds,'proposed':proposed,'accepted':accepted,'verified_rows':verified,
            'actually_fed_rows':seen,'lookup_rounds':found,'matched_context':matched_context,
            'acceptance':accepted/proposed if proposed else 0,
            'tokens_per_round_with_target_bonus':seen/rounds,
            'verification_expansion':verified/seen}


def aggregate(rows):
    keys=['rounds','proposed','accepted','verified_rows','actually_fed_rows','lookup_rounds']
    result={key:sum(row[key] for row in rows) for key in keys}
    result.update(cases=len(rows),acceptance=result['accepted']/result['proposed'] if result['proposed'] else 0,
        tokens_per_round_with_target_bonus=result['actually_fed_rows']/result['rounds'],
        verification_expansion=result['verified_rows']/result['actually_fed_rows'],
        ideal_speedup_ignoring_lookup_and_verifier_cost=result['actually_fed_rows']/result['rounds'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dspark-summary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text());job=json.loads(next(args.source.glob('*-card-job.result.json')).read_text())
    if job['returncode']!=0 or not manifest['canonical_v8_engine_relink_exact'] or manifest['environment']['M31_GPU_CACHED_DECODE']!='0':
        raise ValueError('source is not passing ordinary exact-v8 CPU-routed decode')
    corpus=[];hashes={}
    for item in manifest['inputs']:
        path=Path(item['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=item['sha256']:raise ValueError('frozen prompt hash mismatch')
        generated=args.source/'native'/(item['case']+'.generated.i32');trace=ids(generated)
        if len(trace)!=128:raise ValueError('trace must contain128 actual greedy tokens')
        hashes[item['case']]={'prompt_sha256':digest,'generated_sha256':hashlib.sha256(generated.read_bytes()).hexdigest()}
        corpus.append((item['case'],item['split'],ids(path),trace))
    configs=[]
    # Both declared families are reported in full; select using train only.
    for fallback in (False,True):
        for order in (3,4,5):
            for depth in (1,3,7):
                cases=[]
                for name,split,prompt,generated in corpus:
                    cases.append({'case':name,'split':split,**simulate(prompt,generated,order,depth,fallback)})
                configs.append({'family':'longest3..N' if fallback else 'fixedN','order':order,'depth':depth,
                    'train':aggregate([row for row in cases if row['split']=='train']),
                    'held':aggregate([row for row in cases if row['split']=='held']),'cases':cases})
    selected=max(configs,key=lambda row:(row['train']['tokens_per_round_with_target_bonus'],-row['train']['verification_expansion']))
    dspark=[]
    for line in args.dspark_summary.read_text().splitlines():
        row=json.loads(line)
        if row.get('mode')=='spec' and row.get('depth') in (1,3,7) and row.get('p_min')==0:
            dspark.append({key:row[key] for key in ('case','depth','tokens','rounds','proposed','accepted','acceptance','tps')})
    for row in dspark:row['tokens_per_round_in_driver']=row['tokens']/row['rounds']
    result={'pass':True,'scope':'offline history-only prompt/ngram proposals on8 frozen ordinary128-token traces; train6/held2; no om_bench, no Engine, no measured throughput',
        'proposal_policy':'latest past occurrence; fixedN or longest n in3..N; visible prompt+generated through already sampled anchor only; proposal never extends beyond visible history',
        'bonus_policy':'each successful verification advances matched_drafts+1 target bonus; first token already sampled; exactly127 fed rows/case',
        'source_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),'inputs':hashes,
        'configs':configs,'selected_on_train_only':{key:selected[key] for key in ('family','order','depth','train','held')},
        'dspark_context_note':'DSpark frozen V5 corpus3324 differs from8 short cases; comparison is contextual, not matched-prompt throughput claim',
        'dspark_reference':dspark,
        'no_future_proposal_data':True,'no_cross_case_training_or_held_tuning':True}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':True,'configurations':len(configs),'selected_on_train_only':result['selected_on_train_only'],'dspark_reference':dspark},indent=2))


if __name__=='__main__':main()
