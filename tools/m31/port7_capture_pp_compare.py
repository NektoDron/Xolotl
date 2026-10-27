#!/usr/bin/env python3
"""Read exact om_bench response traces; report PP change without inventing tolerance."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-native-ids',action='store_true')
    args=parser.parse_args()
    arms={}
    for name in ('ordinary','reuse'):
        path=args.directory/name/'speed.tsv.responses.jsonl'
        allrows=[json.loads(line) for line in path.read_text().splitlines()]
        rows=[row for row in allrows if row['path']=='/v1/chat/completions']
        if len(rows)!=3:raise ValueError(f'{name}: expected warmup+32K+32K')
        arms[name]=rows
    # Accept no speed score until actual rendered IDs and native chunk geometry
    # match. HTTP text alone cannot establish the model input recipe.
    protocol={}
    for name in ('ordinary','reuse'):
        folder=args.directory/name
        ids=[json.loads(line) for line in (folder/'prompt-ids.jsonl').read_text().splitlines()]
        log=next(folder.glob('*-http-session.server.log')).read_text()
        plans=[];groups=[]
        for line in log.splitlines():
            match=re.search(r'M31 prefill-plan pos=(\d+) requested=(\d+) planned=(\d+) keys=(\d+)',line)
            if match:plans.append(tuple(map(int,match.groups())))
            if line.startswith('M31 request timings '):
                timings=json.loads(line.removeprefix('M31 request timings '))
                groups.append({'chunks':plans,'prompt_n':timings['prompt_n']});plans=[]
        if plans or len(ids)!=3 or len(groups)!=3:raise ValueError(name+': incomplete native IDs/chunk trace')
        for actual,group in zip(ids,groups):
            tokens=actual['tokens']
            digest=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
            if digest!=actual['tokens_json_sha256'] or len(tokens)!=group['prompt_n']:
                raise ValueError(name+': input IDs hash/count invalid')
            position=0
            for first,requested,n,nk in group['chunks']:
                if first!=position or n<1 or requested<n or nk!=((position+n+255)//256)*256:
                    raise ValueError(name+': invalid native chunk recipe')
                position+=n
            if position!=len(tokens):raise ValueError(name+': native chunks do not cover prompt')
        protocol[name]={'ids':ids,'groups':groups}
        native_path=folder/'native-token-events.jsonl'
        if args.require_native_ids or native_path.exists():
            events=[json.loads(line) for line in native_path.read_text().splitlines()]
            if len(events)!=3:raise ValueError(name+': expected3 complete native token-event traces')
            for index,event in enumerate(events):
                tokens=event['tokens'];done=event['native_done'];timings=done['timings']
                if event['event']!='done' or done.get('finish_reason')!='length':raise ValueError(name+': unexpected native termination')
                if len(tokens)!=timings['predicted_n'] or event['native_terminal_unemitted_samples']!=0:
                    raise ValueError(name+': ignoreEOS paired benchmark lost emitted token IDs')
                if event['token_indexes']!=list(range(len(tokens))):raise ValueError(name+': token indexes missing/duplicated')
                if len(event['acks'])!=timings['interactive_waits'] or len(event['acks'])!=len(tokens) or not all(event['acks']):
                    raise ValueError(name+': ACK/native interactive boundary mismatch')
                if timings['sampling_calls']!=len(tokens):raise ValueError(name+': extra/missing native sampler draw')
                if timings['prompt_n']!=len(ids[index]['tokens']):raise ValueError(name+': native prompt event differs from actual IDs')
                if any(isinstance(token,bool) or not isinstance(token,int) or not 0<=token<event['native_vocab'] for token in tokens):
                    raise ValueError(name+': native token ID out of range')
                digest=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
                if digest!=event['tokens_json_sha256']:raise ValueError(name+': native token hash invalid')
            protocol[name]['token_events']=events
    for index,(a,b) in enumerate(zip(protocol['ordinary']['ids'],protocol['reuse']['ids'])):
        if a!=b:raise ValueError(f'actual token IDs or native parameters differ at {index}')
    if protocol['ordinary']['groups']!=protocol['reuse']['groups']:
        raise ValueError('actual native chunk/nk recipes differ; PP attribution rejected')
    if args.require_native_ids:
        for index,(a,b) in enumerate(zip(protocol['ordinary']['token_events'],protocol['reuse']['token_events'])):
            if a['tokens']!=b['tokens']:raise ValueError(f'exact native greedy IDs differ at{index}')
    records=[]
    for index,(ordinary,reuse) in enumerate(zip(arms['ordinary'],arms['reuse'])):
        left,right=ordinary['body'].copy(),reuse['body'].copy()
        left.pop('model',None);right.pop('model',None)
        if left!=right:raise ValueError('request prompt/sampler differs')
        if left.get('cache_prompt') is not False or left.get('temperature')!=0:
            raise ValueError('expected cold greedy request')
        a,b=ordinary['response'],reuse['response']
        if a['choices']!=b['choices']:raise ValueError(f'greedy output differs at request {index}')
        ta,tb=a['timings'],b['timings']
        for key in ('prompt_n','predicted_n','sampling_calls'):
            if ta[key]!=tb[key]:raise ValueError(f'{key} differs')
        if ta['sampling_calls']!=ta['predicted_n']:raise ValueError('extra/missing sampler draw')
        if tb.get('dspark_mode')!='greedy':raise ValueError('M3 arm fell back')
        if index and ta['prompt_n']<32000:raise ValueError('32K prompt not reached')
        if min(ta['prompt_ms'],tb['prompt_ms'])<=0:raise ValueError('missing precise PP time')
        records.append({'request':index,'warmup':index==0,'prompt_n':ta['prompt_n'],
            'predicted_n':ta['predicted_n'],'actual_tokens_sha256':protocol['ordinary']['ids'][index]['tokens_json_sha256'],
            'actual_chunks':protocol['ordinary']['groups'][index]['chunks'],'body_sha256':hashlib.sha256(json.dumps(left,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'ordinary_prompt_ms':ta['prompt_ms'],'reuse_prompt_ms':tb['prompt_ms'],
            'ordinary_pp_tps':ta['prompt_n']*1000/ta['prompt_ms'],
            'reuse_pp_tps':tb['prompt_n']*1000/tb['prompt_ms'],
            'choices_equal':True,'dspark_rounds':tb.get('dspark_rounds'),
            'ordinary_generation_tps':ta.get('predicted_per_second'),
            'reuse_generation_tps':tb.get('predicted_per_second')})
    if args.require_native_ids:
        for index,record in enumerate(records):
            event=protocol['ordinary']['token_events'][index]
            record['native_token_ids_equal']=True
            record['native_generated_tokens_sha256']=event['tokens_json_sha256']
            record['native_ack_count']=len(event['acks'])
    measured=records[1:];tokens=sum(r['prompt_n'] for r in measured)
    ordinary=tokens*1000/sum(r['ordinary_prompt_ms'] for r in measured)
    reuse=tokens*1000/sum(r['reuse_prompt_ms'] for r in measured)
    result={'pass':True,'scope':'same-V7 paired om_bench cold32K: actual token IDs/parameters/chunk-nk first; then greedy choices/sampler counts; pass is correctness, PP ratio has no invented tolerance',
        'records':records,'ordinary_weighted_pp_tps':ordinary,'reuse_weighted_pp_tps':reuse,
        'reuse_over_ordinary':reuse/ordinary,'pp_change_percent':100*(reuse/ordinary-1),
        'reuse_target_1000_tps':reuse>=1000,'noise_limit':'two sequential repetitions; no statistical confidence claim'}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
