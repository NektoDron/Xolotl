#!/usr/bin/env python3
"""Six actual HTTP requests: representative contexts/depths, not final acceptance."""
import argparse
import array
import hashlib
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();labels=['4K','16K','32K'];sizes=[4096,16384,32768]
    arms={};records=[]
    for depth in (1,3):
        folder=args.directory/f'depth{depth}'
        prompts=[json.loads(line) for line in (folder/'prompt-ids.jsonl').read_text().splitlines()]
        emitted=[json.loads(line) for line in (folder/'native-token-events.jsonl').read_text().splitlines()]
        responses=[json.loads(line) for line in (folder/'speed.tsv.responses.jsonl').read_text().splitlines()]
        responses=[row for row in responses if row['path']=='/v1/chat/completions']
        if len(prompts)!=3 or len(emitted)!=3 or len(responses)!=3:raise ValueError('expected exactly3 requests/depth')
        plans=[];groups=[]
        for line in next(folder.glob('*-http-session.server.log')).read_text().splitlines():
            match=re.search(r'M31 prefill-plan pos=(\d+) requested=(\d+) planned=(\d+) keys=(\d+)',line)
            if match:plans.append(tuple(map(int,match.groups())))
            if line.startswith('M31 request timings '):groups.append(plans);plans=[]
        if plans or len(groups)!=3:raise ValueError('incomplete native PP recipe')
        rows=[]
        for index,(label,size,prompt,native,response,plan) in enumerate(zip(labels,sizes,prompts,emitted,responses,groups)):
            body=response['body'];done=native['native_done'];t=done['timings'];ids=native['tokens'];input_ids=prompt['tokens'];head=native['prefill']
            if body.get('cache_prompt') is not False or body.get('temperature')!=0 or body.get('max_tokens')!=256 or body.get('ignore_eos') is not True:raise ValueError('unexpected sampler/cache contract')
            if not size<=len(input_ids)<size+1024 or t['prompt_n']!=len(input_ids):raise ValueError('unexpected rendered context')
            if native['event']!='done' or done['finish_reason']!='length' or len(ids)!=256 or t['predicted_n']!=256 or t['sampling_calls']!=256:raise ValueError('incomplete greedy samples')
            if native['native_terminal_unemitted_samples']!=0 or native['token_indexes']!=list(range(256)):raise ValueError('emitted ID boundary differs')
            if len(native['acks'])!=256 or not all(native['acks']) or t['interactive_waits']!=256:raise ValueError('ACK/native wait counts differ')
            if done['kv_position']!=len(input_ids)+255 or done['fed_generated_n']!=255:raise ValueError('KV fed boundary differs')
            if t['dspark_mode']!='greedy' or t['dspark_prepare_ms']<=0 or t['predicted_ms']<t['dspark_prepare_ms']:raise ValueError('draft setup hidden/excluded')
            if t['dspark_rounds']+t['dspark_accepted']!=255 or t['dspark_accepted']>t['dspark_proposed']:raise ValueError('bonus/accepted counters inconsistent')
            if head.get('cache_hit') is not False or head.get('cached_n')!=0 or head.get('prompt_eval_n')!=len(input_ids):raise ValueError('cached prefix contaminated exploration')
            actual=[(c['position'],c['tokens'],c['padded_keys']) for c in head['actual_chunks']]
            if actual!=[(p,n,nk) for p,requested,n,nk in plan] or sum(n for p,n,nk in actual)!=len(input_ids):raise ValueError('native actual_chunks differs from PP plan')
            for values,key,digest in [(input_ids,'prompt',prompt['tokens_json_sha256']),(ids,'generated',native['tokens_json_sha256'])]:
                if hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()!=digest:raise ValueError(key+' hash invalid')
                if any(isinstance(value,bool) or not isinstance(value,int) or not 0<=value<native['native_vocab'] for value in values):raise ValueError('invalid ID')
                packed=array.array('i',values);(folder/f'{label}.{key}.i32').write_bytes(packed.tobytes())
            record={'depth':depth,'context_label':label,'prompt_n':len(input_ids),'predicted_n':256,'actually_fed_rows':255,
                'prompt_ids_sha256_json':prompt['tokens_json_sha256'],'generated_ids_sha256_json':native['tokens_json_sha256'],
                'actual_chunks':actual,'pp_ms':t['prompt_ms'],'pp_tps':len(input_ids)*1000/t['prompt_ms'],
                'decode_ms_including_lazy_setup':t['predicted_ms'],'generation_tps':255000/t['predicted_ms'],
                'lazy_prepare_ms':t['dspark_prepare_ms'],'proposed':t['dspark_proposed'],'accepted':t['dspark_accepted'],'rounds':t['dspark_rounds'],
                'acceptance':t['dspark_accepted']/t['dspark_proposed'] if t['dspark_proposed'] else 0,
                'tokens_per_round_with_bonus':255/t['dspark_rounds'],'verified_rows':t['dspark_verified_rows'],
                'verification_expansion':t['dspark_verified_rows']/255,'sampling_calls':256,'ACKs':256,'cache_hit':False,
                'draft_ms':t['dspark_draft_ms'],'verify_ms':t['dspark_verify_ms'],'append_ms':t['dspark_append_ms']}
            rows.append({'record':record,'input':input_ids,'ids':ids,'body':body,'params':prompt['parameters'],'choices':response['response']['choices']});records.append(record)
        arms[depth]=rows
    cross=[]
    for label,a,b in zip(labels,arms[1],arms[3]):
        body_a=a['body'].copy();body_b=b['body'].copy();body_a.pop('model',None);body_b.pop('model',None)
        if a['input']!=b['input'] or a['params']!=b['params'] or body_a!=body_b:raise ValueError('depths received different prompts/samplers')
        if a['record']['actual_chunks']!=b['record']['actual_chunks']:raise ValueError('depths used different PP recipes')
        cross.append({'context':label,'actual_inputs_equal':True,'actual_PP_recipes_equal':True,
                      'native_generated_IDs_equal':a['ids']==b['ids'],'parsed_choices_equal':a['choices']==b['choices']})
    passed=all(row['native_generated_IDs_equal'] and row['parsed_choices_equal'] for row in cross)
    result={'pass':passed,'scope':'six representative actual HTTP greedy requests; exploratory timings/acceptance with lazy setup included, not final om repetitions/GSM/tools acceptance',
        'warmup':'none; exactly6 model requests, per-depth server loaded once; first request includes cold decode captures/setup',
        'records':records,'cross_depth_math_checks':cross}
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
