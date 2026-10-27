#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.input.read_text());cases=data['cases'];groups=defaultdict(list)
    for row in cases:groups[(row['nk'],row['K'],row.get('mask_type','f16'))].append(row)
    summarized=[]
    for (nk,k,mask_type),rows in sorted(groups.items()):
        outs=[x for row in rows for x in row['outputs']]
        summarized.append({'nk':nk,'K':k,'mask_type':mask_type,'cases':len(rows),'exact_cases':sum(x['exact'] for x in rows),
            'values':sum(x['values'] for x in outs),'different':sum(x['different'] for x in outs),
            'max_abs':max(x['max_abs'] for x in outs),'median_scalar_graph_us':statistics.median(x['scalar_graph_us'] for x in rows),
            'median_batch_graph_us':statistics.median(x['batch_graph_us'] for x in rows)})
    result={'scope':data['scope'],'cases':len(cases),'exact_cases':sum(x['exact'] for x in cases),'all_exact':all(x['exact'] for x in cases),
            'values':sum(x['values'] for c in cases for x in c['outputs']),'different':sum(x['different'] for c in cases for x in c['outputs']),
            'block_id_values':sum(x['values'] for c in cases for x in c['outputs'] if x['name']=='block_ids'),
            'block_id_different':sum(x['different'] for c in cases for x in c['outputs'] if x['name']=='block_ids'),
            'groups':summarized,'failures':[x for x in cases if not x['exact']]}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
    if not result['all_exact']:raise SystemExit(1)


if __name__=='__main__':main()
