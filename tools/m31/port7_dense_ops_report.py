#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description='Summarize individual stock-op batching proofs; diagnostic success does not imply all operators are exact.')
    p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.input.read_text());groups=defaultdict(list)
    for row in data['cases']:groups[(row['role'],row['K'])].append(row)
    summary=[]
    for (role,k),rows in sorted(groups.items()):
        outputs=[v for row in rows for v in row['outputs']]
        summary.append({'role':role,'K':k,'cases':len(rows),'exact_cases':sum(bool(v['exact']) for v in rows),
            'production_helper_cases':sum(bool(v.get('production_helper',False)) for v in rows),
            'values':sum(v['values'] for v in outputs),'different':sum(v['different'] for v in outputs),
            'max_abs':max(v['max_abs'] for v in outputs),
            'median_scalar_graph_us':statistics.median(v['scalar_graph_us'] for v in rows),
            'median_batch_graph_us':statistics.median(v['batch_graph_us'] for v in rows)})
    result={'scope':data['scope'],'cases':len(data['cases']),'exact_cases':sum(bool(v['exact']) for v in data['cases']),
            'production_helper_cases':sum(bool(v.get('production_helper',False)) for v in data['cases']),
            'all_exact':all(v['exact'] for v in data['cases']),'groups':summary,
            'failures':[v for v in data['cases'] if not v['exact']]}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))


if __name__=='__main__':main()
