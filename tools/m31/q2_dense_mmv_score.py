#!/usr/bin/env python3
"""Explicit scorer for Q2 actual-weight projection marker proof, never just ELF rc0."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
x=json.loads(a.input.read_text());cases=x['cases'];marked=[v for v in cases if v['marked_mmv']]
sig=lambda v:(v['role'],tuple(v['parts']),v['type'],v['input'],v['output'])
signatures={sig(v) for v in cases}
checks={
 'case_count_consistent':len(cases)==len(signatures)*28,
 'signature_count_consistent':len(signatures)==x['signature_count'],
 'all_five_types':{v['type'] for v in marked}=={'f32','q4_K','q5_K','q6_K','q8_0'},
 'marked_contract_K1to4':all(v['columns'] in (1,2,3,4) and not v['forced_mmv'] for v in marked),
 'marked_bitwise_zero':all(v['different']==0 and v['max_abs']==0 and v['rms']==0 for v in marked),
 'marked_values_shape':all(v['values']==v['output']*v['columns'] for v in marked),
 'all_finite_metrics':all(all(math.isfinite(v[k]) for k in ('max_abs','rms','scalar_mmv_us','batch_mmv_us','stock_graph_us','marked_graph_us')) for v in cases),
 'all_signatures_cover_K1to4_two_families':all({(v['columns'],v['activation_family']) for v in marked if sig(v)==s}=={(k,f) for k in (1,2,3,4) for f in (0,1)} for s in signatures),
 'K8_controls_not_marked':all(not v['marked_mmv'] for v in cases if v['columns']==8),
 'msa_score_isolation':x['unmarked_msa_score']=={'different':0,'recaptured_after_mark':True,'values':4096},
}
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':all(checks.values()),'checks':checks,'scope':'Independent explicit marked K1..4 actual-weight component scorer; not model gate, DSpark model proof, quality or clean speed acceptance','input_path':str(a.input),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'signatures':len(signatures),'total_cases':len(cases),'marked_cases':len(marked),'marked_values':sum(v['values'] for v in marked),'marked_different':sum(v['different'] for v in marked),'K8_control_different_cases':sum(v['columns']==8 and v['different']>0 for v in cases),'types':{t:{'cases':sum(v['type']==t for v in marked),'values':sum(v['values'] for v in marked if v['type']==t)} for t in sorted({v['type'] for v in marked})}}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
