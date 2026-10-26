#!/usr/bin/env python3
"""Validate separate-process CUDA route proof reports; never runs inference."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    reports=[];seen=set();compared_values=0
    types=('iq3_xxs','iq4_xs','nvfp4')
    expected={(mode,kind,n,broadcast) for mode,ns in [('default',(1,7,128,2048)),('disabled',(128,)),('forced',(7,))]
              for kind in types for n in ns for broadcast in (False,True)}
    for mode,count in [('default',24),('disabled',6),('forced',6)]:
        path=Path(str(a.prefix)+'-'+mode+'.json')
        raw=path.read_bytes();report=json.loads(raw)
        if not report.get('pass') or report.get('dispatch_mode')!=mode or not report.get('dispatch_configured_before_cpu_init'):
            raise ValueError('invalid/failing dispatch report: '+str(path))
        if len(report['cases'])!=count or report.get('case_count')!=count:
            raise ValueError('incomplete case count: '+str(path))
        for row in report['cases']:
            key=(row['dispatch_mode'],row['type'],row['n'],row['broadcast'])
            if key in seen or key not in expected or row['used']!=4 or not row['pass']:
                raise ValueError('duplicated/unexpected/failing case: '+str(key))
            seen.add(key)
            metrics=[row[name] for name in ('baseline_vs_cpu','prototype_token_order_vs_baseline','prototype_expert_order_vs_baseline')]
            if any(not metric['bitwise'] or metric['max_abs']!=0 or metric['relative_l2']!=0 for metric in metrics):
                raise ValueError('not bit-identical: '+str(key))
            if len({metric['values'] for metric in metrics})!=1:
                raise ValueError('inconsistent metric shape: '+str(key))
            compared_values+=metrics[0]['values']
        reports.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'dispatch_mode':mode,'cases':count})
    if seen!=expected:raise ValueError('incomplete route proof')
    result={'format':'m31-route-order-summary-v2','pass':True,'cases':len(seen),
            'f32_values_per_comparison':compared_values,'all_three_comparisons_bitwise':True,
            'separate_process_dispatch':True,'reports':reports,
            'limits':'Real matrix-prefix replay, no model inference or measured speedup.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
