#!/usr/bin/env python3
"""Rank actual autoregressive n1 routes. Train/held cases must be disjoint."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct


def load(path):
    header = None; cases = {}; routes = defaultdict(dict); generated = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'header':
            if header is not None or row.get('format') != 'm31-actual-decode-routing-v1':
                raise ValueError('invalid/duplicate actual decode header')
            header = row
        elif row['kind'] == 'case':
            if row['case'] in cases: raise ValueError('duplicate case')
            cases[row['case']] = row
        elif row['kind'] == 'generated':
            if row['case'] in generated: raise ValueError('duplicate generated row')
            generated[row['case']] = row
        elif row['kind'] == 'routes':
            if header is None or row['phase'] != 'decode' or row['n'] != 1:
                raise ValueError('not actual n1 decode')
            name = row['case']; key = row['generated_index'], row['layer']
            if name not in cases or key in routes[name]: raise ValueError('duplicate/unknown case route')
            ids, weights = row['ids'], row['weights']
            if not header['dense_lead'] <= row['layer'] < header['layers'] or len(ids) != header['used'] or len(weights) != len(ids):
                raise ValueError('route dimensions')
            if len(set(ids)) != len(ids) or any(type(e) is not int or not 0 <= e < header['experts'] for e in ids): raise ValueError('route IDs')
            if any(not math.isfinite(w) or w < 0 for w in weights): raise ValueError('route weights')
            routes[name][key] = row
        else: raise ValueError('unknown trace row')
    if header is None or set(cases) != set(generated) or set(cases) != set(routes): raise ValueError('incomplete cases')
    for name, row in generated.items():
        forwards = len(row['ids']) - 1
        expected = {(g,l) for g in range(forwards) for l in range(header['dense_lead'], header['layers'])}
        if row['actual_decode_forwards'] != forwards or set(routes[name]) != expected: raise ValueError('incomplete n1 route coverage')
        if any(x['position'] != cases[name]['prompt_tokens'] + x['generated_index'] for x in routes[name].values()): raise ValueError('decode positions')
    return header, cases, routes


def analysis(header, routes, names, selected):
    stats = Counter(); per_layer = defaultdict(Counter)
    for name in names:
        for (_, layer), row in sorted(routes[name].items()):
            mask = [(layer,e) in selected for e in row['ids']]
            hits = sum(mask); weights = row['weights']
            for s in (stats, per_layer[layer]):
                s['layers'] += 1; s['routes'] += len(mask); s['resident_routes'] += hits
                s['layers_any_hit'] += hits > 0; s['layers_all_hit'] += hits == len(mask)
                s['weight'] += sum(weights); s['resident_weight'] += sum(w for w,hit in zip(weights,mask) if hit)
    def finish(s):
        return dict(s, hit_rate=s['resident_routes']/max(1,s['routes']), full_layer_hit_rate=s['layers_all_hit']/max(1,s['layers']), any_layer_hit_rate=s['layers_any_hit']/max(1,s['layers']), weighted_hit_rate=s['resident_weight']/max(1e-30,s['weight']))
    return dict(overall=finish(stats), per_layer={str(l):finish(s) for l,s in sorted(per_layer.items())})


def union_windows(header, routes, names, expert_bytes, window):
    """Nonoverlapping complete windows within each generated sequence/layer."""
    total=Counter();per_layer=defaultdict(Counter)
    for name in names:
        n=1+max(index for index,_ in routes[name])
        for first in range(0,n-window+1,window):
            for layer in range(header['dense_lead'],header['layers']):
                groups=[set(routes[name][first+i,layer]['ids']) for i in range(window)]
                unique=len(set.union(*groups));base=sum(map(len,groups));raw=expert_bytes[layer]
                for stats in (total,per_layer[layer]):
                    stats['layer_windows']+=1;stats['baseline_expert_reads']+=base;stats['union_expert_reads']+=unique
                    stats['baseline_weight_bytes']+=base*raw;stats['union_weight_bytes']+=unique*raw
    def finish(s):
        return dict(s,union_to_baseline_bytes=s['union_weight_bytes']/max(1,s['baseline_weight_bytes']),
                    theoretical_weight_read_speedup=s['baseline_weight_bytes']/max(1,s['union_weight_bytes']),
                    mean_unique_experts=s['union_expert_reads']/max(1,s['layer_windows']))
    return dict(window=window,stride=window,complete_nonoverlapping_only=True,overall=finish(total),
                per_layer={str(l):finish(stats) for l,stats in sorted(per_layer.items())})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--trace',required=True);p.add_argument('--identity-rank',required=True)
    p.add_argument('--train',required=True,help='comma-separated actual decode case stems');p.add_argument('--held',required=True)
    p.add_argument('--slots',type=int,nargs='+',default=[162,290,400]);p.add_argument('--expert-bytes',default='/mnt/ai/logs/m31-port/cache-vram-budget-v1.json');p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out)
    if out.exists(): raise ValueError('fresh output directory required')
    h,cases,routes=load(a.trace);train=a.train.split(',');held=a.held.split(',')
    if not train or not held or set(train)&set(held) or set(train)|set(held) != set(cases): raise ValueError('explicit disjoint complete train/held split required')
    identity_rank=json.loads(Path(a.identity_rank).read_text());identity=identity_rank['identity'];identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if identity_sha != identity_rank['identity_sha256']: raise ValueError('identity hash')
    geometry=identity['geometry']
    for key in ('layers','dense_lead','experts','used'):
        if geometry[key] != h[key]: raise ValueError('trace geometry differs from model identity')
    if Path(geometry['model']).resolve() != Path(h['model']).resolve(): raise ValueError('trace model differs from model identity')
    # A model path is not an identity check. The existing preflight tool must still
    # check new rank against current GGUF headers/stats before native admission.
    counts=Counter();weights=Counter()
    for name in train:
        for (_,layer),row in routes[name].items():
            for e,w in zip(row['ids'],row['weights']):counts[layer,e]+=1;weights[layer,e]+=w
    order=sorted(((l,e) for l in range(h['dense_lead'],h['layers']) for e in range(h['experts'])),key=lambda k:(-counts[k],-weights[k],k))
    records=[dict(layer=l,expert=e,count=counts[l,e],weight_sum=weights[l,e]) for l,e in order]
    rank=dict(format='m31-expert-ranking-v1',ranking_only=True,identity=identity,identity_sha256=identity_sha,records=records,training_route_count=sum(counts.values()),training_cases=train,held_cases=held,source_trace=str(Path(a.trace).resolve()),source_trace_sha256=hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),calibration='actual autoregressive greedy ignore_eos; prompt routes excluded')
    replay=[]
    for slots in a.slots:
        if not 0 <= slots <= len(order):raise ValueError('invalid slot count')
        selected=set(order[:slots]);replay.append(dict(slots=slots,train=analysis(h,routes,train,selected),held=analysis(h,routes,held,selected)))
    out.mkdir();(out/'rank.json').write_text(json.dumps(rank,indent=2)+'\n')
    binary=b'M31RANK\0'+struct.pack('<4I',1,h['layers'],h['experts'],len(order))+bytes.fromhex(identity_sha)+struct.pack('<'+'I'*len(order),*(l*h['experts']+e for l,e in order))
    (out/'rank.bin').write_bytes(binary)
    byte_file=Path(a.expert_bytes);byte_doc=json.loads(byte_file.read_text());expert_bytes={row['layer']:row['raw_expert_bytes'] for row in byte_doc['layers']}
    if set(expert_bytes) != set(range(h['dense_lead'],h['layers'])): raise ValueError('expert byte geometry differs')
    unions=[union_windows(h,routes,sorted(cases),expert_bytes,k) for k in [2,4,8]]
    union_report=dict(format='m31-actual-decode-union-windows-v1',trace_sha256=rank['source_trace_sha256'],
        expert_bytes_source=str(byte_file.resolve()),expert_bytes_source_sha256=hashlib.sha256(byte_file.read_bytes()).hexdigest(),
        cases=sorted(cases),rows=unions,limits=['Actual generated n1 routes only; prompt routes excluded.','Complete disjoint windows within each case; tail shorter than K excluded.','Unique original GGUF gate/up/down bytes per expert; no fixed-slot padding.','Weight read traffic potential only, not measured speedup or verified speculative decoding acceptance.'])
    (out/'union-windows.json').write_text(json.dumps(union_report,indent=2)+'\n')
    report=dict(format='m31-actual-decode-hotset-analysis-v1',trace=rank['source_trace'],trace_sha256=rank['source_trace_sha256'],train=train,held=held,replay=replay,union_windows=[dict(window=x['window'],**x['overall']) for x in unions],limitations=['Only observed generated sequences; no production hit-rate guarantee.','Static residency replay; no cost or speed claims.','Run cache_seed_preflight.py on the new rank before loading weights.'])
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
