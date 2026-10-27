#!/usr/bin/env python3
"""Online metadata-only replay; no model/weights/GPU and no future-route oracle."""
import collections,datetime,hashlib,json,pathlib,time
R=pathlib.Path('/mnt/ai/logs/m31-port');rankpath=R/'port7-actual-decode-rank-v1/rank.json';budgetpath=R/'cache-vram-budget-v1.json';source=R/'port7-actual-decode-routes-v1/native/routes.jsonl';out=R/'port7-online-cache-sim-v1.json'
start=time.monotonic();rank=json.loads(rankpath.read_text());budget=json.loads(budgetpath.read_text());analysis=json.loads((R/'port7-actual-decode-rank-v1/analysis.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==rank['source_trace_sha256']==analysis['trace_sha256']
train=rank['training_cases'];held=rank['held_cases'];assert train==analysis['train']and held==analysis['held'];assert not set(train)&set(held)
layers={r['layer']:r['raw_expert_bytes']for r in budget['layers']};static=[(r['layer'],r['expert'])for r in rank['records'][:290]];assert len(set(static))==290
cases=collections.defaultdict(lambda:collections.defaultdict(dict))
with source.open()as f:
 for line in f:
  r=json.loads(line)
  if r.get('kind')!='routes'or r.get('phase')!='decode'or r.get('n')!=1:continue
  case=r['case'];step=r['generated_index'];layer=r['layer'];assert layer in layers and layer not in cases[case][step];assert len(r['ids'])==4 and len(set(r['ids']))==4
  cases[case][step][layer]=r
assert set(cases)==set(train+held)
for case,steps in cases.items():
 assert sorted(steps)==list(range(127))
 for rows in steps.values():assert set(rows)==set(layers)

def empty():return {'tokens':0,'layers':0,'routes':0,'hits':0,'all_hit_layers':0,'any_hit_layers':0,'logical_weight_bytes':0,'hit_weight_bytes':0,'h2d_admissions':0,'h2d_raw_weight_bytes':0,'h2d_full_slot_upper_bytes':0,'evictions':0}
def finish(a):
 a=dict(a);a|={'hit_rate':a['hits']/a['routes'],'hit_weight_byte_fraction':a['hit_weight_bytes']/a['logical_weight_bytes'],'full_layer_hit_rate':a['all_hit_layers']/a['layers'],'any_layer_hit_rate':a['any_hit_layers']/a['layers'],'h2d_raw_bytes_per_token':a['h2d_raw_weight_bytes']/a['tokens'],'h2d_GBps_required_at35tps':a['h2d_raw_weight_bytes']/a['tokens']*35/1e9,'h2d_floor_ms_per_token_at50GBps':a['h2d_raw_weight_bytes']/a['tokens']/50e9*1000,'h2d_only_tps_ceiling_at50GBps':50e9*a['tokens']/a['h2d_raw_weight_bytes']if a['h2d_raw_weight_bytes']else None};return a
names=['static290','lru290_cold','lru290_staticseed','last_token228','last_token_plus_static62']
records=[]
for name in names:
 percase=[];pinned=set(static[:62])if name=='last_token_plus_static62'else set();seed=static if name in('static290','lru290_staticseed')else list(pinned)
 seedbytes=sum(layers[l]for l,e in seed)
 for case in train+held:
  resident=collections.OrderedDict((key,None)for key in seed);previous=set();total=empty();steady=empty();current=set()
  for step,rows in sorted(cases[case].items()):
   current=set();stats=empty();stats['tokens']=1
   for layer,r in sorted(rows.items()):
    keys=[(layer,int(e))for e in r['ids']]
    available=set(resident)if name.startswith('lru')or name=='static290'else previous|pinned
    hits=[key in available for key in keys];stats['layers']+=1;stats['routes']+=4;stats['hits']+=sum(hits);stats['all_hit_layers']+=all(hits);stats['any_hit_layers']+=any(hits);stats['logical_weight_bytes']+=4*layers[layer];stats['hit_weight_bytes']+=sum(hits)*layers[layer];current.update(keys)
    # Evaluate all four routes BEFORE admission. Current layer readers stay leased
    # until its end; no current miss can become a same-layer artificial hit.
    if name.startswith('lru'):
     # Touch every leased hit before inserting misses: a current layer hit
     # cannot be evicted and spuriously re-uploaded by an earlier rank's miss.
     for key,hit in zip(keys,hits):
      if hit:resident.move_to_end(key)
     for key,hit in zip(keys,hits):
      if not hit:
       if len(resident)==290:resident.popitem(last=False);stats['evictions']+=1
       resident[key]=None;stats['h2d_admissions']+=1;stats['h2d_raw_weight_bytes']+=layers[layer];stats['h2d_full_slot_upper_bytes']+=budget['slot_stride']
    elif name.startswith('last_token'):
     for key,hit in zip(keys,hits):
      if not hit:stats['h2d_admissions']+=1;stats['h2d_raw_weight_bytes']+=layers[layer];stats['h2d_full_slot_upper_bytes']+=budget['slot_stride']
   previous=current-pinned
   assert len(previous|pinned)<=290
   for key,v in stats.items():total[key]+=v;steady[key]+=v if step>=8 else 0
  percase.append({'case':case,'split':'train'if case in train else'held','all':finish(total),'steady_skip8':finish(steady)})
 splits={}
 for split in('train','held'):
  for metric in('all','steady_skip8'):
   a=empty()
   for c in percase:
    if c['split']==split:
     for key in a:a[key]+=c[metric][key]
   splits[split+'_'+metric]=finish(a)
 record={'policy':name,'resident_slot_budget':290,'initial_seed_count':len(seed),'initial_seed_raw_bytes_per_case':seedbytes,'initial_seed_not_in_runtime_H2D_counters':True,'per_case':percase,'aggregate':splits};records.append(record)
staticheld=records[0]['aggregate']['held_all'];expected=next(r for r in analysis['replay']if r['slots']==290)['held']['overall'];assert staticheld['routes']==expected['routes']and staticheld['hits']==expected['resident_routes']and abs(staticheld['hit_rate']-expected['hit_rate'])<1e-15
result={'format':'m31-online-cache-route-sim-v1','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Metadata-only causal online simulation of existing8x127 n1 greedy forwards (128 samples each). Not model execution, CPU performance, actual H2D overlap or speculative-branch proof.','source':str(source),'source_sha256':sha(source),'rank':str(rankpath),'rank_sha256':sha(rankpath),'expert_bytes_metadata':str(budgetpath),'expert_bytes_metadata_sha256':sha(budgetpath),'train_cases':train,'held_cases':held,'records':records,'static_control_pass':True,'assumptions':['Each case resets residency independently; no training cache state leaks to held cases. Static seed/rank comes only from the fixed6training cases.','LRU admits every miss after current layer consumption and marks it ready immediately; this is an optimistic upper bound ignoring DMA completion delays and PCIe scheduling. No future IDs are used.','last_token228 keeps only previous token routed set (max228keys); plus_static62 pins trained62 plus previous token, <=290slots. Updates after layer consumption have no same-token future-layer benefit because keys include layer.','H2D raw bytes sum original expert gate/up/down payload from header metadata. Full-slot copy upper bound is also recorded; alignment/scales tiny exact transfer layout not replayed.','PCIe ceilings use optimistic sustained50GB/s with perfect overlap and exclude dense/CPU/GPU compute. Initial seed cost is reported separately, not hidden in miss counts.','Current n1 routes are real and held split preserved; speculative wrong-branch routes in DSpark may differ.'],'seconds':time.monotonic()-start};out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'out':str(out),'seconds':result['seconds'],'held':[{'policy':x['policy'],**{k:x['aggregate']['held_steady_skip8'][k]for k in('hit_rate','full_layer_hit_rate','h2d_raw_bytes_per_token','h2d_floor_ms_per_token_at50GBps','h2d_only_tps_ceiling_at50GBps')}}for x in records]},indent=2))
