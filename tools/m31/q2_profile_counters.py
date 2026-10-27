#!/usr/bin/env python3
"""Read-only per-request host profile aggregation. No sampling/GPU/model load.
Outputs are diagnostics, not hardware DRAM/HBM counters or speed acceptance.
"""
import argparse,datetime,hashlib,json,re,statistics
from pathlib import Path
LOGROOT=Path('/mnt/ai/logs/m31-port')
KNOWN={
 'q2':{'routed_weight_bytes_per_n1':4183031808,'fixed_GPU_weight_bytes_including_norms_head':9199661568,'head_weight_bytes':691421184},
 'iq3':{'routed_weight_bytes_per_n1':5117313024,'fixed_GPU_weight_bytes_including_norms_head':10679794176,'head_weight_bytes':1008322560},
}
PATTERNS={
 'forward':'M31 forward ', 'stages':'M31 stages ', 'dense':'M31 dense-stages ',
 'head':'M31 head ', 'shared':'M31 shared-stages ',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def numbers(line):return {k:float(v) if '.' in v or 'e' in v.lower() else int(v) for k,v in re.findall(r'(\w+)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)',line,re.I)}
def speed_rows(path):
 if not path.exists():return []
 lines=path.read_text().splitlines();delim='\\t' if '\\t' in lines[0] else '\t';rows=[line.split(delim)for line in lines if line]
 return [dict(zip(rows[0],row))for row in rows[1:]]
def avg(rows,key):return statistics.mean(row[key]for row in rows)if rows else None
def stats(rows,keys):
 return {'count':len(rows),'mean':{key:avg(rows,key)for key in keys},'median':{key:statistics.median(row[key]for row in rows)if rows else None for key in keys}}
def analyze(directory,quant):
 candidates=list(directory.glob('*http-session.server.log'))
 if len(candidates)!=1:raise ValueError(f'{directory}: expected one server log, found{len(candidates)}')
 logfile=candidates[0];blocks=[];current={key:[]for key in PATTERNS}
 for line in logfile.read_text(errors='strict').splitlines():
  if 'M31 request timings ' in line:
   timings=json.loads(line.split('M31 request timings ',1)[1]);blocks.append((current,timings));current={key:[]for key in PATTERNS}
  else:
   for key,prefix in PATTERNS.items():
    if prefix in line:current[key].append(numbers(line.split(prefix,1)[1]));break
 rows=speed_rows(directory/'speed.tsv');report=[];checks={}
 for index,(events,t) in enumerate(blocks):
  prompt=int(t['prompt_n']);predicted=int(t['predicted_n']);expected=max(0,predicted-1)
  match=[row for row in rows if int(row['prompt_n'])==prompt and int(row['gen_n'])==predicted]
  label=match[0]['depth']if len(match)==1 else ('warmup'if predicted<=32 else 'request-'+str(index))
  dec={kind:[v for v in group if v.get('pos',-1)>=prompt] for kind,group in events.items()}
  pp={kind:[v for v in group if v.get('pos',-1)<prompt] for kind,group in events.items()}
  stage=dec['stages'];forward=dec['forward'];dense=dec['dense'];head=dec['head'];shared=dec['shared']
  tests={
   'forward_count_is_predicted_minus1':len(forward)==expected,
   'decode_positions_contiguous':[v['pos']for v in forward]==list(range(prompt,prompt+expected)),
   'decode_n1_only':all(v['n']==1 for v in forward),
   'stage_count_matches':len(stage)==expected,'dense_count_matches':len(dense)==expected,'head_count_matches':len(head)==expected,
   'prefill_token_sum_matches':sum(v['n']for v in pp['forward'])==prompt,
   'no_cross_boundary_forward':all(v['pos']+v['n']<=prompt or v['pos']>=prompt for v in events['forward']),
   'native_TG_numerator_matches':abs(t['predicted_per_second']-(1000*expected/t['predicted_ms']if t['predicted_ms']>0 else 0))<1e-6,
  }
  routed=avg(stage,'routed_compute_ms');dense_wall=avg(stage,'dense_ms')
  head_total=statistics.mean(v['setup_ms']+v['compute_ms']+v['read_ms']for v in head)if head else None
  steps_ms=t['predicted_ms']/expected if expected else None
  forward_ms=1000*avg(forward,'seconds') if forward else None
  prepare_ms=avg(stage,'prepare_ms')
  # Logical matrix bytes / complete CPU routed host duration, including Q8K
  # quantizers, OAI, pool barriers. No actual RAM transaction count was sampled.
  cpu_logical_GB_s=KNOWN[quant]['routed_weight_bytes_per_n1']/1e6/routed if routed else None
  item={'label':label,'request_index':index,'prompt_n':prompt,'emitted_or_stop_sample_n':predicted,'TG_numerator_forward_n':expected,
   'observed_decode_n1':len(forward),'prefill_forward_count':len(pp['forward']),'prefill_n1_tail_count':sum(v['n']==1 for v in pp['forward']),
   'checks':tests,'native_timings':t,'per_decode_forward_ms':{'request_interval':steps_ms,'forward_excluding_prepare_head':forward_ms,'prepare':prepare_ms,'dense':dense_wall,'CPU_routed_compute':routed,'head_setup_compute_read':head_total,
    'interval_minus_prepare_forward_head':steps_ms-prepare_ms-forward_ms-head_total if expected else None},
   'CPU_routed_logical_GB_s':cpu_logical_GB_s,
   'stages':stats(stage,['prepare_ms','dense_ms','routed_setup_ms','routed_compute_ms','routed_read_ms','residual_ms']),
   'dense_subsets':stats(dense,['build_ms','allocate_ms','upload_ms','compute_ms','read_ms','destroy_ms']),
   'head':stats(head,['setup_ms','compute_ms','read_ms']),
   'shared':stats(shared,['enqueue_ms','wait_ms','read_ms']),
   'speed_tsv_row':match[0]if len(match)==1 else None}
  report.append(item)
 checks={'all_requests_checked':all(all(v['checks'].values())for v in report),'no_unfinished_forward_records':not any(current.values()),'at_least_one_request':bool(report)}
 return {'directory':str(directory),'quant':quant,'log':str(logfile),'log_sha256':sha(logfile),'weight_bytes':KNOWN[quant],
  'speed_tsv_sha256':sha(directory/'speed.tsv')if(directory/'speed.tsv').exists()else None,'checks':checks,'pass':all(checks.values()),'requests':report}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',nargs=2,action='append',metavar=('QUANT','DIR'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 runs=[analyze(Path(path),quant)for quant,path in a.run]
 pooled=[]
 for quant,label in sorted({(run['quant'],r['label'])for run in runs for r in run['requests']if r['label']!='warmup'}):
  group=[r for run in runs if run['quant']==quant for r in run['requests']if r['label']==label]
  count=sum(r['observed_decode_n1']for r in group)
  means={key:sum(r['per_decode_forward_ms'][key]*r['observed_decode_n1']for r in group)/count for key in group[0]['per_decode_forward_ms']}
  pooled.append({'quant':quant,'depth':label,'requests':len(group),'n1_forwards':count,'per_forward_ms':means,'pooled_TG_per_second':1000/means['request_interval'],'CPU_routed_logical_GB_s':KNOWN[quant]['routed_weight_bytes_per_n1']/1e6/means['CPU_routed_compute'],'dense_read_ms':sum(r['dense_subsets']['mean']['read_ms']*r['observed_decode_n1']for r in group)/count,'scope':'Pooled forwards divided by total native TG elapsed, not arithmetic median of per-request om speed.'})
 comparisons=[]
 for q2 in [r for r in runs if r['quant']=='q2']:
  for iq3 in [r for r in runs if r['quant']=='iq3']:
   for q in q2['requests']:
    matched=[b for b in iq3['requests'] if b['label']==q['label'] and q['label']!='warmup']
    if len(matched)!=1:continue
    b=matched[0];qm=q['per_decode_forward_ms'];bm=b['per_decode_forward_ms']
    comparisons.append({'depth':q['label'],'q2_run':q2['directory'],'iq3_run':iq3['directory'],'Q2_minus_IQ3_ms':{key:qm[key]-bm[key] for key in qm},'q2_CPU_logical_GB_s':q['CPU_routed_logical_GB_s'],'iq3_CPU_logical_GB_s':b['CPU_routed_logical_GB_s'],'TG_change_percent':100*(bm['request_interval']/qm['request_interval']-1),'scope':'Same-day macro workload across different quants/outputroutes; not fixed-ID pair or isolated kernel attribution.'})
 report={'script_sha256':sha(Path(__file__)),'runtime_source_sha256':{name:sha(Path('/mnt/ai/Xolotl-m31')/name) for name in ('tools/m31/m31_engine.cpp','tools/m31/m31_serve.cpp','tools/m31/serve_m31.py')},'pooled_by_quant_depth':pooled,'comparisons':comparisons,'format':'m31-q2-host-profile-summary-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':all(v['pass']for v in runs),'runs':runs,
  'semantics':['Engine::forward timings reset on EVERY forward; stages/dense-stages are per-forward aggregates over60layers, not cumulative process counters.',
   'GpuExperts print_profile forward/hit/miss/upload/etc are cumulative since init; engine.reset does not reset_stats. GPU for-wards count layerFFN calls, not full60-layer Engine forwards. Repeated identical n1 snapshots must not be summed. Destructor expertcache/prefilluploads also lifetime totals, including all requests/prefills.','Native sampling_ms counts all samples including sample0 outsideTG; interactive_wait_ms includes finalACK after decode_end. Do not subtract complete aggregate sampling/ACK counters as exact TG-only costs.',
   'Native predicted_n counts sampled outputs, including a sampled EOS/stop; ordinary decode forwards =max(0,predicted_n-1). Sample0 belongs to prefill; use actual forward count, not256 for a255-forward TG interval.','Position>=prompt_n identifies generation forwards/heads. Singleton prefill tails and prompt head are excluded from decode averages. Warmup remains separate; official cold first forwards/bucket transitions are retained.','forward seconds excludes prepare/embedding/attention-input upload and excludes head; stages dense_ms and routed fields partition different host intervals. head setup/compute/read is separate.','dense-stages are subsets of dense_ms; compute_ms may be asynchronous enqueue; read_ms includes GPU kernel wait plus readback/sync, not PCIe DMA duration. Do not add dense_ms and dense subsets.','Shared expert GPU work is queued outside dense and overlaps routed CPU; shared enqueue/wait/read are host intervals, not the shared GPU kernel runtime. No HBM bandwidth can be recovered from these host counters.','CPU routed logical GB/s =all57layers4/128oforiginal routed weight bytes divided by routed_compute host duration. Includes activation quant/OAI/barriers and assumes cachedGPU off. It is not measured DRAM traffic.','GPU fixed-weight payload bytes include original names once (no packed aliases), norms and head; dynamic KV/MSA activation traffic excluded. Byte reduction alone is not a measured latency saving.','Residual interval minus prepare/forward/head includes sampler, RPC/ACK wait, logging and other boundary overhead; these fields are rounded, so it is a diagnostic difference, not exact hardware accounting.'],
  'weight_provenance':{'q2_header':'/mnt/ai/logs/m31-port/port7-ud-q2-header-audit-v2.json','comparison':'/mnt/ai/logs/m31-port/port7-ud-q2-vs-iq3-header-full-v2.json','iq3_dense_fixed_bytes':'AcceptedV8 fixedlogicalpayload10,679,794,176; includes1Dnorms, excludes embedding/expert weights.','routed_bytes_exclude':'NVFP4 external scales negligible, cache/CPU reuse/activation bytes not counted.'},
  'source_semantics_files':['tools/m31/m31_engine.cpp:Engine::forward/logits/routed_cpu_into/layer','tools/m31/m31_serve.cpp ordinary sample/forward timer loop','tools/m31/serve_m31.py done timing stderr'],'scope':'Read-only profile diagnostic. Compare actual om acceptance separately; no CPU/GPU sampling, no build/lock/model mutations.'}
 a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'out':str(a.out),'pass':report['pass'],'requests':[(v['directory'],len(v['requests']))for v in runs]},indent=2))
 raise SystemExit(0 if report['pass'] else 1)
if __name__=='__main__':main()
