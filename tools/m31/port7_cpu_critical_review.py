#!/usr/bin/env python3
"""Read-only small text/metadata analysis; no profiler/model/benchmark execution."""
import collections,datetime,hashlib,json,pathlib,re,statistics
B=pathlib.Path('/mnt/ai/logs/m31-port');R=pathlib.Path('/mnt/ai/Xolotl-m31');result={'format':'m31-cpu-critical-review-v1','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only V5/V6 profiler/text receipts and frozen source. Diagnostic intervals; no new model/CPU test or hardware CPU/DRAM counters.','sweeps':{}}
for folder in ['port7-dspark-shared-v5-sweep-v1','port7-dspark-v6-ops1-cache1-sweep-v1']:
 rows=[];cache=[]
 for path in(B/folder).glob('*-card-job-*.log'):
  for line in path.read_text().splitlines():
   if line.startswith('M31 verify pos='):
    row={k:float(v)for k,v in re.findall(r'([a-z_]+)=([0-9.]+)',line)}
    if row.get('build_ms')==0 and row.get('reconfigure_ms',0)<.01:rows.append(row)
   elif line.startswith('M31 GPU cached batched '):cache.append({k:float(v)for k,v in re.findall(r'([a-z_]+)=([0-9.]+)',line)})
 out={'logs':[str(x)for x in(B/folder).glob('*-card-job-*.log')],'selection':'build_ms=0 and reconfigure_ms<0.01ms; medians are diagnostic, not paired same-route controlled benchmark.','by_n':{}}
 for n in sorted({int(x['n'])for x in rows}):
  subset=[x for x in rows if int(x['n'])==n];out['by_n'][str(n)]={'frames':len(subset),'median_ms':{key:statistics.median(x[key]for x in subset)for key in ['dense_ms','cpu_ms','total_ms']+(['cpu_ffn_ms','hybrid_ms']if'cpu_ffn_ms'in subset[0]else[])}}
  if'cpu_ffn_ms'in subset[0]:out['by_n'][str(n)]['paired_cpu_minus_cpuFFN_median_ms']=statistics.median(x['cpu_ms']-x['cpu_ffn_ms']for x in subset)
 if cache:out['last_available_cumulative_cache_snapshot']=cache[-1];out['cache_snapshot_scope']='Printed at next prefill forward; excludes final arm suffix, not complete all20arms.'
 result['sweeps'][folder]=out
trace=json.loads((B/'port7-verify-trace-v6/trace.json').read_text());result['trace']={'file':str(B/'port7-verify-trace-v6/trace.json'),'scope':trace['scope'],'frames':trace['measured_frames'],'tokens':trace['measured_target_tokens'],'summary':trace['summary'],'cpu_sampling':False}
result['findings']=[
 {'finding':'Actual cache hits lower than held-out n1 estimate','evidence':'Last available V6 sweep snapshot678072routes/81355hits=11.998%, allmiss39291/81567calls=48.17%. Held-out n1 static29019.08% is another corpus/window. Batched grouping further amortizes hot expert bytes.','confidence':'observed counters; exact unique-byte benefit for these speculative branches not instrumented'},
 {'finding':'CPU FFN saves K8 work, wrapper costs consume it','evidence':'V5OFF K8 median wholeCPU151.112ms; V6ON CPUFFN140.907ms, wholeCPU152.296ms; paired wholeCPU−CPUFFN median11.922ms. Difference of medians is illustrative, not a wall decomposition.','confidence':'observed diagnostic intervals; V5/V6 also differ batch_ops, not an isolated cache A/B'},
 {'finding':'Metadata allocation/work partition is not established as main steady bottleneck','evidence':'Frozen CpuBatchedVerify keeps ggml graph/allocator and vector capacities. Weighted bounds computed once/group/worker/role, outside each dot row. No per-forward arena or full expert allocation. Steady trace build_ms=0/reconfigure≈0; CPUFFN remains193.797ms/K8 traced. Small full-rank zero/copy buffers exist but are orders below multi-GB weight traffic.','confidence':'source evidence; host malloc/cycles not profiled'},
 {'finding':'No POSIX msync in CPU FFN; relevant synchronization is graph/pool/barriers and CUDA joins','evidence':'Four explicit ggml_barrier per CPU batch plus ggml pool entry/exit; barriers are seq-cst fan-in/spin.80 threads,poll50,no affinity. CUDA API wait time includes GPU work; do not sum it with CPUFFN or GPU busy.','confidence':'source/config facts; CPU barrier fraction unmeasured'},
 {'finding':'Thread contention remains plausible, not proven','evidence':'80 ggml workers plus8 host-executor workers and cached private worker; EPYC9654 one NUMA node/96physical192logical. No affinity in sweep. Nsys sample=none/cpuctxsw=none, so no call stacks/contextswitch attribution.','confidence':'hypothesis, not cause measured'},
 {'finding':'Cache growth alone is not supported as near-term speed lever','evidence':'Current exact cached n1 GPU kernels use two joins and host OAI/activation quant, lose over12ms/K8 to helper/join/merge beyond CPUFFN. New resident slots also consume64K KV budget. Need cheaper grouped-hit kernels/joins, actual unique-byte savings and cache0/1 same-ELF proof before more VRAM.','confidence':'source/counters plus inference'},
 {'finding':'Cheap next levers','evidence':'Use already proved pair2 optin for small component gain (trace/sweep runpair0), retain80threads; reuse full-K/nk geometry states with bounded SUM-accounted memory to avoid reconfigure/build/CUDAgraph warmup on frequent n changes; V7 sharedFFN overlap can hide shared GPU under CPU. Reserved dedicated GPU-host CPUs/lower bounded poll is a small controlled candidate only, not another broad sweep. Larger CPU vector rewrite should target IQ3(88.1%bytes), preserving per-RHS FMA/hsum ordering.','confidence':'proposals, performance pending'}]
(B/'port7-cpu-critical-review-v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('written',B/'port7-cpu-critical-review-v1.json')
