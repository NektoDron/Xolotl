#!/usr/bin/env python3
"""Prepare a private frozen FFN header/replay only. No build or model run."""
import argparse,hashlib,json,pathlib,shutil
P=pathlib.Path;R=P('/mnt/ai/Xolotl-m31');p=argparse.ArgumentParser();p.add_argument('--out',type=P,required=True);a=p.parse_args();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
base=R/'build-port7-cpu-pair-batch';sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
for name in ('cpu_batched_verify.hpp','cpu_batched_dot.hpp','cpu_fused_decode.hpp','cpu_direct_decode.hpp','cpu_gpu_host_executor.hpp','cpu_threadpool_config.hpp','port7_cpu_actual_replay.cpp'):shutil.copyfile(base/name,out/name)
shutil.copyfile(R/'tools/m31/port7_cpu_stage_trace.hpp',out/'port7_cpu_stage_trace.hpp')
shutil.copyfile(R/'build-port7-candidate/v7/source/m31_geometry.hpp',out/'m31_geometry.hpp')
header=out/'cpu_fused_decode.hpp';s=header.read_text();assert sha(header)=='6bc92628d90d1ec8b127a9136edccb53b2797b1d37be1a568edc85548fb3f59c'
s=s.replace('#include <tuple>','#include <tuple>\n#include "port7_cpu_stage_trace.hpp"');start=s.index('    static void compute(ggml_tensor * output, int ith, int nth, void * opaque)');end=s.index('\npublic:\n    CpuFusedDecode(',start);body=s[start:end];segments=body.split('        ggml_barrier(self.pool);');assert len(segments)==6
items=['ith == 0 ? 1 : 0','port7_cpu_trace::count(self.used*self.ff*2,ith,nth)','port7_cpu_trace::count(self.hidden.size(),ith,nth)','size_t(ith)<self.used ? (self.used-1-size_t(ith))/size_t(nth)+1 : 0','port7_cpu_trace::count(self.used*self.embd,ith,nth)','port7_cpu_trace::count(self.embd,ith,nth)']
for phase in range(6):
 line=f'        port7_cpu_trace::start(ith,{phase},{items[phase]});\n'
 if phase==0:segments[0]=segments[0].replace('        if (ith == 0) {',line+'        if (ith == 0) {',1)
 else:segments[phase]=line+segments[phase]
 if phase<5:segments[phase]+=f'        port7_cpu_trace::finish(ith,{phase});\n        ggml_barrier(self.pool);\n        port7_cpu_trace::passed(ith,{phase});'
 else:
  loc=segments[phase].rfind('\n    }');assert loc>=0;segments[phase]=segments[phase][:loc]+'\n        port7_cpu_trace::finish(ith,5);'+segments[phase][loc:]
header.write_text(s[:start]+''.join(segments)+s[end:]);replay=out/'port7_cpu_actual_replay.cpp';s=replay.read_text();needle='one_output);expected.insert';assert s.count(needle)==1;s=s.replace(needle,'one_output);port7_cpu_trace::commit(l.index,threads);expected.insert');needle='    ggml_backend_cpu_set_threadpool(cpu,nullptr);';assert s.count(needle)==1;s=s.replace(needle,'    port7_cpu_trace::report();\n'+needle)
needle='const int count=int(routes.at({c,layers[0].index}).size());';assert s.count(needle)==1;s=s.replace(needle,'const int count=std::min(32,int(routes.at({c,layers[0].index}).size()));');replay.write_text(s)
manifest={'format':'m31-private-cpu-skew-proposal-v1','status':'SOURCE_PREPARED_NOT_BUILT_OR_RUN','base':str(base),'base_fused_sha256':'6bc92628d90d1ec8b127a9136edccb53b2797b1d37be1a568edc85548fb3f59c','files_sha256':{p.name:sha(p)for p in out.iterdir()},'scope':'Frozen existing actual-route component with private default-mode0 CpuFusedDecode timestamp/row-count records. Production/Engine untouched. Existing serial-vs-grouped bit checks retained.','phases':['inputquant','gate_up','oai','downquant','down','ranksum'],'measurement':'Perthread work and barrier timestamps via monotonic vDSO; GU/down CPU IDs and migrations. Records256Balignedperworker, no timestamp atomics. Worker clock-call overhead not calibrated, scheduling perturbs behavior.','limits':['Original replay serial_s includes observer aggregation; DO NOT use it as throughput benchmark.','Standalone component has no GPU helper contention; model/perf spin proportions do not transfer directly.','Future build and heavy replay require validation after V9 build in a clean window.']};(out/'proposal.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out/'proposal.json')
