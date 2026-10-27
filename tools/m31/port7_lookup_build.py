#!/usr/bin/env python3
"""One C++ TU; reuse the existing event-only CUDA object and frozen archives."""
import datetime,hashlib,json,os,pathlib,shutil,subprocess,time
R=pathlib.Path('/mnt/ai/Xolotl-m31');O=R/'build-port7-lookup/v1';F=R/'build-port7-candidate/v7';T=R/'build-port7-dense/ops-v2/timing.o'
assert os.environ.get('INVOCATION_ID');assert not O.exists()
O.mkdir(parents=True);S=O/'source';S.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();base=json.loads((F/'build-receipt.json').read_text())
for name in ('msa_sorted_lookup.hpp','port7_msa_sorted_lookup_check.cpp'):shutil.copyfile(R/'tools/m31'/name,S/name)
compile=[str(S/'port7_msa_sorted_lookup_check.cpp')if x==str(F/'source/port7_verify_dump.cpp')else str(O/'check.o')if x==str(F/'port7_verify_dump.o')else x for x in base['compile']['port7_verify_dump']]
link=[x for x in base['link']['port7_verify_dump']if x not in (str(F/'m31_engine.o'),str(F/'m31_gpu_experts.o'),str(F/'gpu_cached_exact.o'),'libm31_engine.a')]
link=[str(O/'check.o')if x==str(F/'port7_verify_dump.o')else str(O/'port7_msa_sorted_lookup_check')if x==str(F/'port7_verify_dump')else '-Wl,--dependency-file='+str(O/'link.d')if x.startswith('-Wl,--dependency-file=')else str(R/'build-m31-cuda'/x)if x.endswith('.a')and not pathlib.Path(x).is_absolute()else x for x in link]
link.insert(link.index(str(O/'check.o'))+1,str(T))
archives={x:sha(x)for x in link if x.endswith('.a')};expected=base['libraries']
for name,digest in archives.items():
 if name in expected:assert expected[name]==digest,('archive drift',name)
receipt={'format':'m31-lookup-component-build-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Standalone idx expansion proof/timing only, no Engine/model link or CMake edits. Existing event-only CUDA TU is reused with same private backend context layout.','frozen_base_receipt':str(F/'build-receipt.json'),'frozen_base_receipt_sha256':sha(F/'build-receipt.json'),'source_sha256':{str(p):sha(p)for p in S.iterdir()},'timing_object':str(T),'timing_object_sha256':sha(T),'timing_source_sha256':sha(R/'build-port7-dense/ops-v2/source/port7-dense-ops-timing.cu'),'archives_sha256':archives,'compile_argv':compile,'link_argv':link,'stages':[],'pass':False}
try:
 for name,cmd in [('compile',compile),('link',link)]:
  t=time.monotonic()
  with(O/(name+'.log')).open('wb')as f:rc=subprocess.run(cmd,cwd=O,stdout=f,stderr=subprocess.STDOUT).returncode
  receipt['stages'].append({'name':name,'rc':rc,'seconds':time.monotonic()-t});assert rc==0,(name,rc)
 exe=O/'port7_msa_sorted_lookup_check';exe.chmod(0o555);receipt['elf']={'path':str(exe),'sha256':sha(exe),'bytes':exe.stat().st_size};receipt['pass']=True
except BaseException as e:receipt['error']=repr(e)
(O/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'pass':receipt['pass'],'out':str(O),'elf':receipt.get('elf')}));raise SystemExit(0 if receipt['pass']else 1)
