#!/usr/bin/env python3
"""Build an isolated Q4_K/Q5_K dispatch proof against frozen canonical-v2 CUDA kernels.
Run via systemd under shared .build.lock. Does not alter the canonical build.
"""
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,sys

root=Path('/mnt/ai/Xolotl-m31')
base=root/'build-port7-canonical/v2'
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
source=out/'source';source.mkdir()
vendor=out/'vendor';vendor.mkdir()
shutil.copytree(base/'vendor-managed/ggml',vendor/'ggml')
source_names=('q2_dense_mmv_check.cpp','q2_dense_mmv_kernels.cu','q2_dense_mmv_probe.hpp','q2_dense_mmv_contract.cpp','m31_geometry.hpp')
for name in source_names:shutil.copy2(root/'tools/m31'/name,source/name)
oldguard='weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_Q6_K && weight->type != GGML_TYPE_Q8_0'
newguard='weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_Q4_K && weight->type != GGML_TYPE_Q5_K && weight->type != GGML_TYPE_Q6_K && weight->type != GGML_TYPE_Q8_0'
cu=vendor/'ggml/src/ggml-cuda/ggml-cuda.cu';text=cu.read_text();assert text.count(oldguard)==1;cu.write_text(text.replace(oldguard,newguard))
h=vendor/'ggml/include/ggml-cuda.h';text=h.read_text();assert '2D F32/Q6_K/Q8_0 weights' in text;h.write_text(text.replace('2D F32/Q6_K/Q8_0 weights','2D F32/Q4_K/Q5_K/Q6_K/Q8_0 weights'))
# Ensure the isolated implementation actually matches the patch under review.
subprocess.run(['git','apply','--reverse','--check',str(root/'tools/m31/ggml-exact-mmv.patch')],cwd=vendor,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=base/'build-cuda';archive_source=build/'ggml/src/ggml-cuda/libggml-cuda.a';archive_sha=sha(archive_source)
commands=json.loads((base/'cuda-compile_commands.json').read_text())
old=next(v for v in commands if v['file'].endswith('/ggml-cuda.cu'))
inc=['-I'+str(vendor/'ggml/include'),'-I'+str(vendor/'ggml/src'),'-I'+str(source)]
records=[]
for obj,src in [('ggml-cuda.cu.o',cu),('kernels.o',source/'q2_dense_mmv_kernels.cu')]:
    argv=shlex.split(old['command']);argv[1:1]=inc;argv[argv.index('-o')+1]=str(out/obj);argv[argv.index('-c')+1]=str(src)
    subprocess.run(argv,cwd=old['directory'],check=True);records.append({'cwd':old['directory'],'argv':argv})
archive=out/'libggml-cuda.a';shutil.copy2(archive_source,archive);subprocess.run(['ar','r',str(archive),str(out/'ggml-cuda.cu.o')],check=True)
links={}
for target,src in [('dense-mmv-check','q2_dense_mmv_check.cpp'),('mmv-contract','q2_dense_mmv_contract.cpp')]:
    argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG',*inc,'-I'+str(root/'include'),'-I'+str(base/'vendor-managed/vendor/nlohmann'),'-c',str(source/src),'-o',str(out/(target+'.o'))]
    subprocess.run(argv,check=True);records.append({'cwd':str(root),'argv':argv})
    argv=shlex.split((base/'cuda-m31_generate-link.txt').read_text());argv=[v for v in argv if not v.startswith('-Wl,--dependency-file=')]
    k=argv.index('-o');argv[k-1]=str(out/(target+'.o'));argv[k+1]=str(out/target)
    if target=='dense-mmv-check':argv[argv.index('libm31_engine.a')]=str(out/'kernels.o')
    else:argv.remove('libm31_engine.a')
    argv[argv.index('ggml/src/ggml-cuda/libggml-cuda.a')]=str(archive)
    subprocess.run(argv,cwd=build,check=True);links[target]={'cwd':str(build),'argv':argv}
contract=subprocess.run([str(out/'mmv-contract')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True)
(out/'cpu-contract.json').write_text(contract.stdout)
assert archive_sha==sha(archive_source)
report={'format':'m31-q2-dense-mmv-build-v1','canonical_base':str(base),'canonical_archive_unchanged_sha256':archive_sha,'patch_sha256':sha(root/'tools/m31/ggml-exact-mmv.patch'),'scope':'Isolated canonical-v2 archive overlay replacing dispatch TU only; actual stock MMVQ/MMVF kernels unchanged. Q4_K/Q5_K join existing exact marker set. No Engine ABI dependence.','source':{str(p.relative_to(out)):sha(p) for tree in (source,vendor) for p in tree.rglob('*') if p.is_file()},'compile':records,'link':links,'cpu_contract':json.loads(contract.stdout),'elf':{name:sha(out/name) for name in links},'common_cuh_sha256':sha(vendor/'ggml/src/ggml-cuda/common.cuh')}
(out/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'out':str(out),'elf':report['elf'],'cpu_contract':report['cpu_contract']},indent=2))
