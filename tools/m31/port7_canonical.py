#!/usr/bin/env python3
"""Prepare/run an isolated canonical CMake CPU->CUDA build, never GPU acceptance.

Run only after the coordinator freezes source and selects a runtime profile.
`plan` prints commands without creating directories or configuring/building.
"""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,time
P=pathlib.Path
BASE='6050b467c07a95c7b87debd4ddac2f8ba8d9b807'
EXT={'.c','.cc','.cpp','.cxx','.h','.hpp','.cuh','.cu','.inc','.cmake','.py','.sh','.bat','.ps1','.json','.toml','.yaml','.yml','.patch','.txt','.in','.def','.s','.S','.asm','.ld','.rc','.metal','.comp','.glsl'}
SKIP={'.git','_deps','CMakeFiles','__pycache__','node_modules','.venv','venv'}
TARGETS=('m31_inspect','m31_generate','m31_serve','m31_expert_check','m31_dspark_generate','m31_dspark_probe','port7_verify_dump')
PATCHES=('tools/xolotl/ggml/xolotl-ggml.patch','tools/m31/ggml-decode-fusion.patch','tools/m31/ggml-exact-mmv.patch','tools/m31/ggml-rms-only-fusion.patch')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def fp(p):
 p=P(p);s=p.stat();return {'sha256':sha(p),'bytes':s.st_size,'symlink':os.readlink(p) if p.is_symlink() else None}
def write(p,x):P(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def files_under(directory):
 if not directory.exists():return
 for path,dirs,files in os.walk(directory):
  dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('build-') and d!='build']
  for name in files:
   p=P(path)/name
   if p.is_file() and (p.suffix in EXT or name=='CMakeLists.txt'):yield p

def sources(root):
 # Tiny-model fixtures import these siblings via tools/ from their copied
 # source-tree. Keep the same import closure in the immutable support snapshot.
 selected={root/'CMakeLists.txt',root/'AGENTS.md',root/'tools/gguf_reader.py',root/'tools/gguf_writer.py'}
 for name in ('include','src','cmake','serve','tools/m31','tools/ds4/cmake','tools/xolotl/ggml','third_party/ggml'):
  selected.update(files_under(root/name) or ())
 # CMake helpers outside the obvious M31 subtree can affect included targets.
 for name in ('tools','cmake'):
  selected.update(p for p in (files_under(root/name) or ()) if p.suffix=='.cmake' or p.name=='CMakeLists.txt')
 selected.update(root/p for p in PATCHES)
 return {str(p.relative_to(root)):fp(p) for p in sorted(selected) if p.is_file()}
def vendor_manifest(vendor):
 # Include extensionless support files and templates too, not just compilable TUs.
 paths=[p for p in vendor.rglob('*') if p.is_file() and '.git' not in p.relative_to(vendor).parts]
 return {str(p.relative_to(vendor)):fp(p) for p in sorted(paths)}
def readcache(path):
 result={}
 for line in path.read_text().splitlines():
  if line and not line.startswith(('#','//')) and '=' in line and ':' in line.split('=',1)[0]:
   k,v=line.split('=',1);result[k.split(':',1)[0]]=v
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','build']);p.add_argument('--root',type=P,default=P('/mnt/ai/Xolotl-m31'));p.add_argument('--out',type=P,required=True);p.add_argument('--profile',type=P,required=True);p.add_argument('--tokenizer',type=P,required=True);p.add_argument('--model',type=P,required=True);p.add_argument('--vendor-source',type=P);p.add_argument('--jobs',type=int,default=24);p.add_argument('--draft-fixture',type=P,help='Optional exported real-feature probe JSON; no GPU inference is run');p.add_argument('--execute',action='store_true');a=p.parse_args()
 root=a.root.resolve();out=a.out.resolve();vendor_source=(a.vendor_source or root/'build-m31-cpu/_deps/strata_llamacpp-src').resolve()
 if not 1<=a.jobs<=24:p.error('jobs must be1..24')
 if out==root or not str(out).startswith(str(root/'build-port7-canonical')+'/'):p.error('out must be a NEW subdirectory below ROOT/build-port7-canonical/')
 runtime_profile=json.loads(a.profile.read_text())
 if not isinstance(runtime_profile,dict) or runtime_profile.get('format')!='m31-runtime-profile-v1' or 'inherits' in runtime_profile:
  p.error('--profile must be a resolved m31-runtime-profile-v1 JSON without inherits; resolve parents before canonical freeze')
 command=['sudo','systemd-run','--unit=codex-m31-port7-canonical-WINNER','-p','User=nekto','-p','KillMode=mixed','-p','TimeoutStopSec=600','-p','TimeoutStartSec=24h','/bin/bash',str(root/'tools/m31/port7_canonical_build.sh'),'--out',str(out),'--profile',str(a.profile.resolve()),'--tokenizer',str(a.tokenizer.resolve()),'--model',str(a.model.resolve()),'--vendor-source',str(vendor_source),'--jobs',str(a.jobs),'--execute']
 if a.draft_fixture:command+=['--draft-fixture',str(a.draft_fixture.resolve())]
 if a.command=='plan':
  print(json.dumps({'command':command,'builds':[str(out/'build-cpu'),str(out/'build-cuda')],'vendor':'Sparse local shared-object clone of pinned vendor; own working tree only; apply common+three explicit API patches (decode, exact-MMV, RMS) before configure','targets':TARGETS,'gpu_acceptance':'Deferred to aictl/card-lock validation after this build; this script never acquires a GPU.','execute_requested':False},indent=2));return
 if not a.execute:p.error('build requires explicit --execute after coordinator source-freeze ACK')
 if not os.environ.get('INVOCATION_ID') or os.getuid()!=__import__('pwd').getpwnam('nekto').pw_uid:raise RuntimeError('Build must run in systemd User=nekto through port7_canonical_build.sh')
 if out.exists():raise RuntimeError('Refusing to overwrite an existing canonical output: '+str(out))
 for f in (root/'CMakeLists.txt',a.profile,a.tokenizer,a.model,vendor_source/'ggml/CMakeLists.txt',P('/home/nekto/.local/include/cuda13_glibc_compat.h')):
  if not f.is_file():raise RuntimeError('Missing required local input: '+str(f))
 out.mkdir(parents=True)
 logdir=P('/mnt/ai/logs/m31-port')/('port7-canonical-'+os.environ['INVOCATION_ID'])
 logdir.mkdir();(out/'logs').symlink_to(logdir,target_is_directory=True)
 receipt={'format':'m31-port7-canonical-cmake-v1','started':now(),'source_root':str(root),'output':str(out),'logs':str(logdir),'systemd_invocation':os.environ['INVOCATION_ID'],'jobs':a.jobs,'stages':[],'compile_complete':False,'gpu_acceptance_complete':False,'model_quality_complete':False}
 def stage(name,argv,cwd=None,env=None,timeout=21600):
  start=time.monotonic();log=out/'logs'/(name+'.log')
  with log.open('wb') as f:
   try:r=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);rc=r.returncode
   except subprocess.TimeoutExpired:rc=124
  receipt['stages'].append({'name':name,'argv':[str(x) for x in argv],'cwd':str(cwd) if cwd else None,'rc':rc,'seconds':time.monotonic()-start,'log':str(log),'log_sha256':sha(log)});write(out/'progress.json',receipt)
  if rc:raise RuntimeError(name+' failed rc'+str(rc)+'; '+str(log))
 def capture(argv,cwd=None):return subprocess.check_output([str(x) for x in argv],cwd=cwd,text=True).strip()
 before=sources(root);write(out/'source-before.json',before)
 def stable(label):
  current=sources(root);write(out/(label+'-source.json'),current)
  if before!=current:
   changed=sorted(k for k in set(before)|set(current) if before.get(k)!=current.get(k));write(out/'source-drift.json',changed);raise RuntimeError('Source/support changed during canonical build: '+repr(changed[:15]))
 protected=root/'build-port6-canonical/v8';protected_before={p.name:fp(p) for p in (protected/'m31_generate',protected/'m31_serve',protected/'build-receipt.json')};write(out/'protected-v8-before.json',protected_before)
 support=out/'source-tree'
 for rel,data in before.items():
  target=support/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
  if sha(target)!=data['sha256']:raise RuntimeError('Source changed while snapshotting '+rel)
 helpers=out/'runtime-helpers';helpers.mkdir();tokenizer=helpers/'m31_tokenizer';shutil.copyfile(a.tokenizer,tokenizer);tokenizer.chmod(0o555)
 profile=helpers/'runtime-profile.json';shutil.copyfile(a.profile,profile)
 external={str(f.resolve()):fp(f) for f in (a.profile,a.tokenizer,P('/home/nekto/.local/include/cuda13_glibc_compat.h'))}
 if a.draft_fixture:external[str(a.draft_fixture.resolve())]=fp(a.draft_fixture)
 write(out/'external-before.json',external)
 receipt['base_head']=capture(['git','rev-parse','HEAD'],root);receipt['git_status_before']=capture(['git','status','--porcelain=v1'],root);receipt['model_stat']={k:getattr(a.model.stat(),k) for k in ('st_size','st_ino','st_mtime_ns')};receipt['runtime_profile']=fp(profile);receipt['tokenizer']=fp(tokenizer)
 try:
  if capture(['git','rev-parse','HEAD'],vendor_source)!=BASE:raise RuntimeError('Unexpected local vendor HEAD; no fetch allowed')
  vendor=out/'vendor-managed';stage('vendor-local-clone',['git','clone','--shared','--no-checkout',vendor_source,vendor],timeout=300)
  stage('vendor-sparse',['git','sparse-checkout','set','--cone','ggml','cmake','common','vendor','include','src'],vendor,timeout=300)
  stage('vendor-base-checkout',['git','checkout','--detach',BASE],vendor,timeout=300)
  write(out/'vendor-base.json',vendor_manifest(vendor));receipt['vendor_base']={'commit':BASE,'origin_read_only':str(vendor_source),'patches':{rel:fp(root/rel) for rel in PATCHES}}
  for i,patch in enumerate(PATCHES):stage('vendor-patch'+str(i),['git','apply','--check',root/patch],vendor,timeout=60);stage('vendor-apply'+str(i),['git','apply',root/patch],vendor,timeout=60)
  patched=vendor_manifest(vendor);write(out/'vendor-patched.json',patched);(out/'vendor-managed.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=vendor));receipt['vendor_diff_sha256']=sha(out/'vendor-managed.patch')
  tools={'g++':['/usr/bin/g++-14','--version'],'gcc':['/usr/bin/gcc-14','--version'],'nvcc':['/usr/local/cuda/bin/nvcc','--version'],'cmake':['cmake','--version'],'git':['git','--version'],'python':[sys.executable,'--version']}
  receipt['toolchain']={name:{'version':capture(cmd),'executable':fp(P(shutil.which(cmd[0]) or cmd[0]).resolve())} for name,cmd in tools.items()}
  env={k:v for k,v in os.environ.items() if not k.startswith(('M31_','GGML_CUDA_'))};env|={'CC':'/usr/bin/gcc-14','CXX':'/usr/bin/g++-14','GIT_ALLOW_PROTOCOL':'file','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':''}
  testenv=env|{'M31_TEST_TOKENIZER':str(tokenizer),'M31_TEST_MODEL':str(a.model.resolve())}
  common=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=/usr/bin/gcc-14','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_DENSE=OFF','-DSTRATA_M31=ON','-DSTRATA_M31_DSPARK=ON','-DSTRATA_DSV41_FQ_OP=OFF','-DSTRATA_DSV41_HC_MIX=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DFETCHCONTENT_UPDATES_DISCONNECTED=ON','-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP='+str(vendor)]
  receipt['modes']={}
  for mode in ('cpu','cuda'):
   enabled=mode=='cuda';build=out/('build-'+mode);flags=common+['-DSTRATA_ENABLE_CUDA='+('ON' if enabled else 'OFF'),'-DSTRATA_GGML_CUDA='+('ON' if enabled else 'OFF')]+['-D'+key+'='+('ON' if enabled else 'OFF') for key in ('STRATA_M31_CPU_QUANTS_CUDA','STRATA_M31_FAST_FFN_CUDA','STRATA_M31_REGISTERED_ARENA_CUDA')]
   if enabled:flags+=['-DCMAKE_CUDA_ARCHITECTURES=120a','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCUDAToolkit_ROOT=/usr/local/cuda','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14','-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h']
   stable(mode+'-preconfigure');stage(mode+'-configure',['cmake','-S',root,'-B',build,*flags],env=env,timeout=1800)
   cache=readcache(build/'CMakeCache.txt')
   for key,value in {'STRATA_M31':'ON','STRATA_M31_DSPARK':'ON','STRATA_GGML_CUDA':'ON' if enabled else 'OFF','FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP':str(vendor)}.items():
    if cache.get(key)!=value:raise RuntimeError('Canonical cache mismatch '+key+': '+str(cache.get(key)))
   stage(mode+'-build',['cmake','--build',build,'--target',*TARGETS,'-j',str(a.jobs)],env=env);stable(mode+'-postbuild')
   if patched!=vendor_manifest(vendor):raise RuntimeError('Managed vendor source changed after its explicit patch phase')
   bins=out/('bin-'+mode);bins.mkdir();elves={}
   for name in TARGETS:
    source=build/name;target=bins/name;shutil.copyfile(source,target);target.chmod(0o555);elves[name]=fp(target)
    if elves[name]['sha256']!=sha(source):raise RuntimeError('ELF changed while freezing '+name)
   metadata={}
   for name in ('CMakeCache.txt','compile_commands.json'):
    target=out/(mode+'-'+name);shutil.copyfile(build/name,target);metadata[target.name]=fp(target)
   links={}
   for name in TARGETS:
    f=build/'CMakeFiles'/(name+'.dir')/'link.txt';target=out/(mode+'-'+name+'-link.txt');shutil.copyfile(f,target);links[name]=fp(target)
   libraries={str(f.relative_to(build)):fp(f) for f in sorted(build.rglob('*.a'))}
   dynamic={}
   for name in TARGETS:
    target=bins/name
    stage(mode+'-'+name+'-readelf',['readelf','-h','-l','-d',target],env=env,timeout=60)
    dynamic[name]=capture(['ldd',target])
   write(out/(mode+'-dynamic-libraries.json'),dynamic)
   receipt['modes'][mode]={'elf':elves,'cmake_metadata':metadata,'link_commands':links,'libraries':libraries,'configuration_argv':flags}
   tests=support/'tools/m31';stage(mode+'-compare-unit',[sys.executable,tests/'tests/test_compare.py'],env=testenv,timeout=300)
   stage(mode+'-math-final-elf',[sys.executable,tests/'tests/test_math.py','--engine',bins/'m31_generate','--inspect',bins/'m31_inspect'],env=testenv,timeout=900)
   stage(mode+'-prefix-real-worker',[sys.executable,tests/'tests/test_prefix_cache.py','--engine',bins/'m31_serve'],env=testenv,timeout=300)
   stage(mode+'-canonical-real-worker',[sys.executable,tests/'tests/test_canonical_worker.py','--worker',bins/'m31_serve'],env=testenv,timeout=300)
   stage(mode+'-server-real-vocab',[sys.executable,tests/'test_server.py'],env=testenv,timeout=600)
   stage(mode+'-common-server',[sys.executable,tests/'serve/test_common.py'],env=testenv,timeout=600)
   stage(mode+'-prefix-api',[sys.executable,tests/'tests/test_prefix_api.py'],env=testenv,timeout=300)
   stage(mode+'-dspark-launcher',[sys.executable,tests/'tests/test_dspark_launcher.py','--launcher',tests/'serve_m31.py'],env=testenv,timeout=300)
   # Compile isolated controller fixture only; final model ELFs above are never manually relinked.
   fixture=out/(mode+'-dspark-session-fixture');stage(mode+'-dspark-session-build',['/usr/bin/g++-14','-std=c++20','-O2','-I'+str(tests),'-I'+str(support/'include'),'-I'+str(vendor/'ggml/include'),tests/'tests/dspark_session_check.cpp','-o',fixture],env=env,timeout=300);stage(mode+'-dspark-session-contract',[fixture],env=testenv,timeout=300)
   if a.draft_fixture:
    d=json.loads(a.draft_fixture.read_text());cmd=[bins/'m31_dspark_probe','--backend','cpu','--model',a.model,'--sidecar',d['sidecar'],'--features',d['features'],'--rows',str(d['rows']),'--first',str(d['first']),'--depth',str(d.get('depth',3)),'--anchor',str(d['anchor'])]
    stage(mode+'-dspark-real-cpu-probe',cmd,env=testenv,timeout=1800)
   stable(mode+'-postfixtures')
  final_external={path:fp(P(path)) for path in external}
  if final_external!=external:raise RuntimeError('External profile/tokenizer/compat/fixture changed during build')
  protected_after={p.name:fp(p) for p in (protected/'m31_generate',protected/'m31_serve',protected/'build-receipt.json')}
  if protected_after!=protected_before:raise RuntimeError('Protected v8 artifacts changed')
  write(out/'protected-v8-after.json',protected_after);write(out/'vendor-final.json',vendor_manifest(vendor));stable('final')
  receipt|={'compile_complete':True,'source_stable':True,'vendor_stable_after_patch':True,'protected_v8_unchanged':True,'cpu_contract_checks_pass':True,'finished':now(),'git_status_after':capture(['git','status','--porcelain=v1'],root),'head_after':capture(['git','rev-parse','HEAD'],root)}
  write(out/'gpu-validation-required.json',{'status':'PENDING','elf':receipt['modes']['cuda']['elf'],'binaries':str(out/'bin-cuda'),'runtime_profile':fp(profile),'tokenizer':fp(tokenizer),'required':['GPU-offload oracle gate20 using Engine.verify, rollback/features/head1','long16K/32K selected rows and actual verify chunks','two-round unchanged om_bench4K/16K/32K in clean windows, PP no worse than v8','final ELF + frozen adapter GSM8K100>=95 and tools23>=19','native HTTP/SSE cancel/next request/tools/prefix+DSpark fallback smoke as configured'],'resource_rule':'Only coordinator aictl/card-lock launcher, GPU0; this builder never launches GPU work.'})
 except BaseException as exc:
  receipt['error']=repr(exc);receipt['finished']=now();write(out/'build-receipt.json',receipt);raise
 write(out/'build-receipt.json',receipt);print(json.dumps({'directory':str(out),'compile_complete':True,'cpu_checks':True,'gpu_model_acceptance':'PENDING'},indent=2))
if __name__=='__main__':main()
