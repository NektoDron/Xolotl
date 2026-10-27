#!/usr/bin/env python3
"""Lightweight evidence index. Metadata only: never opens logits/weights/ELFs."""
import argparse,datetime,hashlib,json,pathlib
ROOT=pathlib.Path('/mnt/ai/Xolotl-m31');LOG=pathlib.Path('/mnt/ai/logs/m31-port')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=pathlib.Path,required=True,help='NEW immutable JSON index; an existing path is never overwritten')
args=parser.parse_args();OUT=args.out.resolve()
if OUT.suffix!='.json':parser.error('--out must end in .json')
if OUT.exists():parser.error('--out already exists; choose a NEW basename')
if not OUT.parent.is_dir():parser.error('--out parent must already exist')
seen={};MAX=2_000_000

def load(path):
 p=pathlib.Path(path)
 if not p.is_absolute():p=LOG/p
 if not p.exists():return None
 if p.suffix not in ('.json','.jsonl') or p.stat().st_size>MAX:return None
 raw=p.read_bytes();seen[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 return [json.loads(line) for line in raw.decode().splitlines()] if p.suffix=='.jsonl' else json.loads(raw)
def ref(path):
 p=pathlib.Path(path)
 if not p.is_absolute():p=LOG/p
 return {'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
def scalar(d):return {k:v for k,v in d.items() if isinstance(v,(str,int,float,bool)) or v is None}
def compact_compare(path):
 d=load(path)
 if d is None:return ref(path)
 out=ref(path)|scalar(d)
 if isinstance(d.get('cases'),list):
  out['case_count']=len(d['cases'])
  if len(d['cases'])<=64:out['cases']=[{k:c[k] for k in ('case','tokens','prompt_tokens','rows','top1','top1_matches','mean_kl','max_kl','max_abs','identical','anchor','tail','internal','verify_calls','partial_commits','committed_feature_files_bitwise','bitwise_parity') if k in c} for c in d['cases']]
 if isinstance(d.get('commands'),list):
  out['command_returncodes']=[c.get('rc',c.get('returncode')) for c in d['commands']]
  out['all_commands_rc0']=bool(d['commands']) and all(rc==0 for rc in out['command_returncodes'])
 if isinstance(d.get('checks'),list):out['check_count']=len(d['checks']);out['failed_checks']=[x for x in d['checks'] if not x.get('pass',False)]
 return out
def manifestation(folder):
 path=folder+'/manifest.json';d=load(path)
 if d is None:return {'manifest':ref(path)}
 out={'manifest':ref(path)}
 for k in ('scope','purpose','timing_scope','executable','executable_sha256','source_receipt_sha256','profile','arms','expected_actual_prefix_chunks'):
  if k in d:out[k]=d[k]
 if 'files'in d:out['frozen_files_sha256']=d['files']
 sr=d.get('source_receipt') or d.get('build_receipt')
 if isinstance(sr,str):out['build_receipt']=ref(sr)
 elif isinstance(sr,dict):out['embedded_build_receipt']={k:sr[k] for k in ('scope','elf','cuda_override','cuda_header_sha256') if k in sr}
 return out

def variant(name,status,scope,folder=None,compares=(),notes=()):
 d={'name':name,'status':status,'proof_scope':scope,'release_accepted':False,'final_quality_accepted':False,'notes':list(notes)}
 if folder:d|=manifestation(folder)
 d['evidence']=[compact_compare(p) for p in compares]
 return d

old=load('port6r4-final-v8-acceptance-v1/speed/report.json');quality=load('port6r4-final-v8-acceptance-v1/quality/report.json');same=load('port7-v8-baseline-v1/report.json')
old_speed=old.get('speed',{}).get('arms',{}) if old else {}
base={'accepted_before_port7':True,'canonical_build_receipt':ref(ROOT/'build-port6-canonical/v8/build-receipt.json'),'canonical_elf_paths':[str(ROOT/'build-port6-canonical/v8'/n) for n in ('m31_generate','m31_serve')],'same_day':{'report':ref('port7-v8-baseline-v1/report.json'),'speed':same['speed']['arms'],'two_repetitions':False},'prior_acceptance':{'speed_report':ref('port6r4-final-v8-acceptance-v1/speed/report.json'),'speed':old_speed,'quality_report':ref('port6r4-final-v8-acceptance-v1/quality/report.json'),'quality':{k:quality['quality'][k] for k in ('complete','gsm8k','tools','pass')},'gate20':{k:old['numeric_gate'][k] for k in ('source','pass','rows','top1','mean_kl','max_kl')},'long':{'source':old['long_numeric_gate']['source'],'pass':old['long_numeric_gate']['pass'],'cases':[{k:c[k] for k in ('case','tokens','rows','top1','mean_kl','max_kl','max_abs') if k in c} for c in old['long_numeric_gate']['cases']]}}}
v8build=load(ROOT/'build-port6-canonical/v8/build-receipt.json')
base['canonical_recorded_elf_sha256']=v8build['elf'];base['canonical_source_stable']=v8build.get('source_stable');base['elf_rehash_performed_by_index']=False

variants=[]
variants.append(variant('CUDA fusion V3 fenced ON/OFF','selected_short_model_numerical_pass','Fixed256 teacher forcing: prefix128+decode128, 128 evaluated rows. Both fenced modes exact vs accepted v8, not a full gate20 or quality/speed acceptance.','port7-dense-fusion-v3',('port7-dense-fusion-v3/fusion_off-compare.json','port7-dense-fusion-v3/fusion_on-compare.json')))
variants.append(variant('CUDA fusion V4 unfenced ON','rejected_numerical','Single selected prompt: unfenced graph fusion differs strongly. Do not confuse with V3 fenced fusion ON.','port7-dense-fusion-v4',('port7-dense-fusion-v4/fusion_on-compare.json',)))
for folder,name in [('port7-dspark-naive-parity-numeric-v2','Naive joined batch K8'),('port7-dspark-native-k2-parity-v1','Naive joined batch K2'),('port7-dspark-exact-parity-numeric-v1','Exact per-query verifier V2'),('port7-dspark-batched-v4-parity-v1','Batched dense exact MMV verifier V4'),('port7-dspark-shared-v5-parity-v1','Shared-copy verifier V5')]:
 rows=load(folder+'/native/summary.jsonl') or []
 parity=[r for r in rows if r.get('mode')=='batch_parity'];good=bool(parity) and all(r.get('exact_greedy') and r.get('max_abs')==0 for r in parity)
 v=variant(name,'selected_short_model_parity_pass' if good else 'rejected_greedy_numerical','One selected corpus_3324 prompt, 128 teacher-forced rows; numerical-only jobs do not establish speed, final gate20, long-context or quality acceptance.',folder)
 v['summary']= [r for r in rows if r.get('mode')in('serial','batch_parity')];v['summary_file']=ref(folder+'/native/summary.jsonl');variants.append(v)
for folder,name in [('port7-dspark-persistent-v3-pilot-v1','Persistent verifier V3 pilot'),('port7-dspark-batched-v4-sweep-v1','V4 depth/p_min sweep'),('port7-dspark-shared-v5-sweep-v1','V5 shared-input depth/p_min sweep')]:
 rows=load(folder+'/native/summary.jsonl') or load(folder+'/spec/summary.jsonl') or [];arms=[r for r in rows if r.get('mode')=='spec']
 v=variant(name,'research_only_goal_not_reached','Actual DSpark selected-prompt greedy sequence/sweep; diagnostic timing only, never om benchmark acceptance.',folder)
 if arms:v['arms_summary']={'count':len(arms),'min_tps':min(r['tps'] for r in arms),'max_tps':max(r['tps'] for r in arms),'fastest':max(arms,key=lambda r:r['tps'])}
 for n in ('strict-sweep-audit-v1.json','strict-greedy-compare-v1.json'):
  if (LOG/folder/n).exists():v['evidence'].append(compact_compare(folder+'/'+n))
 v['notes'].append('Sweep timings may overlap builds as declared in manifests/audits; no clean om speed claim.');variants.append(v)
cache=load('port7-cached-agent-summary.json')
for gate in cache['gates']:
 v=variant('GPU cached decode '+gate['name'],'selected_short_numerical_pass' if gate['pass'] else 'rejected_numerical','One code128 teacher-forced tail against cache-OFF; native GPU arithmetic can fail on hot expert routes. Exact cached helper is separate.')
 v['evidence']=[compact_compare(gate['path'])];variants.append(v)
variants.append(variant('GPU exact cached hot290 autoregressive','selected_greedy_model_pass','Eight actual greedy prompt continuations, 1024 generated IDs total; no om/gate20/GSM/tools acceptance.','port7-cached-exact-greedy290-v1',('port7-cached-exact-greedy290-v1/greedy-compare.json',)))
variants.append(variant('Exact prefix cache','selected_model_protocol_pass','Real Engine, prompt head plus eight decode rows; specific prompt geometry. Separate from DSpark feature-ring cache.','port7-prefix-gpu-v1',('port7-prefix-gpu-v1/raw/result.json',)))
variants.append(variant('Native M3 worker','native_protocol_smoke_pass','Actual same-ELF/profile ordinary-vs-DSpark RPC checks, EOS/cancel/fed-position/sampling fallback. Large raw result is linked, not re-read by index.','port7-dspark-worker-smoke-v2',('port7-dspark-worker-smoke-v2/independent-audit.json',)))
variants[-1]['large_result_not_read']=ref('port7-dspark-worker-smoke-v2/results/result.json')
variants.append(variant('Standalone M3 HTTP facade','http_smoke_pass','Actual standalone HTTP greedy/sampling fallback/stop/cancel recovery, not benchmark or GSM/tools acceptance.','port7-dspark-http-smoke-v1',('port7-dspark-http-smoke-v1/result.json',)))
# V6 integration proof was subsequently completed; latest evidence is appended below.

components=[]
for name,path,scope in [
 ('CPU fused decode sweep','port7-cpu-sweep-v1/receipt.json','Original quant arithmetic; component bit proof and n1 thread/quant timings. Cheap fused optins not faster enough; v8 default unchanged.'),
 ('CPU batch initial grouping','port7-cpu-actual-check-v1/receipt.json','Actual-route CPU component; initial equal-row grouping slower, corrected by weighted partition.'),
 ('CPU weighted grouping','port7-cpu-weighted-check-v1/receipt.json','Component ownership and real-route serial-vs-grouped; no model TPS.'),
 ('CPU pair2 exact raw dots','port7-cpu-pair-proof-v1/receipt.json','1,966,080 real quant/RHS/scales dot results bitwise; not full model backend proof.'),
 ('CPU integrated pair0/1/2','port7-cpu-pair-batch-check-v1/receipt.json','Real weight FFN and actual-route component, synthetic activations; all modes bitwise.'),
 ('CPU mode2 threads80/64/96','port7-cpu-batch-threads-v1/receipt.json','Clean exclusive actual-route component, synthetic activations:80 threads best K2/4/8. Logical byte counters, not hardware DRAM counters/model TPS.'),
 ('CPU portability guard','port7-cpu-portability-v1/receipt.json','Native GCC/Clang, forced CPUID-false and minimal nonnative SDK stubs; not full MSVC/AArch64/HIP/SYCL model build.'),
 ('CPU masked assignments','port7-cpu-masked-proof-v1/receipt.json','All-hit/all-miss/mixed full-rank terms, independent F32 n1 reference; no model speed.'),
 ('GPU batched cached helper','port7-cached-batch-component-v1/component-result.json','Real quants component, separate standalone helper.'),
 ('GPU+masked CPU joint','port7-cached-batch-joint-component-v1/component-result.json','Joint full-rank contributions,39cases/3,391,488F32 bitwise; no Engine model integration acceptance.'),
 ('GPU cached wrapper','port7-cached-wrapper-component-v1/component-result.json','Real cache masks, callback lifetimes/recovery; later selected V6 integration parity is indexed separately, not universal model acceptance.'),
 ('Dense operators V2','port7-dense-ops-v2/summary.json','444 exact component cases including real captured activations/weights,412 production-helper cases; not full model verifier proof.'),
 ('Verify dump CPU/CUDA syntax','port7-verify-dump-syntax-v1/receipt.json','Syntax only, no link, target execution, GPU initialization, model/backend acceptance.')]:
 d=load(path);entry={'name':name,'proof_scope':scope,'receipt':ref(path),'release_accepted':False}
 if d is not None:
  entry['result']=scalar(d)
  if 'by_shape'in d:entry['by_shape']=d['by_shape']
  if name=='CPU mode2 threads80/64/96':entry['recommend_threads']=80;entry['same_work_counters']=[{k:a[k] for k in ('tokens','windows','token_rows','serial_weight_bytes','unique_weight_bytes','unique_experts')} for a in d['stages'][0]['aggregates']]
 components.append(entry)


# Completed evidence after the original v1 snapshot. These remain bound to their
# own frozen source/profile; successful selected proofs do not promote a release.
def receipt_variant(name,scope,folder,paths,status='selected_model_proof_pass'):
 v=variant(name,status,scope,folder,paths)
 proof=[load(path) for path in paths]
 v['receipt_pass']=bool(proof) and all(isinstance(d,dict) and d.get('pass') is True for d in proof)
 if not v['receipt_pass']:v['status']='receipt_missing_or_not_pass'
 return v

v6=[]
for folder in ('port7-dspark-v6-ops0-cache0-parity-v1','port7-dspark-v6-ops1-cache0-parity-v1','port7-dspark-v6-ops1-cache1-parity-v1'):
 rows=load(folder+'/native/summary.jsonl') or []
 parity=[r for r in rows if r.get('mode')=='batch_parity']
 v6.append({'folder':folder,'summary_file':ref(folder+'/native/summary.jsonl'),'parity':parity,'pass':bool(parity) and all(r.get('exact_greedy') and r.get('max_abs')==0 for r in parity)})
v=variant('V6 batched ops + masked GPU-cache verifier','selected_short_model_parity_pass' if all(a['pass'] for a in v6) else 'receipt_missing_or_not_pass','Three combined actual Engine.verify profiles,128 selected teacher-forced rows each. No longer pending; not final gate20/quality or speed acceptance.')
v['arms']=v6;v['atomic_result']=compact_compare('port7-v6-three-parity-v1/port7-v6-three-parity-v1-card-job.result.json');variants.append(v)
for folder,name,audits in [
 ('port7-dspark-v7-overlap-causal-mini-v2','V7 shared-overlap causal model proof',('strict-overlap0-audit-v1.json','strict-overlap1-audit-v1.json')),
 ('port7-dspark-p7v8-view-packed-causal-v1','P7-v8 view/packed causal model proof',tuple('strict-causal-arm'+str(i)+'-audit-v1.json' for i in (1,3,5))),
 ('port7-dspark-p7v9-msa-lookup-group-numeric-v1','P7-v9 MSA/lookup/group selected numerical proof',tuple('strict-causal-arm'+str(i)+'-audit-v1.json' for i in (1,3,5,7,9))),
 ('port7-dspark-p7v10-coalesce-causal-v1','P7-v10 coalesce causal model proof',('strict-causal-arm1-audit-v1.json','strict-causal-arm3-audit-v1.json'))]:
 variants.append(receipt_variant(name,'Frozen selected corpus3324 parity plus actual greedy/raw-ID strict provenance audits; diagnostic timing does not establish a runtime winner or final quality.',folder,tuple(folder+'/'+n for n in audits)))
for length in (16384,32768):
 folder='port7-p7v8-verify-long'+str(length)+'-v1'
 v=receipt_variant('P7-v8 selected long '+str(length),'Selected anchor+32 tail rows against GPU-offload oracle; actual PP/singleton replay, verify/partial commits/features bitwise. Capacity-equivalence retry is explicit; not final gate20 or full long-quality acceptance.',folder,(folder+'/compare.json',))
 v['capacity_equivalence']=compact_compare(folder+'/capacity-equivalence-replay-v1.json');v['replay_receipt']=compact_compare(folder+'/replay-receipt.json');variants.append(v)
variants.append(receipt_variant('Bounded capture VIEW raw lifetime','Whole4096=3840+256 split and rows0-to17, after-workspace-release lease exact; live-lease forward reject/rewind stale guards. Not HTTP cancel or PP benchmark.','port7-capture-view-raw-v9-v1',('port7-capture-view-raw-v9-v1/independent-audit.json',)))
variants.append(receipt_variant('Ordinary exact-cache290 clean mini','Same P7-v9 ELF/input/recipe128 actual greedy IDs; primer included. Numerically passes but cached ON slower; one selected native CLI run, not om acceptance.','port7-p7v9-ordinary-exactcache290-v1',('port7-p7v9-ordinary-exactcache290-v1/compare.json',),'selected_greedy_pass_speed_negative'))
rpc='port7-dspark-p7v10-coalesce-native-clean-v1'
v=variant('V10 clean native RPC mini','selected_greedy_pass_speed_negative','Four actual native RPC arms: coalesce0/1 ordinary/DSpark, raw128 IDs equal; lazy setup/seed included in predicted_ms, finish separate. Clean exclusive window, not HTTP/full om or quality.',rpc)
v['arms']=[compact_compare(rpc+'/coalesce'+str(c)+'/'+mode+'/summary.json') for c in (0,1) for mode in ('ordinary','dspark')]
v['receipt_pass']=all(a.get('pass') is True for a in v['arms']);v['atomic_result']=compact_compare(rpc+'/'+rpc+'-card-job.result.json');variants.append(v)
http='port7-dspark-http-representative-v10-v1'
v=receipt_variant('V10 representative six actual HTTP requests','Depth1/3 x actual4K/16K/32K, fixedUUID/native256IDs and actual PP recipes equal across depths. No ordinary arm or warmup; lazy setup included. No35tps winner or final GSM/tools acceptance.',http,(http+'/exploration-report.json',http+'/phase-counter-audit.json'),'selected_cross_depth_pass_speed_goal_not_reached')
d=load(http+'/exploration-report.json') or {};v['records']=d.get('records',[]);v['cross_depth_math_checks']=d.get('cross_depth_math_checks',[]);v['warmup']=d.get('warmup');variants.append(v)
aba='port7-p7v11-ordinary-imma-rank-aba-v1';d=load(aba+'/aba-comparison.json')
v=variant('P7-v11 ordinary IQ3 IMMA/rank OFF-ON-OFF','pending_atomic_result' if d is None else 'completed_receipt_needs_scope_read','Prepared clean ordinary same-ELF ABA; do not infer model gain from IQ3/GU components. Primer must remain inside decode timing.',aba,(aba+'/aba-comparison.json',),notes=('Snapshot only; queued/live result can arrive after this immutable index.',))
if isinstance(d,dict):v['recorded_comparison']=d
v['atomic_result']=compact_compare(aba+'/aba-card-job.result.json');variants.append(v)
for name,path,scope in [
 ('IQ3 exact n1 IMMA lane','port7-iq3-lane-v1/component.json','1059 raw and684 actual cases,1079336 I32/414870 F32 exact; isolated component timing, not integrated model speed.'),
 ('Integrated IQ3 plus gate/up helper GU0','port7-cached-imma-gu-component-v1/component-GU0.json','60 original-rank weighted F32 component cases, rotated physical working sets; no model TG.'),
 ('Integrated IQ3 plus gate/up helper GU1','port7-cached-imma-gu-component-v1/component-GU1.json','60 component cases, contiguous joined gate/up and gap fallback; no model TG.'),
 ('Independent IQ3/GU audit','port7-cached-imma-gu-component-v1/independent-iq3-gu-audit-v1.json','120 helper cases,2949120 weighted F32 exact; source/job/provenance audit, not Engine quality/performance.'),
 ('RMS-only fusion direct stock kernel','port7-rms-selective-v1/component.json','116 captured activation/weight component cases exact; selective backend patch/model integration proof separate.'),
 ('CPU skew baseline','port7-cpu-skew-check-v1/receipt.json','3840 token-FFN rows exact, private CPU trace; not model TG.'),
 ('CPU strict-spread80 skew','port7-cpu-skew-spread-check-v1/receipt.json','3840 rows exact;80 physical cores/no SMT duplicates. Tail not eliminated, no model TG improvement claim.'),
 ('CPU skew comparison','port7-cpu-skew-spread-comparison-v1.json','Read-only matched component comparison, not model speed or hardware DRAM measurement.'),
 ('MWAITX private wake proposal','port7-cpu-mwaitx-check-v1/receipt.json','38 arms PASS including forced fallback; wake worse than PAUSE, no power measurement or runtime default change.'),
 ('Capture VIEW host','port7-capture-view-host-v3.json','958 actual CPU ggml scatter/owner/epoch checks, not GPU math.'),
 ('Capture VIEW session','port7-capture-view-session-v3.json','9720 cases/55728 generated rows, legacy/view seed append recipe and cancellation/lazy controller proof, not model quality.')]:
 components.append({'name':name,'proof_scope':scope,'release_accepted':False,'evidence':compact_compare(path)})

builds=[]
for p in sorted((ROOT/'build-port7-candidate').glob('*/build-receipt.json')):
 d=load(p);builds.append({'name':'incremental-'+p.parent.name,'receipt':ref(p),'base_head':d['base_head'],'proof_scope':d.get('scope'),'canonical':False,'source_directory':str(p.parent/'source'),'source_hash_receipt':ref(p.parent/'source-sha256.json'),'key_sources':{k:v for k,v in d['source'].items() if k in ('m31_engine.cpp','m31_engine.hpp','m31_dspark_verify.inc','m31_dspark_verify_state.hpp','cpu_batched_verify.hpp','cpu_batched_dot.hpp','m31_gpu_experts.cpp')},'elf':{name:{'path':str(p.parent/name),'recorded_sha256':digest,'exists':(p.parent/name).exists(),'rehash_performed_by_index':False} for name,digest in d['elf'].items()},'cuda_override':d.get('cuda_override'),'cuda_header_sha256':d.get('cuda_header_sha256')})

index={'format':'m31-port7-evidence-index-v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_only':True,'scope':'Independent lightweight metadata/receipt index; no F32/weights/ELF reread, benchmark, model test or build. Artifact claims must match their own frozen source/ELF, not current workspace HEAD.','headline_status':{'v8_baseline_accepted':True,'port7_final_release_accepted':False,'generation_ge35_tps_achieved':False,'generation_ge40_tps_achieved':False,'port7_final_GSM8K100_completed':False,'port7_final_tools23_completed':False,'port7_final_quality_accepted':False,'port7_final_gate20_complete':False,'port7_final_long16K32K_complete':False,'selected_P7v8_long16K32K_oracle_diagnostics_complete':all((load('port7-p7v8-verify-long'+str(n)+'-v1/compare.json') or {}).get('pass') is True for n in (16384,32768)),'selected_V6_combined_model_parity_complete':all(a['pass'] for a in v6),'V10_representative_HTTP_completed':(load(http+'/exploration-report.json') or {}).get('pass') is True,'candidate_V11_build_complete':bool(load(ROOT/'build-port7-candidate/v11/build-receipt.json')),'port7_final_two_clean_om_rounds_complete':False,'port7_canonical_CPU_CUDA_complete':False,'full_model_backend_portability_verified':False},'baseline_v8':base,'variants':variants,'components':components,'incremental_builds':builds,'pending_final_gates':[{'gate':'Select winning runtime profile and freeze source/support','status':'PENDING'},{'gate':'Canonical CPU then CUDA full CMake build and final-ELF fixtures','status':'PENDING','script':str(ROOT/'tools/m31/port7_canonical.py'),'note':'Prepared canonical process; candidate V11 incremental CPU/CUDA build is recorded separately and is not a final canonical/quality acceptance.'},{'gate':'GPU-offload PR#31192 oracle gate20 through actual Engine.verify/rollback/features/head1','status':'PENDING','dumper':str(ROOT/'tools/m31/port7_verify_dump.cpp'),'criteria':{'top1_min':.99,'mean_kl_max':.001,'max_kl_position_max':.05},'note':'CPU oracle is debug-only; selected short parity cannot replace final20.'},{'gate':'Final canonical/profile long-context16K/32K checks','status':'PENDING_FINAL_PROFILE','selected_P7v8_diagnostics':'COMPLETED_PASS, indexed separately; do not transfer frozen profiles'},{'gate':'Two clean unchanged om_bench4K/16K/32K repetitions; PP preserved vs v8; TG35–40+','status':'PENDING','goal_achieved':False},{'gate':'GSM8K100>=95 and tools23>=19 on final port7 ELF/profile','status':'PENDING','completed':False,'note':'v8 scores97/100 and19/23 do not transfer to port7.'},{'gate':'Final winner native HTTP/SSE EOS/stop/cancel→next/tools/prefix/fallback smoke','status':'PENDING','note':'Selected V5 smoke exists; rerun/attribute to final canonical ELF.'}],'portability_limits':['CPU dot GCC/Clang/native CPUID and minimal nonnative stubs are component/static guard evidence, not full MSVC or AArch64 SDK/model builds.','CUDA graph/MMV/operator components and CPU/CUDA dumper syntax are not complete canonical CPU/CUDA model backend acceptance.','HIP/SYCL model builds not tested.','64K cache admission is static metadata planning; no real64K init/forward acceptance claimed.'],'interpretation_limits':['card-job rc0 means commands completed, not universal numerical/speed/quality acceptance.','Short greedy equality/teacher forcing is prompt-specific; full gate20 and final quality remain separate.','V3 fenced fusion short ON/OFF passed; V4 UNFENCED fusion failed. Do not conflate modes.','Dense read_ms includes queued GPU execution and wait; compute_ms is mostly host enqueue. Stage categories overlap.','DSpark sweep diagnostic TPS cannot replace clean two-round HTTP om benchmarks, especially when builds overlap.','Logical expert bytes/seconds are not measured DRAM traffic or PCIe counters.','Large native RPC result linked via independent small audit, not re-read by this index.'],'source_receipts_read':seen}
with OUT.open('x') as stream:stream.write(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out':str(OUT),'bytes':OUT.stat().st_size,'variants':len(variants),'components':len(components),'builds':len(builds),'small_metadata_files_read':len(seen),'raw_F32_read':False,'goal35_40_achieved':False,'final_quality_completed':False}))
