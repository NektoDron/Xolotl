#!/usr/bin/env python3
"""Coordinator-only frozenV8 decode flat cycles; no global perf/THP changes."""
import argparse,datetime,hashlib,json,os,pathlib,queue,re,select,signal,subprocess,threading,time,uuid
P=pathlib.Path;R=P('/mnt/ai/Xolotl-m31');L=P('/mnt/ai/logs/m31-port')
p=argparse.ArgumentParser();p.add_argument('--out',type=P,required=True);a=p.parse_args();out=a.out.resolve()
assert os.environ.get('M31_CARD_SESSION')=='1','aictl/card runner required'
assert not out.exists(),'new output directory required'
out.mkdir(parents=True);receipt={'format':'m31-n1-flat-perf-v1','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Flat99Hz cycles:u on owned frozenV8 Engine PID, marker-gated n1 decode. Not model quality/om or off-CPU wait profile.','pass':False,'callgraph':False,'inherit_default_enabled':True,'global_sysctl_changes':False,'capabilities':['CAP_PERFMON'],'stages':[]}
def sha(path):
 h=hashlib.sha256()
 with P(path).open('rb')as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
profile=json.loads((L/'port7-v8-baseline-v1/manifest.json').read_text());frozen=R/'build-port6-canonical/v8';engine=frozen/'m31_generate';build=json.loads((frozen/'build-receipt.json').read_text());assert sha(engine)==build['elf']['m31_generate'],'frozenV8 ELF drift'
prompt=L/'port7-dspark-matched-serial-numeric-v1/cases/corpus_3324.i32';tail=L/'port7-dspark-matched-serial-numeric-v1/native/corpus_3324.generated.i32';raw=prompt.read_bytes();generated=tail.read_bytes();assert len(raw)==3324*4 and len(generated)==128*4
ids=out/'corpus3324plus128.i32';ids.write_bytes(raw+generated)
env={k:v for k,v in os.environ.items()if not k.startswith(('M31_','GGML_CUDA_'))};env.update(profile['environment']);env['M31_CARD_SESSION']='1'
argv=[str(engine),'-m','/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf','--ids-file',str(ids),'--out',str(out/'native'),'--backend','cuda','--ctx','40000','--chunk','32768','--threads','80','--arena-gib','170','--gpu-experts-gib','7','--gpu-experts-min-tokens','32','--reuse-allocations','1','--persistent-cpu-pool','1','--cpu-residual','1','--cpu-decode-cache','1','--dense-decode-cache','1','--profile','1','--decode-tail','128','--head-tail','1','--rows','last:128']
receipt|={'native_argv':argv,'environment':profile['environment'],'engine_sha256':build['elf']['m31_generate'],'engine_receipt':str(frozen/'build-receipt.json'),'prompt_sha256':sha(prompt),'continuation_sha256':sha(tail),'perf_event_paranoid':P('/proc/sys/kernel/perf_event_paranoid').read_text().strip()}
ctl=out/'perf-control.fifo';ack=out/'perf-ack.fifo';os.mkfifo(ctl,0o600);os.mkfifo(ack,0o600);ctl_fd=os.open(ctl,os.O_RDWR|os.O_NONBLOCK);ack_fd=os.open(ack,os.O_RDWR|os.O_NONBLOCK)
child=None;reader=None;collector=None;events=queue.Queue();state={'last_n1':None};enabled=False;collector_stopped=False

def control(command):
 os.write(ctl_fd,(command+'\n').encode());deadline=time.monotonic()+3
 while time.monotonic()<deadline:
  if select.select([ack_fd],[],[],max(0,deadline-time.monotonic()))[0]:
   text=os.read(ack_fd,4096).decode();receipt['stages'].append({'event':'perf_'+command+'_ack','monotonic':time.monotonic(),'ack':text.strip(),'last_observed_native_n1':state['last_n1']});return
 raise RuntimeError('perf control acknowledgement absent; inspect collector.log (CAP_PERFMON/PMU may be unavailable)')

def stop_collector():
 global collector_stopped
 if collector and not collector_stopped:
  r=subprocess.run(['sudo','systemctl','kill','--kill-who=all','--signal=SIGINT',collector],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);receipt['stages'].append({'event':'stop_owned_perf_collector','rc':r.returncode,'output':r.stdout});collector_stopped=True
try:
 with(out/'native.stdout').open('wb')as stdout:
  child=subprocess.Popen(argv,env=env,stdout=stdout,stderr=subprocess.PIPE,text=True,bufsize=1)
  own=P('/proc')/str(child.pid);assert own.stat().st_uid==os.getuid()and P(os.readlink(own/'exe')).resolve()==engine.resolve(),'unexpected target PID/exe/owner'
  receipt['native_pid']=child.pid
  def receive():
   with(out/'native.stderr').open('w')as log:
    for line in child.stderr:
     log.write(line);log.flush();match=re.match(r'M31 forward pos=(\d+) n=1\b',line)
     if match:state['last_n1']=int(match[1]);events.put(('n1',int(match[1]),time.monotonic()))
    events.put(('eof',None,time.monotonic()))
  reader=threading.Thread(target=receive,daemon=True);reader.start()
  deadline=time.monotonic()+1800
  while True:
   kind,position,stamp=events.get(timeout=max(.1,deadline-time.monotonic()))
   if kind=='eof':break
   if position>=3324 and collector is None:
    collector='codex-m31-port7-cpu-flat-perf-'+str(child.pid)+'-'+uuid.uuid4().hex[:8];receipt['target_tids_at_attach']=sorted(int(t.name)for t in(own/'task').iterdir())
    command=['sudo','systemd-run','--unit='+collector,'-p','User=nekto','-p','AmbientCapabilities=CAP_PERFMON','-p','CapabilityBoundingSet=CAP_PERFMON','-p','KillMode=mixed','-p','TimeoutStopSec=600','-p','RuntimeMaxSec=15','-p','StandardOutput=append:'+str(out/'collector.log'),'-p','StandardError=append:'+str(out/'collector.log'),'perf','record','-e','cycles:u','-F','99','-T','--sample-cpu','--delay=-1','--control','fifo:'+str(ctl)+','+str(ack),'-p',str(child.pid),'-o',str(out/'cycles.data'),'--','sleep','10']
    receipt['collector_argv']=command;receipt['stages'].append({'event':'decode_gate_trigger','position':position,'monotonic':stamp})
    # Revalidate immediately before privileged attach after the long init. The
    # Popen child is not reaped/recycled between this check and perf attach.
    assert child.poll() is None, 'native child exited before perf attach'
    assert own.stat().st_uid==os.getuid() and P(os.readlink(own/'exe')).resolve()==engine.resolve(), 'perf attach target identity changed'
    receipt['stages'].append({'event':'pre_attach_owned_PID_UID_exe_guard_pass','pid':child.pid,'uid':os.getuid(),'exe':str(engine.resolve()),'monotonic':time.monotonic()})
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);receipt['collector_start']={'rc':result.returncode,'output':result.stdout};assert result.returncode==0,'collector unit start failed'
    control('enable');enabled=True
   if position>=3451 and enabled:control('disable');enabled=False;stop_collector()
  receipt['native_rc']=child.wait(timeout=60);reader.join(timeout=10)
 if enabled:control('disable');enabled=False
 stop_collector()
 for _ in range(100):
  status=subprocess.run(['systemctl','show',collector or 'nonexistent','-p','ActiveState','--value'],stdout=subprocess.PIPE,text=True).stdout.strip()
  if status not in ('active','activating','deactivating'):break
  time.sleep(.1)
 else:raise RuntimeError('owned collector did not exit after SIGINT')
 receipt['collector_unit_result']=subprocess.run(['systemctl','show',collector or 'nonexistent','-p','ActiveState','-p','Result','-p','ExecMainStatus'],stdout=subprocess.PIPE,text=True).stdout
 data=out/'cycles.data';assert receipt['native_rc']==0 and data.exists()and data.stat().st_size>0,'native/perf output missing'
 for name,cmd in [('report',['perf','report','--stdio','--no-children','--sort','symbol,dso','--percent-limit','0.1','-i',str(data)]),('script',['perf','script','-i',str(data),'-F','time,pid,tid,cpu,event,ip,sym,dso'])]:
  with(out/(name+'.txt')).open('wb')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT);receipt['stages'].append({'event':name,'rc':r.returncode});assert r.returncode==0,name+' failed'
 sample_lines=(out/'script.txt').read_text().splitlines();receipt['sample_line_count']=sum(bool(line.strip())and not line.startswith('#')for line in sample_lines);assert receipt['sample_line_count']>0,'no CPU samples'
 receipt['perf_data_sha256']=sha(data);receipt['pass']=True
except BaseException as e:receipt['error']=repr(e)
finally:
 if collector:stop_collector()
 if child and child.poll()is None:
  # The collector cannot mutate the model. Let the card-owned native job finish;
  # only the outer card-runner timeout may terminate an unresponsive model.
  try:receipt['native_rc_after_diagnostic_failure']=child.wait(timeout=60)
  except subprocess.TimeoutExpired:receipt['native_still_running_after60s']=True
 os.close(ctl_fd);os.close(ack_fd);receipt['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pass':receipt['pass'],'out':str(out),'error':receipt.get('error'),'samples':receipt.get('sample_line_count')}));raise SystemExit(0 if receipt['pass']else 1)
