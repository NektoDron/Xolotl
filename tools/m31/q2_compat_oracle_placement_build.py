#!/usr/bin/env python3
"""Build bounded opt-in e3/e4 placement overlay against frozen PR archives."""
import argparse,datetime,hashlib,json,pathlib,shutil,subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory',type=pathlib.Path,required=True)
parser.add_argument('--base',type=pathlib.Path,default=pathlib.Path('/mnt/ai/Xolotl-m31/build-port6-bench-oracle/replay-v3'))
args=parser.parse_args()
base=args.base.resolve(strict=True);out=args.directory.resolve();out.mkdir(parents=True,exist_ok=False)
source=out/'source';shutil.copytree(base/'source',source)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((base/'build-receipt.json').read_text())
for path,digest in original['source'].items():
 if sha(pathlib.Path(path))!=digest:raise ValueError('base oracle source changed '+path)
p=source/'oracle/m31_oracle.cpp';s=p.read_text();old='''        llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps\\\\..*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
        };
        if (args.backend == "cuda") mp.tensor_buft_overrides = overrides;'''
new=r'''        // Bounded opt-in diagnostic of resident expert placement. Defaults
        // preserve the established all-CPU storage + GPU op-offload oracle.
        const char * placement_env = std::getenv("M31_ORACLE_GPU_EXPERT_LAYERS");
        const std::string placement = placement_env ? placement_env : "";
        std::string placement_pattern;
        std::vector<llama_model_tensor_buft_override> overrides;
        if (!placement.empty()) {
            if (args.backend != "cuda" || !args.oracle_op_offload ||
                    (placement != "10,11,12" && placement != "10,11,12,13"))
                throw std::runtime_error("placement overlay permits CUDA op-offload with e3/e4 only");
            placement_pattern = placement == "10,11,12" ?
                "blk\\.(10|11|12)\\.ffn_.*_exps\\..*" :
                "blk\\.(10|11|12|13)\\.ffn_.*_exps\\..*";
            const auto cuda_device = ggml_backend_dev_by_name("CUDA0");
            if (!cuda_device) throw std::runtime_error("placement overlay cannot find CUDA0");
            overrides.push_back({placement_pattern.c_str(), ggml_backend_dev_buffer_type(cuda_device)});
        }
        overrides.push_back({".*ffn_.*_exps\\..*", ggml_backend_cpu_buffer_type()});
        overrides.push_back({nullptr, nullptr});
        if (args.backend == "cuda") mp.tensor_buft_overrides = overrides.data();
        std::fprintf(stderr, "M31 oracle resident placement: layers=%s pattern=%s op_offload=%d\n",
                     placement.empty() ? "none" : placement.c_str(),
                     placement_pattern.c_str(), int(args.oracle_op_offload));
        {
            std::ofstream placement_receipt(run::fs::path(args.out) / "oracle-placement.json");
            placement_receipt << "{\"gpu_expert_layers\":" << run::quote(placement)
                << ",\"pattern\":" << run::quote(placement_pattern)
                << ",\"oracle_op_offload\":" << (args.oracle_op_offload ? "true" : "false")
                << ",\"context_requested\":" << args.context
                << ",\"kv\":\"f16\",\"scope\":\"Separate resident-placement numeric diagnostic, not the Q8 HTTP server contract.\"}\n";
            if (!placement_receipt) throw std::runtime_error("cannot write placement receipt");
        }'''
if s.count(old)!=1:raise ValueError('base override block not uniquely found')
p.chmod(0o644)
p.write_text(s.replace(old,new))
argv=['cmake','-S',str(source/'oracle'),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14','-DCUDAToolkit_ROOT=/usr/local/cuda','-DM31_LLAMA_SOURCE=/mnt/ai/llama.cpp-m31pr-1026','-DM31_LLAMA_BUILD=/mnt/ai/llama.cpp-m31pr-1026/build']
subprocess.run(argv,check=True);subprocess.run(['cmake','--build',str(out/'build'),'--target','m31_oracle','-j2'],check=True)
receipt={'format':'m31-q2-placement-oracle-overlay-v1','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pr_sha':original['pr_sha'],'base_oracle_build_receipt':{'path':str(base/'build-receipt.json'),'sha256':sha(base/'build-receipt.json')},'source':{str(path):sha(path) for path in sorted(source.rglob('*')) if path.is_file()},'elf':{'m31_oracle':sha(out/'build/m31_oracle')},'configure_argv':argv,'scope':'Only optional e3/e4 expert storage override plus sidecar metadata; unchanged default all-CPU expert storage. Reuses completed read-only PR archives; canonical Xolotl and protected V8 unchanged.'}
with(out/'build-receipt.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
