// One model initialization: native full-row history gates and exact KV restore.
// This process never reserves/releases a card. Run only from the owned launcher.
#include "m31_engine.hpp"
#include "m31_mixed.hpp"
#include "m31_run.hpp"
#include "json.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

namespace {
using namespace strata::m31;
using J = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
double milliseconds(Clock::time_point from) {
    return std::chrono::duration<double,std::milli>(Clock::now()-from).count();
}
void save(const fs::path & path,const J & value) {
    std::ofstream file(path);file<<value.dump(2)<<'\n';
    require(bool(file),"cannot write "+path.string());
}
bool bytes_equal(const fs::path & a,const fs::path & b) {
    const auto size=fs::file_size(a);
    if(size!=fs::file_size(b))return false;
    std::ifstream af(a,std::ios::binary),bf(b,std::ios::binary);
    std::vector<char> av(1024*1024),bv(av.size());
    for(uint64_t offset=0;offset<size;) {
        const size_t count=size_t(std::min<uint64_t>(av.size(),size-offset));
        af.read(av.data(),std::streamsize(count));bf.read(bv.data(),std::streamsize(count));
        require(bool(af)&&bool(bf),"short logits read");
        if(std::memcmp(av.data(),bv.data(),count))return false;
        offset+=count;
    }
    return true;
}
J check_oracle(const fs::path & oracle,const run::Args & args,const std::vector<int32_t> & tokens,int vocab) {
    std::ifstream input(oracle/"manifest.json");J ref;input>>ref;
    require(ref.at("format")=="m31-logits-v1"&&ref.at("engine")=="llama.cpp-PR31192"&&
            ref.at("backend")=="cuda"&&ref.at("oracle_op_offload")==true&&
            ref.at("kv")=="f16"&&ref.at("full_last_layer_before_crop")==true,
            "requires accepted PR GPU-op-offload F16/full-layer oracle");
    require(ref.at("model")==fs::canonical(args.model).string()&&ref.at("tokens")==tokens.size()&&
            ref.at("vocab")==vocab&&ref.value("head_tail",0)==args.head_tail,
            "oracle model/dimensions/head geometry differs");
    const auto positions=ref.at("positions").get<std::vector<int>>();
    require(positions==run::selected_rows("all",int(tokens.size())),"oracle is not all-row");
    const auto actual=ref.at("actual_chunks");
    require(actual.size()==1&&actual[0].at("position")==0&&actual[0].at("tokens")==tokens.size(),
            "oracle was not one complete prefill");
    const auto ids=run::read_ids(oracle/"tokens.i32");
    require(ids==tokens,"oracle tokens/visual placeholders differ");
    require(fs::file_size(oracle/"logits.f32")==tokens.size()*size_t(vocab)*sizeof(float),
            "oracle logits are truncated");
    return ref;
}
void full_prefill(Engine & engine,const MixedInput & mixed,const std::vector<int32_t> & tokens,
                  bool visual,const run::Args & args,const fs::path & destination) {
    std::string error;
    engine.reset();
    const int n=int(tokens.size());
    require(engine.position()==0,"reset did not clear position");
    const int planned=engine.prefill_chunk(n,error);
    require(planned==n,"gate requires a single complete prefill; planned="+std::to_string(planned)+" "+error);
    const bool forwarded=visual
        ?engine.forward_mixed(tokens.data(),mixed.visual_rows.data(),mixed.embeddings.data(),mixed.visual_count,n,error)
        :engine.forward(tokens.data(),n,error);
    require(forwarded,"prefill failed: "+error);
    require(engine.position()==n,"prefill position differs");
    std::vector<float> logits;
    require(engine.logits(0,n,logits,error),"all-row head failed: "+error);
    require(logits.size()==size_t(n)*size_t(engine.geometry().vocab),"all-row logits size differs");
    run::Dump dump(destination,args,tokens,int(engine.geometry().vocab));
    dump.record_chunk(0,n);
    for(int row=0;row<n;++row)dump.row(row,logits.data()+size_t(row)*size_t(engine.geometry().vocab));
    dump.finish("xolotl-m31-correctness",n);
}
J checkpoint_gate(Engine & engine,const std::vector<int32_t> & continuation) {
    std::string error;
    const auto prefix=engine.position();
    Engine::PrefixCheckpointPtr checkpoint;
    const auto expected_bytes=engine.prefix_checkpoint_bytes();
    const auto snapshot_start=Clock::now();
    require(engine.checkpoint_prefix(1024ull*1024*1024,checkpoint,error),"checkpoint failed: "+error);
    const double snapshot_ms=milliseconds(snapshot_start);
    require(checkpoint&&checkpoint->position()==prefix&&checkpoint->bytes()==expected_bytes&&expected_bytes,
            "checkpoint dimensions differ");
    auto append=[&]() {
        std::vector<std::vector<float>> result;
        for(const int32_t token:continuation) {
            require(engine.forward(&token,1,error),"continuation forward failed: "+error);
            result.emplace_back();
            require(engine.logits(0,1,result.back(),error),"continuation logits failed: "+error);
            require(result.back().size()==size_t(engine.geometry().vocab),"continuation logits size differs");
            for(float value:result.back())require(std::isfinite(value),"nonfinite continuation logits");
        }
        return result;
    };
    const auto fresh=append();
    require(engine.position()==prefix+int64_t(continuation.size()),"fresh continuation position differs");
    const auto restore_start=Clock::now();
    require(engine.restore_prefix(checkpoint,error),"restore failed: "+error);
    const double restore_ms=milliseconds(restore_start);
    require(engine.position()==prefix,"restore position differs");
    std::vector<float> stale;
    require(!engine.logits(0,1,stale,error),"restore exposed stale last-chunk logits");
    const auto replay=append();
    require(engine.position()==prefix+int64_t(continuation.size()),"replayed continuation position differs");
    for(size_t step=0;step<fresh.size();++step)
        require(fresh[step].size()==replay[step].size()&&
                !std::memcmp(fresh[step].data(),replay[step].data(),fresh[step].size()*sizeof(float)),
                "restored continuation differs at step "+std::to_string(step));
    return J{{"pass",true},{"prefix_position",prefix},{"prefix_bytes",expected_bytes},
             {"continuation_tokens",continuation},{"steps",continuation.size()},
             {"vocab_values_compared",continuation.size()*size_t(engine.geometry().vocab)},
             {"snapshot_ms",snapshot_ms},{"restore_ms",restore_ms},
             {"timing_scope","correctness diagnostic, not speed benchmark"}};
}
} // namespace

int main(int argc,char ** argv) {
    using namespace strata::m31;
    ggml_backend_t gpu=nullptr;
    fs::path output;
    J result={{"format","m31-ttft-native-gate-v1"},{"pass",false},{"cases",J::array()}};
    try {
        fs::path plan_path;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"usage: m31_ttft_gate --plan PLAN.json --out NEW_DIR");
            const std::string key=argv[i];
            if(key=="--plan")plan_path=argv[i+1];
            else if(key=="--out")output=argv[i+1];
            else throw std::runtime_error("unknown argument "+key);
        }
        require(!plan_path.empty()&&!output.empty(),"usage: m31_ttft_gate --plan PLAN.json --out NEW_DIR");
        require(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1"&&
                std::getenv("CUDA_VISIBLE_DEVICES")&&std::string(std::getenv("CUDA_VISIBLE_DEVICES"))=="0",
                "requires an aictl-owned GPU0 card session");
        for(const char * flag:{"M31_GPU_PREFETCH_HISTORY","M31_GPU_PREFETCH_NEXT","M31_GPU_PREFILL_CHAIN","GGML_CUDA_DISABLE_FUSION"})
            require(std::getenv(flag)&&std::string(std::getenv(flag))=="1",std::string("gate requires ")+flag+"=1");
        for(const char * flag:{"M31_GPU_EXACT","M31_GPU_CPU_QUANTS","M31_GPU_FAST_FFN"})
            require(!std::getenv(flag)||std::string(std::getenv(flag))=="0",std::string("gate requires native math; disable ")+flag);
        result["scope"]="fresh/repeat clear KV each time; history persists across all cases; all rows vs accepted PR oracle";
        result["engine_initializations"]=1;
        std::ifstream plan_file(plan_path);J plan;plan_file>>plan;
        require(plan.at("format")=="m31-ttft-gate-plan-v1"&&plan.at("cases").is_array()&&plan.at("cases").size()==3,
                "plan must specify exactly image/video/code128");
        std::set<std::string> names;
        for(const auto & item:plan.at("cases"))names.insert(item.at("name").get<std::string>());
        require(names==std::set<std::string>{"image","video","code128"},"plan must contain image/video/code128 once");
        require(fs::create_directory(output),"output directory already exists");
        result["plan"]=fs::canonical(plan_path).string();
        result["configuration"]={{"context",4096},{"chunk",4096},{"threads",80},{"arena_gib",170},
                                  {"gpu_experts_gib",7},{"history",true}};
#ifdef M31_HAVE_CUDA
        gpu=ggml_backend_cuda_init(0);
        require(gpu!=nullptr,"GPU0 initialization failed");
#else
        throw std::runtime_error("built without CUDA");
#endif
        {
            Config cfg;cfg.backend=gpu;cfg.context=4096;cfg.max_chunk=4096;cfg.threads=80;
            cfg.arena_gib=170;cfg.gpu_experts_gib=7;cfg.gpu_experts_min_tokens=32;
            cfg.reuse_allocations=true;cfg.persistent_cpu_pool=true;cfg.cpu_residual=true;
            cfg.cpu_decode_cache=true;cfg.dense_decode_cache=true;cfg.profile=true;
            Engine engine;std::string error;
            const auto model=plan.at("model").get<std::string>();
            require(engine.init(model,cfg,error),"engine initialization failed: "+error);
            const auto continuation=plan.value("continuation_tokens",std::vector<int32_t>{7,11});
            require(continuation.size()==2,"checkpoint gate requires exactly two continuation tokens");
            for(int token:continuation)require(token>=0&&token<engine.geometry().vocab,"invalid continuation token");
            for(const auto & item:plan.at("cases")) {
                const auto name=item.at("name").get<std::string>();
                const bool visual=item.contains("mixed_manifest");
                MixedInput mixed;
                std::vector<int32_t> tokens;
                run::Args args;args.model=model;args.backend="cuda";args.context=4096;args.chunk=4096;
                args.threads=80;args.arena_gib=170;args.gpu_experts_gib=7;args.gpu_experts_min_tokens=32;
                args.reuse_allocations=true;args.persistent_cpu_pool=true;args.cpu_residual=true;
                args.cpu_decode_cache=true;args.dense_decode_cache=true;args.profile=true;args.rows="all";args.taps="none";
                args.head_tail=visual?4096:0;
                if(visual) {
                    args.mixed_manifest=item.at("mixed_manifest").get<std::string>();
                    mixed=load_mixed_manifest(args.mixed_manifest,int(engine.geometry().vocab),cfg.context);
                    tokens=mixed.tokens;
                } else tokens=run::read_ids(item.at("ids_file").get<std::string>());
                require(!tokens.empty()&&tokens.size()+continuation.size()<=size_t(cfg.context),"case exceeds context");
                const fs::path oracle=item.at("oracle").get<std::string>();
                check_oracle(oracle,args,tokens,int(engine.geometry().vocab));
                result["cases"].push_back(J{{"name",name},{"rows",tokens.size()},
                                          {"oracle",fs::canonical(oracle).string()},{"pass",false}});
                auto & current=result["cases"].back();
                for(const char * variant:{"fresh","repeat"}) {
                    const fs::path directory=output/(name+"-"+variant);
                    full_prefill(engine,mixed,tokens,visual,args,directory);
                    const bool equal=bytes_equal(directory/"logits.f32",oracle/"logits.f32");
                    current[variant]={{"directory",directory.string()},{"all_rows_bit_identical",equal}};
                    require(equal,name+" "+variant+" logits differ from PR oracle");
                    if(std::string(variant)=="fresh") {
                        current["checkpoint"]=checkpoint_gate(engine,continuation);
                        save(output/(name+"-checkpoint.json"),current["checkpoint"]);
                    }
                }
                current["pass"]=true;
                save(output/"summary.json",result);
            }
        }
        ggml_backend_free(gpu);gpu=nullptr;
        result["pass"]=true;save(output/"summary.json",result);
        std::cout<<result.dump()<<'\n';return 0;
    } catch(const std::exception & error) {
        result["error"]=error.what();
        if(!output.empty()&&fs::is_directory(output))save(output/"summary.json",result);
        if(gpu)ggml_backend_free(gpu);
        std::cerr<<error.what()<<'\n';return 1;
    }
}
