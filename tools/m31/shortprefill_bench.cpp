// Native adaptive-prefill sweep. Card ownership belongs to vision_card_launch.py.
// No logits dumps: correctness reads the accepted oracle directly in memory.
#include "m31_engine.hpp"
#include "m31_mixed.hpp"
#include "m31_run.hpp"
#include "json.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

namespace {
using namespace strata::m31;
using J=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace fs=std::filesystem;
void check(bool ok,const std::string & why){if(!ok)throw std::runtime_error(why);}
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void save(const fs::path & path,const J & value){std::ofstream f(path);f<<value.dump(2)<<'\n';check(bool(f),"write failed: "+path.string());}
struct Environment {
    std::map<std::string,std::pair<bool,std::string>> original;
    explicit Environment(const J & arms){
        for(const char * key:{"M31_SHORT_PREFILL_MODE","M31_SHORT_PREFILL_CPU_MAX_N"}){
            const char * old=std::getenv(key);original[key]={old!=nullptr,old?old:""};
        }
        for(const auto & arm:arms)for(auto it=arm.at("env").begin();it!=arm.at("env").end();++it){
            check(it.key().rfind("M31_SHORT_",0)==0,"arms may change only M31_SHORT_* variables");
            const char * old=std::getenv(it.key().c_str());original[it.key()]={old!=nullptr,old?old:""};
        }
    }
    void restore(){for(const auto & item:original)if(item.second.first)setenv(item.first.c_str(),item.second.second.c_str(),1);else unsetenv(item.first.c_str());}
    void apply(const J & arm){restore();for(auto it=arm.at("env").begin();it!=arm.at("env").end();++it){check(it.value().is_string(),"env value must be string");setenv(it.key().c_str(),it.value().get<std::string>().c_str(),1);}}
    ~Environment(){restore();}
};
struct Input {
    std::vector<int32_t> ids,prefix;
    MixedInput mixed;
    bool visual=false;
};
Input load(const J & item,int vocab,int context){
    Input input;input.visual=item.contains("mixed_manifest");
    check(input.visual!=item.contains("ids_file"),"case needs exactly ids_file or mixed_manifest");
    if(input.visual){input.mixed=load_mixed_manifest(item.at("mixed_manifest"),vocab,context);input.ids=input.mixed.tokens;}
    else input.ids=run::read_ids(item.at("ids_file").get<std::string>());
    if(item.contains("take")){const int n=item.at("take");check(n>0&&size_t(n)<=input.ids.size(),"invalid take");input.ids.resize(size_t(n));if(input.visual)input.mixed.visual_rows.resize(size_t(n));}
    if(item.contains("prefix_ids_file"))input.prefix=run::read_ids(item.at("prefix_ids_file").get<std::string>());
    if(item.contains("prefix_take")){const int n=item.at("prefix_take");check(n>=0&&size_t(n)<=input.prefix.size(),"invalid prefix_take");input.prefix.resize(size_t(n));}
    check(input.prefix.size()+input.ids.size()<=size_t(context),"case exceeds context");
    return input;
}
J difference(const Engine::PrefillStats & a,const Engine::PrefillStats & b){
    return {{"gpu_upload_bytes",b.gpu_upload_bytes-a.gpu_upload_bytes},
        {"gpu_upload_experts",b.gpu_upload_experts-a.gpu_upload_experts},
        {"gpu_cache_hits",b.gpu_cache_hits-a.gpu_cache_hits},
        {"gpu_resident_slots_before",a.gpu_resident_slots},{"gpu_resident_slots_after",b.gpu_resident_slots},
        {"cpu_weight_demand_bytes",b.cpu_weight_demand_bytes-a.cpu_weight_demand_bytes},
        {"cpu_assignments",b.cpu_assignments-a.cpu_assignments},{"cpu_layers",b.cpu_layers-a.cpu_layers},
        {"cpu_lane_ms",1000*(b.cpu_lane_seconds-a.cpu_lane_seconds)},
        {"cpu_setup_ms",1000*(b.cpu_setup_seconds-a.cpu_setup_seconds)},
        {"cpu_compute_ms",1000*(b.cpu_compute_seconds-a.cpu_compute_seconds)},
        {"cpu_reduce_ms",1000*(b.cpu_reduce_seconds-a.cpu_reduce_seconds)},
        {"cpu_reduce_scope","Nested within cpu_compute_ms; do not add twice."},
        {"cpu_correction_ms",1000*(b.cpu_correction_seconds-a.cpu_correction_seconds)},
        {"cpu_correction_groups",b.cpu_correction_groups-a.cpu_correction_groups},
        {"cpu_correction_bytes_after",b.cpu_correction_bytes},
        {"cpu_correction_scope","Nested within cpu_compute_ms; do not add twice."},
        {"cpu_gpu_quant_calls",b.cpu_gpu_quant_calls-a.cpu_gpu_quant_calls},
        {"cpu_gpu_quant_h2d_bytes",b.cpu_gpu_quant_h2d_bytes-a.cpu_gpu_quant_h2d_bytes},
        {"cpu_gpu_quant_d2h_bytes",b.cpu_gpu_quant_d2h_bytes-a.cpu_gpu_quant_d2h_bytes},
        {"cpu_gpu_quant_ms",1000*(b.cpu_gpu_quant_seconds-a.cpu_gpu_quant_seconds)},
        {"cpu_gpu_quant_scope","Nested within cpu_compute_ms; do not add twice."},
        {"cpu_gpu_quant_reserve_calls",b.cpu_gpu_quant_reserve_calls-a.cpu_gpu_quant_reserve_calls},
        {"cpu_gpu_quant_allocations",b.cpu_gpu_quant_allocations-a.cpu_gpu_quant_allocations},
        {"cpu_gpu_quant_runtime_allocations",b.cpu_gpu_quant_runtime_allocations-a.cpu_gpu_quant_runtime_allocations},
        {"cpu_gpu_quant_reserve_ms",1000*(b.cpu_gpu_quant_reserve_seconds-a.cpu_gpu_quant_reserve_seconds)},
        {"cpu_gpu_quant_reserve_scope","Engine preparation outside cpu_compute_ms; included in forward_ms."},
        {"cpu_gpu_quant_host_copy_ms",1000*(b.cpu_gpu_quant_host_copy_seconds-a.cpu_gpu_quant_host_copy_seconds)},
        {"cpu_gpu_quant_submit_ms",1000*(b.cpu_gpu_quant_submit_seconds-a.cpu_gpu_quant_submit_seconds)},
        {"cpu_gpu_quant_wait_ms",1000*(b.cpu_gpu_quant_wait_seconds-a.cpu_gpu_quant_wait_seconds)},
        {"cpu_gpu_quant_unpack_ms",1000*(b.cpu_gpu_quant_unpack_seconds-a.cpu_gpu_quant_unpack_seconds)},
        {"cpu_gpu_quant_stages_scope","host_copy/submit/wait/unpack nested within cpu_gpu_quant_ms and cpu_compute_ms."},
        {"cpu_gpu_quant_device_bytes_before",a.cpu_gpu_quant_device_bytes},
        {"cpu_gpu_quant_device_bytes_after",b.cpu_gpu_quant_device_bytes},
        {"cpu_gpu_quant_pinned_bytes_before",a.cpu_gpu_quant_pinned_bytes},
        {"cpu_gpu_quant_pinned_bytes_after",b.cpu_gpu_quant_pinned_bytes},
        {"cpu_gpu_quant_reserved_max_values_before",a.cpu_gpu_quant_reserved_max_values},
        {"cpu_gpu_quant_reserved_max_values_after",b.cpu_gpu_quant_reserved_max_values},
        {"cpu_gpu_quant_reserved_max_rows_before",a.cpu_gpu_quant_reserved_max_rows},
        {"cpu_gpu_quant_reserved_max_rows_after",b.cpu_gpu_quant_reserved_max_rows},
        {"cpu_gpu_quant_stream_priority_before",a.cpu_gpu_quant_stream_priority},
        {"cpu_gpu_quant_stream_priority_after",b.cpu_gpu_quant_stream_priority},
        {"cpu_cuda_fallback_layers",b.cpu_cuda_fallback_layers-a.cpu_cuda_fallback_layers},
        {"gpu_execute_ms",1000*(b.gpu_execute_seconds-a.gpu_execute_seconds)},
        {"short_hybrid_forwards",b.short_hybrid_forwards-a.short_hybrid_forwards},
        {"short_cpu_experts",b.short_cpu_experts-a.short_cpu_experts},
        {"short_gpu_experts",b.short_gpu_experts-a.short_gpu_experts},
        {"short_cached_experts",b.short_cached_experts-a.short_cached_experts},
        {"short_cpu_routes",b.short_cpu_routes-a.short_cpu_routes},
        {"short_gpu_routes",b.short_gpu_routes-a.short_gpu_routes},
        {"short_cold_gpu_bytes",b.short_cold_gpu_bytes-a.short_cold_gpu_bytes},
        {"short_cpu_ms",1000*(b.short_cpu_seconds-a.short_cpu_seconds)},
        {"short_wall_ms",1000*(b.short_wall_seconds-a.short_wall_seconds)},
        {"cpu_executor_graph_builds",b.cpu_executor_graph_builds-a.cpu_executor_graph_builds},
        {"cpu_executor_graph_hits",b.cpu_executor_graph_hits-a.cpu_executor_graph_hits},
        {"cpu_executor_graph_build_ms",1000*(b.cpu_executor_graph_build_seconds-a.cpu_executor_graph_build_seconds)},
        {"cpu_executor_pack_ms",1000*(b.cpu_executor_pack_seconds-a.cpu_executor_pack_seconds)},
        {"cpu_executor_read_ms",1000*(b.cpu_executor_read_seconds-a.cpu_executor_read_seconds)},
        {"cpu_executor_retained_graphs_before",a.cpu_executor_retained_graphs},
        {"cpu_executor_retained_graphs_after",b.cpu_executor_retained_graphs},
        {"cpu_executor_retained_scratch_bytes_before",a.cpu_executor_retained_scratch_bytes},
        {"cpu_executor_retained_scratch_bytes_after",b.cpu_executor_retained_scratch_bytes},
        {"cpu_bytes_scope","Selected weight payload once per layer/expert; not measured DRAM traffic."}};
}
J append(Engine & engine,const Input & input,bool prefix=false){
    const auto & ids=prefix?input.prefix:input.ids;J chunks=J::array();std::string error;
    for(size_t offset=0;offset<ids.size();){
        const int n=engine.prefill_chunk(int(ids.size()-offset),error);check(n>0,"planner failed: "+error);
        const int64_t position=engine.position();const auto start=Clock::now();
        const bool ok=!prefix&&input.visual
            ?engine.forward_mixed(ids.data()+offset,input.mixed.visual_rows.data()+offset,input.mixed.embeddings.data(),input.mixed.visual_count,n,error)
            :engine.forward(ids.data()+offset,n,error);
        check(ok,"forward failed: "+error);chunks.push_back({{"position",position},{"tokens",n},{"forward_ms",ms(start)}});offset+=size_t(n);
    }
    return chunks;
}
J compare(Engine & engine,const fs::path & directory,const fs::path & model,const Input & input,const J & chunks,const J & rules){
    check(input.prefix.empty(),"oracle gate with prefix is not supported");
    std::ifstream manifest(directory/"manifest.json");J ref;manifest>>ref;
    const int n=int(input.ids.size()),vocab=int(engine.geometry().vocab);
    check(ref.at("format")=="m31-logits-v1"&&ref.at("engine")=="llama.cpp-PR31192"&&ref.at("backend")=="cuda"&&ref.at("oracle_op_offload")==true&&ref.at("kv")=="f16"&&ref.at("full_last_layer_before_crop")==true,"oracle must be accepted GPU-op-offload/F16/full-layer PR oracle");
    check(fs::canonical(ref.at("model").get<std::string>())==fs::canonical(model),"oracle model differs from loaded model");
    check(ref.at("tokens")==n&&ref.at("vocab")==vocab&&run::read_ids(directory/"tokens.i32")==input.ids,"oracle ids or geometry mismatch");
    check(chunks.size()==1&&chunks[0].at("position")==0&&chunks[0].at("tokens")==n&&ref.at("actual_chunks").size()==1&&ref.at("actual_chunks")[0].at("position")==0&&ref.at("actual_chunks")[0].at("tokens")==n,"oracle gate requires same one-chunk recipe at position zero");
    // A saved vision oracle may specify 4096 as its upper head bound. When
    // n<=4096 that is exactly the same uncropped [0,n) head as head_tail=0.
    const int head_tail=ref.value("head_tail",0);
    check(head_tail==0||head_tail>=n,"oracle cropped its head: full-row head geometry differs");
    const auto positions=ref.at("positions").get<std::vector<int>>();
    check(positions==run::selected_rows("all",n),"oracle must contain all rows");
    check(fs::file_size(directory/"logits.f32")==size_t(n)*size_t(vocab)*4,"oracle file truncated");
    std::string error;std::vector<float> actual;const auto head=Clock::now();
    check(engine.logits(0,n,actual,error),"full-row gate head failed: "+error);const double head_ms=ms(head);
    check(actual.size()==size_t(n)*size_t(vocab),"head size mismatch");
    std::ifstream f(directory/"logits.f32",std::ios::binary);std::vector<float> expected(size_t(vocab),0);
    size_t equal_rows=0,equal_values=0,same_top=0;double kl=0,max_kl=0,max_abs=0;
    for(int row=0;row<n;++row){
        f.read(reinterpret_cast<char *>(expected.data()),std::streamsize(expected.size()*4));check(bool(f),"short oracle read");
        const float * a=actual.data()+size_t(row)*size_t(vocab);const float * b=expected.data();
        if(!std::memcmp(a,b,size_t(vocab)*4)){++equal_rows;equal_values+=size_t(vocab);++same_top;continue;}
        int ai=0,bi=0;double am=-INFINITY,bm=-INFINITY;
        for(int v=0;v<vocab;++v){check(std::isfinite(a[v])&&std::isfinite(b[v]),"nonfinite logits");if(a[v]>am){am=a[v];ai=v;}if(b[v]>bm){bm=b[v];bi=v;}if(!std::memcmp(a+v,b+v,4))++equal_values;max_abs=std::max(max_abs,std::abs(double(a[v])-b[v]));}
        same_top+=ai==bi;double az=0,bz=0;for(int v=0;v<vocab;++v){az+=std::exp(double(a[v])-am);bz+=std::exp(double(b[v])-bm);}
        const double al=am+std::log(az),bl=bm+std::log(bz);double row_kl=0;
        for(int v=0;v<vocab;++v)row_kl+=std::exp(double(b[v])-bl)*(double(b[v])-bl-double(a[v])+al);
        row_kl=std::max(0.0,row_kl);kl+=row_kl;max_kl=std::max(max_kl,row_kl);
    }
    const double top=double(same_top)/n,mean_kl=kl/n;
    const bool require_bitwise=rules.value("bitwise",true);
    const bool pass=require_bitwise?equal_rows==size_t(n):(top>=rules.value("min_top1",.99)&&mean_kl<=rules.value("max_mean_kl",1e-3)&&max_kl<=rules.value("max_kl",.05));
    return {{"pass",pass},{"oracle",fs::canonical(directory).string()},{"rows",n},{"vocab",vocab},{"full_head_rows",n},{"oracle_head_tail",head_tail},{"bitwise_rows",equal_rows},{"bitwise_values",equal_values},{"top1",top},{"mean_kl_oracle_to_candidate",mean_kl},{"max_kl",max_kl},{"max_abs",max_abs},{"head_ms",head_ms},{"rules",rules}};
}
}
int main(int argc,char ** argv){
    ggml_backend_t backend=nullptr;fs::path output;J result={{"format","m31-shortprefill-result-v1"},{"pass",false},{"runs",J::array()}};
    try{
        fs::path plan_path;for(int i=1;i<argc;i+=2){check(i+1<argc,"usage: --plan PLAN --out NEW_DIR");const std::string flag=argv[i];if(flag=="--plan")plan_path=argv[i+1];else if(flag=="--out")output=argv[i+1];else throw std::runtime_error("unexpected argument "+flag);}
        check(!plan_path.empty()&&!output.empty(),"usage: --plan PLAN --out NEW_DIR");
        check(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1"&&std::getenv("CUDA_VISIBLE_DEVICES")&&std::string(std::getenv("CUDA_VISIBLE_DEVICES"))=="0","requires aictl-owned GPU0 session");
        std::ifstream file(plan_path);J plan;file>>plan;check(plan.at("format")=="m31-shortprefill-plan-v1","unknown plan format");
        const bool profile_capture=plan.value("cuda_profile_capture",false);
#ifndef M31_HAVE_CUDA
        check(!profile_capture,"cuda_profile_capture requires a CUDA build/backend");
#endif
        const auto arms=plan.at("arms");check(arms.is_array()&&!arms.empty()&&plan.at("cases").is_array()&&!plan.at("cases").empty(),"empty arms/cases");
        Environment env(arms);std::set<std::string> arm_names;
        for(const auto & arm:arms)check(arm_names.insert(arm.at("name").get<std::string>()).second,"duplicate arm name");
        check(fs::create_directory(output),"output already exists or parent missing");
        save(output/"plan.json",plan);result["plan"]=fs::canonical(plan_path).string();
        result["notes"]={"One Engine initialization; expert LRU and route history persist across arms.","KV reset before each sample; no prompt checkpoint restore or reuse.","Each repetition rotates arm order; warmups use the same scheduling.","prefix prefill is excluded from suffix timing; modes forced off during prefix.","Forward includes synchronous device completion; head timing is last-row only except oracle gates."};
#ifdef M31_HAVE_CUDA
        backend=ggml_backend_cuda_init(0);check(backend,"CUDA initialization failed");
#else
        throw std::runtime_error("built without CUDA");
#endif
        {
            Config cfg;const auto options=plan.value("configuration",J::object());cfg.backend=backend;
            cfg.context=options.value("context",4096);cfg.max_chunk=options.value("max_chunk",4096);cfg.threads=options.value("threads",80);
            cfg.arena_gib=options.value("arena_gib",170.0);cfg.gpu_experts_gib=options.value("gpu_experts_gib",7.0);cfg.gpu_experts_min_tokens=options.value("gpu_experts_min_tokens",32);
            cfg.reuse_allocations=true;cfg.persistent_cpu_pool=true;cfg.cpu_residual=true;cfg.cpu_decode_cache=true;cfg.dense_decode_cache=true;cfg.profile=options.value("profile",false);
            J observed=J::array();const bool routes=options.value("record_routes",false);
            if(routes)cfg.route_observer=[&](int layer,int n,const std::vector<int32_t> & ids,const std::vector<float> &){std::map<int,size_t> counts;for(int id:ids)++counts[id];J histogram=J::array();for(const auto & count:counts)histogram.push_back({count.first,count.second});observed.push_back({{"layer",layer},{"tokens",n},{"distinct",counts.size()},{"assignment_counts",histogram}});};
            Engine engine;std::string error;const auto init=Clock::now();check(engine.init(plan.at("model"),cfg,error),"init failed: "+error);result["init_ms"]=ms(init);result["configuration"]=options;
            result["vocab"]=engine.geometry().vocab;result["engine_initializations"]=1;
            const int warmups=plan.value("warmups",1),repeats=plan.value("repeats",3);check(warmups>=0&&repeats>=1&&repeats<=100,"invalid repetition counts");
            std::ofstream lines(output/"runs.jsonl");
            for(const auto & item:plan.at("cases")){
                const auto input=load(item,int(engine.geometry().vocab),int(cfg.context));
                for(int iteration=-warmups;iteration<repeats;++iteration)for(size_t index=0;index<arms.size();++index){
                    const size_t rotate=size_t(iteration+warmups)%arms.size();const auto & arm=arms[(index+rotate)%arms.size()];
                    engine.reset();env.apply(arm);setenv("M31_SHORT_PREFILL_MODE","off",1);observed=J::array();
                    const auto prefix_chunks=append(engine,input,true);env.apply(arm);observed=J::array();
                    J row={{"case",item.at("name")},{"arm",arm.at("name")},{"iteration",iteration},{"warmup",iteration<0},{"env",arm.at("env")},{"tokens",input.ids.size()},{"prefix_tokens",input.prefix.size()},{"prefix_chunks",prefix_chunks}};
                    std::cerr<<"M31 shortprefill sample case="<<item.at("name")<<" arm="<<arm.at("name")<<" iteration="<<iteration<<" n="<<input.ids.size()<<" prefix="<<input.prefix.size()<<'\n';
                    const auto before=engine.prefill_stats();
#ifdef M31_HAVE_CUDA
                    if(profile_capture&&iteration>=0)check(cudaProfilerStart()==cudaSuccess,"cudaProfilerStart failed");
#endif
                    const auto start=Clock::now();row["chunks"]=append(engine,input);row["forward_ms"]=ms(start);row["counters"]=difference(before,engine.prefill_stats());
                    const int last_n=row["chunks"].back().at("tokens");std::vector<float> logits;const auto head=Clock::now();check(engine.logits(last_n-1,1,logits,error),"last head failed: "+error);row["head_ms"]=ms(head);row["native_ttft_ms"]=ms(start);
#ifdef M31_HAVE_CUDA
                    if(profile_capture&&iteration>=0){check(cudaProfilerStop()==cudaSuccess,"cudaProfilerStop failed");row["cuda_profile_capture"]=true;}
#endif
                    check(logits.size()==size_t(engine.geometry().vocab),"last logits size mismatch");for(float v:logits)check(std::isfinite(v),"nonfinite last logits");row["greedy_token"]=std::max_element(logits.begin(),logits.end())-logits.begin();
                    row["tokens_per_second"]=1000.0*input.ids.size()/row["forward_ms"].get<double>();if(routes)row["routes"]=observed;
                    if(item.contains("oracle")&&iteration>=0)row["gate"]=compare(engine,item.at("oracle").get<std::string>(),plan.at("model").get<std::string>(),input,row["chunks"],item.value("gate",J{{"bitwise",true}}));
                    row["pass"]=!row.contains("gate")||row["gate"]["pass"].get<bool>();result["runs"].push_back(row);lines<<row.dump()<<'\n';lines.flush();save(output/"summary.json",result);
                    check(row["pass"],"oracle gate failed "+item.at("name").get<std::string>()+" "+arm.at("name").get<std::string>());
                }
            }
        }
        ggml_backend_free(backend);backend=nullptr;result["pass"]=true;save(output/"summary.json",result);std::cout<<J{{"pass",true},{"runs",result["runs"].size()},{"summary",(output/"summary.json").string()}}.dump()<<'\n';return 0;
    }catch(const std::exception & error){result["error"]=error.what();if(!output.empty()&&fs::is_directory(output))save(output/"summary.json",result);if(backend)ggml_backend_free(backend);std::cerr<<error.what()<<'\n';return 1;}
}
