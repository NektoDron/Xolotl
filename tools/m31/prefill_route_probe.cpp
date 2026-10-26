// Independent scheduling equivalence probe. Real quant bytes, no model inference.
#include "prefill_route_quants.hpp"
#include "prefill_route_worklist.hpp"
#include "gpu_cpu_quants.hpp"
#include "m31_geometry.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "json.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

using namespace strata::m31;
using json=nlohmann::json;
static void check(bool ok,const char * message) {if(!ok)throw std::runtime_error(message);}
static void ck(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
static std::vector<float> random_values(size_t count,uint64_t seed) {
    std::vector<float> out(count);
    for(auto & value:out) {
        seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
        value=float((seed>>40)&0xffffff)*(1.f/8388608.f)-1.f;
    }
    return out;
}
static std::vector<int32_t> routes(int n,int used,int slots,uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> out(size_t(n)*used);
    for(int token=0;token<n;++token) {
        for(int rank=0;rank<used;++rank) {
            int id;
            do{id=int(rng()%uint64_t(slots-3));}
            while(std::find(out.begin()+size_t(token)*used,out.begin()+size_t(token)*used+rank,id)!=out.begin()+size_t(token)*used+rank);
            out[size_t(token)*used+rank]=id;
        }
        if(token<7)out[size_t(token)*used]=slots-3;
        else if(token<15)out[size_t(token)*used]=slots-2;
        else if(token<24)out[size_t(token)*used]=slots-1;
    }
    return out;
}
struct Weight {
    ggml_context * context=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_tensor * tensor=nullptr;
    uint8_t * device=nullptr;
    int columns=0,rows=0,slots=0;
    size_t stride=0;
    std::string name;
    Weight(const strata::GgufModel & model,const Geometry & g,int type,int want_rows,int slot_count,ggml_backend_t cpu) {
        const strata::TensorInfo * info=nullptr;
        size_t shard=0;
        for(int layer=int(g.dense_lead);layer<g.layers && !info;++layer)
            for(const char * role:{"gate","up","down"}) {
                const auto candidate=layer_name(layer,std::string("ffn_")+role+"_exps.weight");
                size_t candidate_shard=0;
                const auto * t=model.find(candidate,&candidate_shard);
                if(t && int(t->type)==type) {info=t;name=candidate;shard=candidate_shard;break;}
            }
        check(info && info->shape.size()==3 && info->shape[2]>=uint64_t(slot_count),"missing requested real quant tensor");
        columns=int(info->shape[0]);rows=want_rows?std::min(want_rows,int(info->shape[1])):int(info->shape[1]);slots=slot_count;
        const auto quant=ggml_type(type);
        const size_t row_bytes=ggml_row_size(quant,columns);
        stride=row_bytes*size_t(rows);
        const size_t original_stride=row_bytes*size_t(info->shape[1]);
        std::vector<uint8_t> packed(stride*slots);
        const auto * source=model.shard(shard).tensor_data(*info);
        for(int slot=0;slot<slots;++slot)
            std::memcpy(packed.data()+size_t(slot)*stride,source+size_t(slot)*original_stride,stride);
        context=ggml_init({64*1024,nullptr,true});check(context,"weight metadata allocation");
        tensor=ggml_new_tensor_3d(context,quant,columns,rows,slots);
        buffer=ggml_backend_alloc_ctx_tensors(context,cpu);check(buffer,"CPU weight allocation");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(tensor,packed.data(),0,packed.size());
        ck(cudaMalloc(reinterpret_cast<void **>(&device),packed.size()));
        ck(cudaMemcpy(device,packed.data(),packed.size(),cudaMemcpyHostToDevice));
    }
    ~Weight(){cudaFree(device);if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);}
};
static std::vector<float> cpu_reference(ggml_backend_t cpu,const Weight & w,int used,int n,bool broadcast,
                                       const std::vector<float> & input,const std::vector<int32_t> & ids) {
    auto * context=ggml_init({1024*1024,nullptr,true});check(context,"CPU probe metadata");
    auto * graph=ggml_new_graph_custom(context,128,false);
    auto * x=ggml_new_tensor_3d(context,GGML_TYPE_F32,w.columns,broadcast?1:used,n);
    auto * routes=ggml_new_tensor_2d(context,GGML_TYPE_I32,used,n);
    ggml_set_input(x);ggml_set_input(routes);
    auto * result=ggml_mul_mat_id(context,w.tensor,x,routes);
    ggml_set_output(result);ggml_build_forward_expand(graph,result);
    auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    check(allocator && ggml_gallocr_alloc_graph(allocator,graph),"CPU probe allocator");
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    ggml_backend_tensor_set(routes,ids.data(),0,ids.size()*sizeof(int32_t));
    check(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"CPU MUL_MAT_ID failed");
    std::vector<float> out(size_t(w.rows)*ids.size());
    ggml_backend_tensor_get(result,out.data(),0,out.size()*sizeof(float));
    ggml_gallocr_free(allocator);ggml_free(context);
    return out;
}
static json metric(const std::vector<float> & a,const std::vector<float> & b) {
    check(a.size()==b.size(),"output shape mismatch");
    double maxabs=0,aa=0,dd=0;
    for(size_t i=0;i<a.size();++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite probe output");
        const double difference=double(a[i])-b[i];
        maxabs=std::max(maxabs,std::abs(difference));dd+=difference*difference;aa+=double(a[i])*a[i];
    }
    return {{"bitwise",std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0},
            {"max_abs",maxabs},{"relative_l2",std::sqrt(dd/std::max(aa,1e-300))},{"values",a.size()}};
}
int main(int argc,char ** argv) {
    ggml_backend_t cpu=nullptr;
    try {
        std::string model,out,dispatch_mode="default";
        int row_cap=64,bench_iters=0;
        for(int i=1;i<argc;++i) {
            check(i+1<argc,"missing argument");
            const std::string key=argv[i],value=argv[++i];
            if(key=="--model")model=value;
            else if(key=="--out")out=value;
            else if(key=="--rows")row_cap=std::stoi(value);
            else if(key=="--bench-iters")bench_iters=std::stoi(value);
            else if(key=="--dispatch-mode")dispatch_mode=value;
            else throw std::runtime_error("unexpected option "+key);
        }
        check(!model.empty() && !out.empty() && !std::filesystem::exists(out),"require model and NEW output file");
        check(row_cap>=0 && row_cap<=65536 && bench_iters>=0 && bench_iters<=100,"invalid rows/benchmark count");
        check(dispatch_mode=="default" || dispatch_mode=="disabled" || dispatch_mode=="forced","invalid dispatch mode");
        const auto * session=std::getenv("M31_CARD_SESSION"),*visible=std::getenv("CUDA_VISIBLE_DEVICES");
        check(session && std::string(session)=="1" && visible && std::string(visible)=="0",
              "GPU0 probe requires aictl/flock runner and CUDA_VISIBLE_DEVICES=0");
        if(bench_iters) {
            const auto * window=std::getenv("M31_ROUTE_SPEED_WINDOW");
            check(window && std::string(window)=="1","benchmark requires explicit coordinator measurement window");
        }
        ck(cudaSetDevice(0));
        // CPU tiled flags are cached with call_once: each mode MUST run in a
        // separate process, configured before CPU backend initialization.
        if(dispatch_mode=="default") {unsetenv("GGML_CPU_TILED_MM");unsetenv("GGML_CPU_TILED_MM_FORCE");}
        else if(dispatch_mode=="disabled") {setenv("GGML_CPU_TILED_MM","0",1);unsetenv("GGML_CPU_TILED_MM_FORCE");}
        else {setenv("GGML_CPU_TILED_MM","1",1);setenv("GGML_CPU_TILED_MM_FORCE","1",1);}
        cpu=ggml_backend_cpu_init();check(cpu,"CPU backend initialization");
        ggml_backend_cpu_set_n_threads(cpu,16);
        const auto file=strata::GgufModel::open(model);
        const auto g=read_geometry(file);
        json cases=json::array();
        bool passed=true;
        for(int type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS,GGML_TYPE_NVFP4}) {
            Weight weight(file,g,type,row_cap,23,cpu);
            GpuCpuQuants baseline;
            PrefillRouteQuants candidate;
            {
                const auto & mode=dispatch_mode;
                for(int n:{1,7,128,2048}) {
                    if(mode=="disabled" && n!=128)continue;
                    if(mode=="forced" && n!=7)continue;
                    for(bool broadcast:{true,false}) {
                        constexpr int used=4;
                        const auto ids=routes(n,used,weight.slots,107+uint64_t(n));
                        const auto input=random_values(size_t(weight.columns)*n*(broadcast?1:used),1009+uint64_t(n)+type);
                        const auto cpu_out=cpu_reference(cpu,weight,used,n,broadcast,input,ids);
                        const auto original=baseline.multiply(type,weight.device,weight.stride,weight.columns,weight.rows,used,n,input,ids);
                        const auto same_order=candidate.multiply(type,weight.device,weight.stride,weight.columns,weight.rows,used,n,input,ids,false);
                        const auto sorted=candidate.multiply(type,weight.device,weight.stride,weight.columns,weight.rows,used,n,input,ids,true);
                        const auto work=prefill_route_worklist(ids,weight.slots,used,mode!="disabled",mode=="forced");
                        json row={{"type",ggml_type_name(ggml_type(type))},{"tensor",weight.name},
                            {"columns",weight.columns},{"rows",weight.rows},{"slots",weight.slots},
                            {"n",n},{"used",used},{"broadcast",broadcast},{"dispatch_mode",mode},
                            {"baseline_vs_cpu",metric(cpu_out,original)},
                            {"prototype_token_order_vs_baseline",metric(original,same_order)},
                            {"prototype_expert_order_vs_baseline",metric(original,sorted)},
                            {"original_expert_switches",work.original_expert_switches},
                            {"scheduled_expert_switches",work.scheduled_expert_switches}};
                        const bool row_pass=row["baseline_vs_cpu"]["bitwise"].get<bool>() &&
                            row["prototype_token_order_vs_baseline"]["bitwise"].get<bool>() &&
                            row["prototype_expert_order_vs_baseline"]["bitwise"].get<bool>();
                        row["pass"]=row_pass;passed=passed && row_pass;
                        if(bench_iters && mode=="default" && n>=128) {
                            auto bench=[&](bool sort) {
                                const auto start=std::chrono::steady_clock::now();
                                for(int repeat=0;repeat<bench_iters;++repeat)
                                    (void)candidate.multiply(type,weight.device,weight.stride,weight.columns,weight.rows,used,n,input,ids,sort);
                                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/bench_iters;
                            };
                            // Alternate order to retain simple raw repetitions.
                            json samples=json::array();
                            for(int pair=0;pair<2;++pair) {
                                const double a=bench(pair!=0),b=bench(pair==0);
                                samples.push_back({{"token_order_wall_ms",pair?b:a},{"expert_order_wall_ms",pair?a:b}});
                            }
                            row["wall_repetitions"]=samples;
                        }
                        cases.push_back(std::move(row));
                        std::fprintf(stderr,"route probe type=%s n=%d layout=%s mode=%s pass=%d\n",ggml_type_name(ggml_type(type)),n,broadcast?"broadcast":"per-route",mode.c_str(),int(row_pass));
                    }
                }
            }
        }
        unsetenv("GGML_CPU_TILED_MM");unsetenv("GGML_CPU_TILED_MM_FORCE");
        json result={{"format","m31-route-order-probe-v1"},{"pass",passed},{"cases",cases},
                     {"case_count",cases.size()},{"model",model},{"row_cap",row_cap},
                     {"dispatch_mode",dispatch_mode},{"dispatch_configured_before_cpu_init",true},
                     {"note","Original real quant bytes; scheduler only changes route work order, output indices/input rows/full histogram retained. No full-model inference. Offline switch counts do not establish speedup."}};
        if(bench_iters)result["benchmark_note"]="Mean wall per call includes CPU Q8 quantization, metadata sorting/H2D, kernel and D2H; excludes weight load. Diagnostic matrix replay, not model t/s. Two alternating-order pairs retained.";
        std::ofstream stream(out);stream<<result.dump(2)<<'\n';check(bool(stream),"cannot write route report");
        ggml_backend_free(cpu);cpu=nullptr;
        return passed?0:1;
    } catch(const std::exception & e) {
        if(cpu)ggml_backend_free(cpu);
        std::cerr<<"route probe: "<<e.what()<<'\n';return 1;
    }
}
