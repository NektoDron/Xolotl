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
// Isolated component speed comparison; this is not an om_bench M2 result.
int main(int argc,char ** argv) {
 ggml_backend_t cpu=nullptr;
 try {
  check(argc==3,"usage: port5_kernel_speed MODEL NEW_OUT");
  const char * session=std::getenv("M31_CARD_SESSION"),*visible=std::getenv("CUDA_VISIBLE_DEVICES"),*window=std::getenv("M31_ROUTE_SPEED_WINDOW");
  check(session && std::string(session)=="1" && visible && std::string(visible)=="0" && window && std::string(window)=="1","isolated GPU0 measurement window required");
  check(!std::filesystem::exists(argv[2]),"output already exists");
  ck(cudaSetDevice(0));cpu=ggml_backend_cpu_init();check(cpu,"CPU backend");
  const auto file=strata::GgufModel::open(argv[1]);const auto g=read_geometry(file);
  const char * tensor_env=std::getenv("M31_COMPONENT_TENSOR");
  const bool tensor_comparison=tensor_env && std::string(tensor_env)=="1";
  auto select=[&](const char * candidate) {
   setenv("M31_GPU_CPU_QUANTS_BATCHED",tensor_comparison ? "1" : candidate,1);
   setenv("M31_GPU_CPU_QUANTS_TENSOR",tensor_comparison && std::string(candidate)=="1" ? "1" : "0",1);
  };
  json results=json::array();
  for(int type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS,GGML_TYPE_NVFP4}) {
   Weight weight(file,g,type,0,23,cpu);
   for(int n:{128,2048}) {
    const bool broadcast=type==GGML_TYPE_IQ3_XXS;
    const auto ids=routes(n,4,weight.slots,107+uint64_t(n));
    const auto input=random_values(size_t(weight.columns)*n*(broadcast?1:4),1009+uint64_t(n)+type);
    GpuCpuQuants baseline,candidate;
    select("0");
    std::vector<float> original,updated;
    baseline.multiply_into(type,weight.device,weight.stride,weight.columns,weight.rows,4,n,input,ids,original);
    select("1");
    candidate.multiply_into(type,weight.device,weight.stride,weight.columns,weight.rows,4,n,input,ids,updated);
    check(std::memcmp(original.data(),updated.data(),original.size()*sizeof(float))==0,"bitwise changed in full-row speed probe");
    auto measure=[&](GpuCpuQuants & implementation,std::vector<float> & output,const char * batch) {
      select(batch);
      const auto start=std::chrono::steady_clock::now();
      for(int repeat=0;repeat<2;++repeat)implementation.multiply_into(type,weight.device,weight.stride,weight.columns,weight.rows,4,n,input,ids,output);
      return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/2;
    };
    const double a=measure(baseline,original,"0"),b=measure(candidate,updated,"1");
    results.push_back({{"type",ggml_type_name(ggml_type(type))},{"tensor",weight.name},{"n",n},{"columns",weight.columns},{"rows",weight.rows},{"baseline_ms",a},{"batched_ms",b},{"speedup",a/b},{"bitwise",true}});
    baseline.print_profile("scalar");candidate.print_profile("batched");
   }
  }
  std::ofstream(argv[2])<<json({{"scope","isolated-multiply-component-not-M2-om-bench"},{"reusable_readback",true},{"comparison",tensor_comparison ? "DP4A-vs-WMMA" : "scalar-vs-DP4A"},{"cases",results}}).dump(2)<<'\n';
  ggml_backend_free(cpu);return 0;
 }catch(const std::exception & e){std::cerr<<e.what()<<'\n';if(cpu)ggml_backend_free(cpu);return 1;}
}
