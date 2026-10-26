// Independent scheduling equivalence probe. Real quant bytes, no model inference.
#include "prefill_route_quants.hpp"
#include "gpu_prepared_pair.hpp"
#include "cpu_gpu_host_executor.hpp"
#include "ggml-cuda.h"
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
    Weight(const strata::GgufModel & model,const Geometry & g,int type,int want_rows,int slot_count,ggml_backend_t cpu,int min_layer=-1) {
        const strata::TensorInfo * info=nullptr;
        size_t shard=0;
        for(int layer=std::max(int(g.dense_lead),min_layer);layer<g.layers && !info;++layer)
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
 ggml_backend_t cpu=nullptr,gpu=nullptr;ggml_context * metadata=nullptr;
 ggml_backend_buffer_t pool=nullptr;ggml_backend_event_t ready=nullptr;
 try {
  check(argc==3,"usage: port5_prepared_pair_check MODEL NEW_OUT");
  const char * session=std::getenv("M31_CARD_SESSION"),*visible=std::getenv("CUDA_VISIBLE_DEVICES");
  check(session && std::string(session)=="1" && visible && std::string(visible)=="0","GPU0 card session required");
  check(!std::filesystem::exists(argv[2]),"output already exists");
  ck(cudaSetDevice(0));cpu=ggml_backend_cpu_init();gpu=ggml_backend_cuda_init(0);
  check(cpu && gpu,"backend init");ggml_backend_cpu_set_n_threads(cpu,16);
  ready=ggml_backend_event_new(ggml_backend_get_device(gpu));check(ready,"event allocation");
  const auto file=strata::GgufModel::open(argv[1]);const auto geometry=read_geometry(file);
  GpuHostExecutor host(GpuHostExecutor::Options{16,32,4});GpuPreparedPair pair(host);
  json cases=json::array();
  for(int type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS}) {
   Weight gate(file,geometry,type,67,23,cpu,3),up(file,geometry,type,67,23,cpu,4);
   check(gate.columns==up.columns && gate.rows==up.rows && gate.stride==up.stride,"pair shape");
   metadata=ggml_init({65536,nullptr,true});check(metadata,"pool metadata");
   auto * gate_proxy=ggml_new_tensor_1d(metadata,GGML_TYPE_I8,int64_t(gate.stride*gate.slots));
   auto * up_proxy=ggml_new_tensor_1d(metadata,GGML_TYPE_I8,int64_t(up.stride*up.slots));
   pool=ggml_backend_alloc_ctx_tensors(metadata,gpu);check(pool,"pool allocation");
   for(int n:{128,2048}) for(bool broadcast:{true,false}) {
    const auto ids=routes(n,4,gate.slots,907+uint64_t(n));
    const auto input=random_values(size_t(gate.columns)*n*(broadcast?1:4),101+uint64_t(n)+type);
    const auto expected_gate=cpu_reference(cpu,gate,4,n,broadcast,input,ids);
    const auto expected_up=cpu_reference(cpu,up,4,n,broadcast,input,ids);
    pair.prepare(type,type,gate.columns,gate.rows,4,n,input,ids);
    bool rejected=false;
    try {pair.prepare(type,type,gate.columns,gate.rows,4,n,input,ids);}catch(const std::exception &){rejected=true;}
    check(rejected,"nested prepare accepted");
    std::vector<float> actual_gate,actual_up;
    rejected=false;try{pair.read_into(actual_gate,actual_up);}catch(const std::exception &){rejected=true;}
    check(rejected,"incomplete read accepted");
    const auto order=prefill_route_worklist(ids,gate.slots,4);
    std::vector<int32_t> group,physical;
    for(int expert=0;expert<gate.slots;++expert) {
     if(!order.histogram[size_t(expert)])continue;
     const int slot=(expert*7+3)%gate.slots;
     const auto * gs=static_cast<const uint8_t *>(gate.tensor->data)+size_t(expert)*gate.stride;
     const auto * us=static_cast<const uint8_t *>(up.tensor->data)+size_t(expert)*up.stride;
     ggml_backend_tensor_set_async(gpu,gate_proxy,gs,size_t(slot)*gate.stride,gate.stride);
     ggml_backend_tensor_set_async(gpu,up_proxy,us,size_t(slot)*up.stride,up.stride);
     group.push_back(expert);physical.push_back(slot);
     if(group.size()==3 || expert==gate.slots-1) {
      ggml_backend_event_record(ready,gpu);
      pair.enqueue(gate_proxy->data,up_proxy->data,gate.stride,group,physical,ready);
      rejected=false;try{pair.enqueue(gate_proxy->data,up_proxy->data,gate.stride,group,physical,ready);}catch(const std::exception &){rejected=true;}
      check(rejected,"duplicate group accepted");group.clear();physical.clear();
     }
    }
    check(group.empty(),"test forgot final group");
    pair.read_into(actual_gate,actual_up);
    const auto a=metric(expected_gate,actual_gate),b=metric(expected_up,actual_up);
    check(a["bitwise"].get<bool>() && b["bitwise"].get<bool>(),"prepared pair differs from CPU");
    cases.push_back({{"type",ggml_type_name(ggml_type(type))},{"n",n},{"rows",gate.rows},{"columns",gate.columns},{"broadcast",broadcast},{"gate",a},{"up",b}});
   }
   pair.abort();ggml_backend_buffer_free(pool);pool=nullptr;ggml_free(metadata);metadata=nullptr;
  }
  std::ofstream(argv[2])<<json({{"pass",true},{"cases",cases},{"event_bridge",true},{"logical_to_physical_permutation",true},{"partial_rows",true},{"scratch_bytes",pair.scratch_bytes()}}).dump(2)<<'\n';
  ggml_backend_event_free(ready);ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
 }catch(const std::exception & error) {
  std::cerr<<error.what()<<'\n';
  if(ready)ggml_backend_event_free(ready);if(pool)ggml_backend_buffer_free(pool);if(metadata)ggml_free(metadata);
  if(gpu)ggml_backend_free(gpu);if(cpu)ggml_backend_free(cpu);return 1;
 }
}
