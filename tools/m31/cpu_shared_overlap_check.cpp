#include "m31_engine.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::m31;
static void check(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
struct Mode {bool dense=true,taps=false,gpu_decode=false;};
struct Output {std::vector<std::vector<float>> rows;std::vector<std::vector<uint8_t>> taps;};
static Output run(const std::string & model,ggml_backend_t backend,bool flag,Mode mode,bool inject=false){
    setenv("M31_CPU_SHARED_OVERLAP",flag?"1":"0",1);
    setenv("M31_CPU_SHARED_OVERLAP_TRACE","1",1);
    if(inject)setenv("M31_CPU_SHARED_OVERLAP_FAIL_AFTER_LAUNCH","1",1);
    else unsetenv("M31_CPU_SHARED_OVERLAP_FAIL_AFTER_LAUNCH");
    Engine engine;Config cfg;cfg.backend=backend;cfg.threads=2;cfg.context=640;cfg.max_chunk=128;
    cfg.reuse_allocations=true;cfg.cpu_residual=true;cfg.persistent_cpu_pool=true;
    cfg.cpu_decode_cache=true;cfg.dense_decode_cache=mode.dense;cfg.profile=true;
    if(mode.gpu_decode){cfg.arena_gib=.01;cfg.gpu_experts_gib=.01;cfg.gpu_experts_min_tokens=1;}
    Output out;
    if(mode.taps)cfg.tap=[&](const std::string &name,int64_t,const ggml_tensor*t){
        if(name!="l_out-1"&&name!="ffn_moe_out-1")return;
        check(ggml_is_contiguous(t),"noncontiguous fixture tap");std::vector<uint8_t> b(ggml_nbytes(t));
        ggml_backend_tensor_get(t,b.data(),0,b.size());out.taps.push_back(std::move(b));
    };
    std::string error;if(!engine.init(model,cfg,error))throw std::runtime_error(error);
    if(inject){
        const int32_t id=3;
        check(!engine.forward(&id,1,error)&&error.find("injected failure after shared FFN launch")!=std::string::npos,"failure hook not exercised");
        check(!engine.forward(&id,1,error),"poisoned Engine continued without reset");
        unsetenv("M31_CPU_SHARED_OVERLAP_FAIL_AFTER_LAUNCH");engine.reset();
    }
    std::mt19937 random(1417);
    const auto phases=mode.taps||!mode.dense||mode.gpu_decode
        ?std::vector<std::vector<int>>{{7,1,1},{1,7,1}}
        :std::vector<std::vector<int>>{{128,128,1,1,128,128,1},{7,1,128,1}};
    for(const auto &phase:phases){engine.reset();for(int n:phase){
        std::vector<int32_t> ids(size_t(n),0);for(auto &id:ids)id=int32_t(random()%64);
        if(!engine.forward(ids.data(),n,error))throw std::runtime_error(error);
        std::vector<float> logits;if(!engine.logits(0,n,logits,error))throw std::runtime_error(error);
        out.rows.push_back(std::move(logits));
    }}
    const auto saved=out.rows.front();engine.reset();
    check(out.rows.front()==saved,"owned logits changed during reset");return out;
}
static void identical(const Output&a,const Output&b){
    check(a.rows.size()==b.rows.size(),"row count mismatch");
    for(size_t i=0;i<a.rows.size();++i)check(a.rows[i].size()==b.rows[i].size()&&!std::memcmp(a.rows[i].data(),b.rows[i].data(),a.rows[i].size()*4),"shared overlap changed logits bits");
    check(a.taps==b.taps,"fallback changed tap bytes");
}
int main(int argc,char**argv){
    ggml_backend_t gpu=nullptr;
    try{
        check(argc==3,"usage: cpu_shared_overlap_check MODEL cpu|cuda");const bool cuda=std::string(argv[2])=="cuda";
        unsetenv("M31_GPU_CPU_QUANTS");unsetenv("M31_GPU_PREFILL_ASYNC");unsetenv("M31_GPU_SEED_HOT");
        if(cuda){
#ifdef M31_HAVE_CUDA
            check(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","aictl runner required");
            gpu=ggml_backend_cuda_init(0);check(gpu,"CUDA0 init failed");
#else
            throw std::runtime_error("CUDA unavailable");
#endif
        }
        for(Mode mode:{Mode{},Mode{true,true,false},Mode{false,false,false},Mode{true,false,true}}){
            if(mode.gpu_decode&&!cuda)continue;
            const auto base=run(argv[1],gpu,false,mode),opt=run(argv[1],gpu,true,mode);identical(base,opt);
            if(cuda&&!mode.taps&&mode.dense&&!mode.gpu_decode)identical(base,run(argv[1],gpu,true,mode,true));
            std::cout<<"PASS overlap bits: backend="<<argv[2]<<" dense="<<mode.dense<<" taps="<<mode.taps<<" gpu_decode="<<mode.gpu_decode<<" reset/context/owned-output\n";
        }
        if(gpu)ggml_backend_free(gpu);return 0;
    }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(gpu)ggml_backend_free(gpu);return 1;}
}
