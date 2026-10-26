// Direct GpuExperts F32 probe: tiny attention CUDA support is not needed.
#include "m31_gpu_experts.hpp"
#include "cache_seed_receipt.hpp"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace strata::m31;
static void check(bool value,const char * message) {if(!value)throw std::runtime_error(message);}
static void clear_seed_env() {
    for(const char * name:{"M31_GPU_SEED_HOT","M31_GPU_SEED_PATH","M31_GPU_SEED_IDENTITY","M31_GPU_SEED_SHA256",
                           "M31_GPU_SEED_RECEIPT","M31_GPU_SEED_RECEIPT_SHA256"})unsetenv(name);
}
static void seed_environment(const cache_seed::Receipt & r,const std::string & path,const std::string & sha) {
    setenv("M31_GPU_SEED_HOT",std::to_string(r.hot).c_str(),1);
    setenv("M31_GPU_SEED_PATH",r.rank_path.c_str(),1);
    setenv("M31_GPU_SEED_IDENTITY",cache_seed::digest_hex(r.expected.identity).c_str(),1);
    setenv("M31_GPU_SEED_SHA256",cache_seed::digest_hex(r.expected.file_sha256).c_str(),1);
    setenv("M31_GPU_SEED_RECEIPT",path.c_str(),1);
    setenv("M31_GPU_SEED_RECEIPT_SHA256",sha.c_str(),1);
}
static void identical(const std::vector<float> & a,const std::vector<float> & b) {
    check(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"seeded output differs from unseeded GPU");
}
int main(int argc,char ** argv) {
    ggml_backend_t backend=nullptr;
    try {
        check(argc==5,"usage: cache_seed_gpu_check TINY.gguf PREFLIGHT.seal SEAL_SHA256 REPORT.json");
        check(std::getenv("M31_CARD_SESSION") && std::string(std::getenv("M31_CARD_SESSION"))=="1","aictl runner required");
        const auto receipt=cache_seed::read_receipt(argv[2],cache_seed::hex_digest(argv[3]));
        strata::GgufModel model(strata::gguf_split_paths(argv[1]));
        const auto g=read_geometry(model);
        check(g.embd==32 && g.experts==4 && g.used==2 && receipt.slots==6 && receipt.hot==2,"small six-slot fixture expected");
        cache_seed::verify_current_model(receipt,model);
        clear_seed_env();unsetenv("M31_GPU_CPU_QUANTS");unsetenv("M31_GPU_PREFILL_TEST_FAIL_AFTER");
        backend=ggml_backend_cuda_init(0);check(backend,"CUDA0 init failed");
        {
            ExpertArena arena;arena.init(model,g,.01);
            // All roles are F32[32,32], 3*4096 bytes per slot; exactly six slots.
            const double budget=double(6*3*32*32*sizeof(float))/1073741824.0;
            unsetenv("M31_GPU_PREFILL_ASYNC");
            GpuExperts baseline;baseline.init(model,g,arena,backend,budget,7);
            check(baseline.stats().slot_count==6 && baseline.stats().seeded_slots==0,"default path is seeded or unexpected slot padding");
            setenv("M31_GPU_PREFILL_ASYNC","1",1);
            GpuExperts direct;direct.init(model,g,arena,backend,budget,7);
            direct.seed_hot(receipt.rank_path,receipt.expected,2);
            seed_environment(receipt,argv[2],argv[3]);
            GpuExperts via_env;via_env.init(model,g,arena,backend,budget,7);
            check(direct.stats().seeded_slots==2 && via_env.stats().seeded_slots==2,"cold seed count differs");
            check(direct.stats().occupied_slots==2 && via_env.stats().occupied_slots==2,"cold seed residency differs");
            const uint64_t seed_bytes=via_env.stats().upload_bytes;
            direct.reset_stats();via_env.reset_stats();
            check(direct.stats().seeded_slots==2 && via_env.stats().seeded_slots==2,"reset_stats lost hot pins");
            std::vector<float> x(size_t(g.embd)*7),weights(size_t(g.used)*7);
            std::vector<int32_t> ids(weights.size());
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.12f);
            for(int t=0;t<7;++t)for(int rank=0;rank<g.used;++rank){
                const auto i=size_t(t)*size_t(g.used)+size_t(rank);
                ids[i]=(t+rank)%4;weights[i]=rank?.7f:1.3f;
            }
            size_t batches=0,elements=0;
            for(int repeat=0;repeat<3;++repeat)for(int layer=int(g.dense_lead);layer<g.layers;++layer){
                const auto expected=baseline.forward(layer,7,x,ids,weights);
                identical(expected,direct.forward(layer,7,x,ids,weights));
                identical(expected,via_env.forward(layer,7,x,ids,weights));
                ++batches;elements+=2*expected.size();
            }
            check(via_env.stats().evictions>0,"seed protection not tested across eviction");
            const auto hits=via_env.stats().hits;
            identical(baseline.forward(int(g.dense_lead),7,x,ids,weights),via_env.forward(int(g.dense_lead),7,x,ids,weights));
            check(via_env.stats().hits-hits==2,"pinned seeds did not survive all-layer scans");
            const auto preserved_evictions=via_env.stats().evictions;
            const auto fail_after=via_env.stats().misses+1;
            setenv("M31_GPU_PREFILL_TEST_FAIL_AFTER",std::to_string(fail_after).c_str(),1);
            bool forward_failed=false;
            try{via_env.forward(2,7,x,ids,weights);}catch(const std::exception & e){forward_failed=std::string(e.what())=="M31 prefill injected upload failure";}
            check(forward_failed,"forward failure was not injected");
            check(via_env.stats().occupied_slots==0 && via_env.stats().seeded_slots==0,"recovery left stale hot metadata");
            unsetenv("M31_GPU_PREFILL_TEST_FAIL_AFTER");
            identical(baseline.forward(2,7,x,ids,weights),via_env.forward(2,7,x,ids,weights));
            setenv("M31_GPU_PREFILL_TEST_FAIL_AFTER","1",1);
            GpuExperts partial;
            bool partial_failed=false;
            try{partial.init(model,g,arena,backend,budget,7);}catch(const std::exception & e){partial_failed=std::string(e.what())=="M31 prefill injected upload failure";}
            check(partial_failed,"partial seed DMA failure was not injected");
            check(partial.stats().seeded_slots==0 && partial.stats().occupied_slots==0,"partial seed was published");
            bool terminal=false;
            try{partial.forward(1,7,x,ids,weights);}catch(const std::exception & e){terminal=std::string(e.what()).find("uninitialized backend")!=std::string::npos;}
            check(terminal,"failed seed cache remained usable");
            unsetenv("M31_GPU_PREFILL_TEST_FAIL_AFTER");clear_seed_env();
            std::ofstream out(argv[4]);
            out<<"{\"pass\":true,\"seed_api\":true,\"seed_env\":true,\"cold_seeds\":2,\"slots\":6,\"stream_slots\":4,"
                 "\"seed_upload_bytes\":"<<seed_bytes<<",\"bitwise_batches\":"<<batches<<",\"bitwise_elements\":"<<elements
               <<",\"evictions\":"<<preserved_evictions<<",\"hot_hits_after_scan\":2,\"forward_failure_recovered\":true,"
                 "\"partial_seed_failure_terminal\":true}\n";
            check(bool(out),"cannot write seed probe report");
        }
        ggml_backend_free(backend);return 0;
    }catch(const std::exception & e){std::cerr<<"seed GPU probe: "<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}
}
