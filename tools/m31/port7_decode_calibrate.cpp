// Actual autoregressive greedy routing. Separate tool; no production observer/logging.
#include "m31_engine.hpp"
#include "m31_run.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include "json.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
using json=nlohmann::json;
using namespace strata::m31;
int main(int argc,char **argv){
 ggml_backend_t backend=nullptr;
 try{
  const auto args=run::parse(argc,argv);
  if(args.taps!="none"||args.oracle_op_offload||args.decode_tail<=0)throw std::runtime_error("requires --decode-tail GENERATED_TOKENS, --taps none, no oracle placement");
  const auto inputs=run::cases(args);
  if(!run::fs::create_directory(args.out))throw std::runtime_error("fresh output directory required");
  std::ofstream trace(run::fs::path(args.out)/"routes.jsonl");
  if(!trace)throw std::runtime_error("cannot create route trace");
  if(args.backend=="cuda"){
#ifdef M31_HAVE_CUDA
   backend=ggml_backend_cuda_init(0);if(!backend)throw std::runtime_error("cannot initialize GPU0");
#else
   throw std::runtime_error("built without CUDA");
#endif
  }
  {
   Config cfg;cfg.backend=backend;cfg.threads=args.threads;cfg.context=args.context;cfg.max_chunk=args.chunk;
   cfg.arena_gib=args.arena_gib;cfg.gpu_experts_gib=args.gpu_experts_gib;cfg.gpu_experts_min_tokens=args.gpu_experts_min_tokens;
   cfg.reuse_allocations=args.reuse_allocations;cfg.persistent_cpu_pool=args.persistent_cpu_pool;cfg.cpu_residual=args.cpu_residual;
   cfg.cpu_decode_cache=args.cpu_decode_cache;cfg.dense_decode_cache=args.dense_decode_cache;cfg.profile=args.profile;
   std::string name,phase;int64_t position=0;int generated=0;
   cfg.route_observer=[&](int layer,int n,const std::vector<int32_t>&ids,const std::vector<float>&weights){
    if(phase=="decode")trace<<json{{"kind","routes"},{"phase",phase},{"case",name},{"position",position},{"generated_index",generated},{"layer",layer},{"n",n},{"ids",ids},{"weights",weights}}.dump()<<'\n';
    if(!trace)throw std::runtime_error("route trace write failed");
   };
   Engine engine;std::string error;if(!engine.init(args.model,cfg,error))throw std::runtime_error(error);
   const auto &g=engine.geometry();
   trace<<json{{"kind","header"},{"format","m31-actual-decode-routing-v1"},{"model",run::fs::absolute(args.model).string()},{"backend",args.backend},{"layers",g.layers},{"dense_lead",g.dense_lead},{"experts",g.experts},{"used",g.used},{"generated_per_case",args.decode_tail},{"sampling","argmax ignore_eos"},{"notes","Only actual autoregressive n=1 forward routes are ranked. First generated token uses prompt head and has no decode forward."}}.dump()<<'\n';
   for(const auto &input:inputs){
    const auto ids=run::read_ids(input);if(ids.size()+size_t(args.decode_tail)>size_t(args.context))throw std::runtime_error("prompt+decode exceeds context");
    name=input.stem().string();phase="prefill";engine.reset();
    trace<<json{{"kind","case"},{"case",name},{"input",run::fs::absolute(input).string()},{"prompt_tokens",ids.size()}}.dump()<<'\n';
    int last_n=0;for(position=0;position<int64_t(ids.size());){
     const int n=engine.prefill_chunk(std::min(args.chunk,int(ids.size()-size_t(position))),error);if(n<=0)throw std::runtime_error(error);
     if(!engine.forward(ids.data()+position,n,error))throw std::runtime_error(error);position+=n;last_n=n;
    }
    std::vector<float> logits;if(!engine.logits(last_n-1,1,logits,error))throw std::runtime_error(error);
    std::vector<int32_t> generated_ids;generated_ids.reserve(args.decode_tail);
    const auto t0=std::chrono::steady_clock::now();phase="decode";
    for(generated=0;generated<args.decode_tail;++generated){
     const auto best=std::max_element(logits.begin(),logits.end());if(best==logits.end()||!std::isfinite(*best))throw std::runtime_error("invalid logits");
     const int32_t token=int32_t(best-logits.begin());generated_ids.push_back(token);
     if(generated+1<args.decode_tail){position=engine.position();if(!engine.forward(&token,1,error)||!engine.logits(0,1,logits,error))throw std::runtime_error(error);}
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    trace<<json{{"kind","generated"},{"case",name},{"ids",generated_ids},{"seconds",seconds},{"actual_decode_forwards",args.decode_tail-1}}.dump()<<'\n';trace.flush();
    std::ofstream token_file(run::fs::path(args.out)/(name+".generated.i32"),std::ios::binary);token_file.write(reinterpret_cast<const char *>(generated_ids.data()),std::streamsize(generated_ids.size()*sizeof(int32_t)));
    if(!token_file)throw std::runtime_error("generated token write failed");
    std::cerr<<"actual decode calibrated "<<name<<" generated="<<generated_ids.size()<<" seconds="<<seconds<<'\n';
   }
  }
  if(backend)ggml_backend_free(backend);return 0;
 }catch(const std::exception&e){std::cerr<<"port7_decode_calibrate: "<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}
}
