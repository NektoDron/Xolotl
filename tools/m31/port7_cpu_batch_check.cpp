#include "cpu_batched_verify.hpp"
#include "cpu_fused_decode.hpp"
#include "m31_geometry.hpp"
#include <chrono>
#include <cstdio>
#include <memory>
using namespace strata::m31;
using Clock=std::chrono::steady_clock;
void ensure(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
struct Layer{int index;std::array<ggml_tensor *,3>w{};std::array<std::vector<float>,3>s;std::array<bool,3>has{};size_t bytes_per_expert=0;};
int main(int argc,char ** argv) {
 try {
    std::setvbuf(stdout,nullptr,_IOLBF,0);ensure(argc>=2,"usage: port7_cpu_batch_check MODEL [threads=80] [repeats=16]");
    const int threads=argc>2?std::stoi(argv[2]):80,repeats=argc>3?std::stoi(argv[3]):16;
    ensure(threads>0&&threads<=96&&repeats>=0&&repeats<=1024,"invalid threads/repeats");
    strata::GgufModel model(strata::gguf_split_paths(argv[1]));const auto g=read_geometry(model);
    auto * context=ggml_init({4*1024*1024,nullptr,true});ensure(context,"context");
    std::vector<Layer> layers;const char * roles[]={"gate","up","down"};
    for(int index:{3,10,52,58,59}){Layer l;l.index=index;for(int role=0;role<3;++role){const auto name=layer_name(index,std::string("ffn_")+roles[role]+"_exps.weight");size_t shard=0;auto info=model.find(name,&shard);ensure(info,"weight");auto * w=ggml_new_tensor_3d(context,ggml_type(info->type),info->shape[0],info->shape[1],info->shape[2]);w->data=const_cast<uint8_t *>(model.shard(shard).tensor_data(*info));l.w[role]=w;l.bytes_per_expert+=w->nb[2];const auto sn=name.substr(0,name.size()-6)+"scale";size_t ss=0;if(auto scale=model.find(sn,&ss)){l.has[role]=true;l.s[role].resize(g.experts);std::memcpy(l.s[role].data(),model.shard(ss).tensor_data(*scale),g.experts*sizeof(float));}}layers.push_back(std::move(l));}
    auto cpu=ggml_backend_cpu_init();ensure(cpu,"CPU backend");ggml_backend_cpu_set_n_threads(cpu,threads);auto params=ggml_threadpool_params_default(threads);apply_cpu_pool_environment(params);auto pool=ggml_threadpool_new(&params);ensure(pool,"pool");ggml_backend_cpu_set_threadpool(cpu,pool);
    setenv("M31_CPU_FUSED_PAIRED_QUANT","0",1);
    {
      CpuFusedDecode serial(cpu,pool,unsigned(threads));CpuBatchedVerify batch(cpu,pool,unsigned(threads));
      std::vector<float>x,weights,actual,expected,one_x,one_weights,one_output;std::vector<int32_t> ids,one_ids;
      auto fill=[&](int tokens,int used,int pattern,int round){x.resize(size_t(tokens)*g.embd);ids.resize(size_t(tokens)*used);weights.resize(ids.size());for(int token=0;token<tokens;++token){for(int i=0;i<g.embd;++i)x[size_t(token)*g.embd+i]=std::sin(float(i)*.017f+float(token)*.731f);for(int rank=0;rank<used;++rank){const int shift=pattern==0?0:pattern==1?token:token*used;ids[size_t(token)*used+rank]=(round*7+shift+rank*(pattern==1?1:29))%int(g.experts);weights[size_t(token)*used+rank]=1.f/float(used+token+rank+1);}}};
      auto run_batch=[&](const Layer & l){batch.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,x,ids,weights,actual);};
      auto run_serial=[&](const Layer & l,int tokens,int used){expected.clear();for(int token=0;token<tokens;++token){one_x.assign(x.begin()+size_t(token)*g.embd,x.begin()+size_t(token+1)*g.embd);one_ids.assign(ids.begin()+size_t(token)*used,ids.begin()+size_t(token+1)*used);one_weights.assign(weights.begin()+size_t(token)*used,weights.begin()+size_t(token+1)*used);serial.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,one_x,one_ids,one_weights,one_output);expected.insert(expected.end(),one_output.begin(),one_output.end());}};
      int batches_checked=0,token_rows_checked=0;
      for(auto & l:layers)for(int tokens:{1,2,4,8})for(int used:{1,2,4})for(int pattern=0;pattern<3;++pattern){fill(tokens,used,pattern,13);run_serial(l,tokens,used);run_batch(l);ensure(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"batched CPU differs bitwise from v8 per-token CPU fused");++batches_checked;token_rows_checked+=tokens;}
      std::printf("{\"kind\":\"bitproof\",\"pass\":true,\"batches\":%d,\"token_rows\":%d,\"layers\":5,\"batch_sizes\":[1,2,4,8],\"route_counts\":[1,2,4],\"routing_patterns\":[\"same\",\"partial\",\"disjoint\"],\"threads\":%d}\n",batches_checked,token_rows_checked,threads);
      if(repeats)for(int tokens:{2,4,8})for(int pattern=0;pattern<3;++pattern){
        // Warm every expert before either timed arm; no mmap fault advantage.
        for(int round=0;round<int(g.experts);++round){fill(tokens,4,pattern,round);for(auto & l:layers)run_batch(l);}
        double serial_s=0,batch_s=0;size_t serial_bytes=0,unique_bytes=0;
        // Alternate the ordering each round; both arms visit identical routes.
        for(int round=0;round<repeats;++round){fill(tokens,4,pattern,round);for(auto & l:layers){auto timed_serial=[&]{auto t=Clock::now();run_serial(l,tokens,4);serial_s+=std::chrono::duration<double>(Clock::now()-t).count();};auto timed_batch=[&]{auto t=Clock::now();run_batch(l);batch_s+=std::chrono::duration<double>(Clock::now()-t).count();};if(round%2){timed_batch();timed_serial();}else{timed_serial();timed_batch();}serial_bytes+=size_t(tokens)*4*l.bytes_per_expert;unique_bytes+=batch.unique_experts()*l.bytes_per_expert;ensure(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"timed batch differs bitwise");}}
        const int calls=repeats*int(layers.size());std::printf("{\"kind\":\"timing\",\"tokens\":%d,\"pattern\":%d,\"calls\":%d,\"serial_ms_per_batch\":%.6f,\"batch_ms\":%.6f,\"speedup\":%.6f,\"assignment_GBps\":%.6f,\"unique_weight_GBps\":%.6f,\"weight_reuse\":%.6f,\"bitwise\":true}\n",tokens,pattern,calls,serial_s*1000/calls,batch_s*1000/calls,serial_s/batch_s,serial_bytes/batch_s/1e9,unique_bytes/batch_s/1e9,double(serial_bytes)/unique_bytes);
      }
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);ggml_free(context);
 }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
