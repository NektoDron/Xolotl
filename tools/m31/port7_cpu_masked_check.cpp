#include "cpu_batched_verify.hpp"
#include "cpu_fused_decode.hpp"
#include "m31_geometry.hpp"
#include <chrono>
#include <cstdio>
#include <memory>
#include <bit>
using namespace strata::m31;
using Clock=std::chrono::steady_clock;
void ensure(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
struct Layer{int index;std::array<ggml_tensor *,3>w{};std::array<std::vector<float>,3>s;std::array<bool,3>has{};size_t bytes_per_expert=0;};
void merge_original_rank(int tokens,int used,size_t embd,const std::vector<uint8_t> & cpu_mask,
 const std::vector<float> & cpu,const std::vector<float> & gpu,std::vector<float> & output){
 output.resize(size_t(tokens)*embd);for(int token=0;token<tokens;++token)for(size_t row=0;row<embd;++row){
  auto term=[&](int rank){const size_t a=size_t(token)*size_t(used)+size_t(rank);return (cpu_mask[a]?cpu:gpu)[a*embd+row];};
  float sum=term(0);for(int rank=1;rank<used;++rank)sum+=term(rank);output[size_t(token)*embd+row]=sum;
 }
}
int main(int argc,char ** argv) {
 try {
    std::setvbuf(stdout,nullptr,_IOLBF,0);ensure(argc>=2,"usage: port7_cpu_masked_check MODEL [threads=80] [unused=0]");
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
      std::vector<float>x,weights,actual,expected,expected_terms,one_x,one_weights,one_output;std::vector<int32_t> ids,one_ids;
      auto fill=[&](int tokens,int used,int pattern){x.resize(size_t(tokens)*g.embd);ids.resize(size_t(tokens)*used);weights.resize(ids.size());for(int token=0;token<tokens;++token){for(int i=0;i<g.embd;++i)x[size_t(token)*g.embd+i]=std::sin(float(i)*.017f+float(token)*.731f);for(int rank=0;rank<used;++rank){const int shift=pattern==0?0:pattern==1?token:token*used;ids[size_t(token)*used+rank]=(13*7+shift+rank*(pattern==1?1:29))%int(g.experts);weights[size_t(token)*used+rank]=1.f/float(used+token+rank+1);}}};
      auto run_serial=[&](const Layer & l,int tokens,int used){expected.clear();expected_terms.clear();for(int token=0;token<tokens;++token){one_x.assign(x.begin()+size_t(token)*g.embd,x.begin()+size_t(token+1)*g.embd);one_ids.assign(ids.begin()+size_t(token)*used,ids.begin()+size_t(token+1)*used);one_weights.assign(weights.begin()+size_t(token)*used,weights.begin()+size_t(token+1)*used);serial.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,one_x,one_ids,one_weights,one_output);expected.insert(expected.end(),one_output.begin(),one_output.end());const auto & terms=serial.contributions();expected_terms.insert(expected_terms.end(),terms.begin(),terms.end());}};
      auto run_unmasked=[&](CpuBatchedVerify & target,const Layer & l){target.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,x,ids,weights,actual);};
      auto run_masked=[&](CpuBatchedVerify & target,const Layer & l,const std::vector<uint8_t> & mask){target.forward_masked(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,x,ids,weights,mask,actual);};
      // An armed real backend abort callback proves all-hit bypasses graph compute.
      fill(1,4,0);bool abort_control=false;
      {CpuBatchedVerify probe(cpu,pool,unsigned(threads));std::vector<uint8_t> allhit(4,0);ggml_backend_cpu_set_abort_callback(cpu,[](void*){return true;},nullptr);run_masked(probe,layers[0],allhit);ensure(actual.size()==size_t(g.embd)*4,"allhit term size");for(float value:actual)ensure(std::bit_cast<uint32_t>(value)==0,"allhit term nonzero");try{run_unmasked(probe,layers[0]);}catch(const std::runtime_error &){abort_control=true;}ggml_backend_cpu_set_abort_callback(cpu,nullptr,nullptr);ensure(abort_control,"abort callback control did not trap an actual graph");run_serial(layers[0],1,4);run_unmasked(probe,layers[0]);ensure(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),"unmasked recovery after allhit/abort changed arithmetic");}
      uint64_t batches=0,tokenrows=0,assignmentrows=0,unmasked_transitions=0;
      for(int mode:{0,2}){setenv("M31_CPU_BATCH_PAIR",std::to_string(mode).c_str(),1);CpuBatchedVerify masked(cpu,pool,unsigned(threads));
       for(auto & l:layers)for(int tokens:{1,2,4,8})for(int used:{1,2,4})for(int pattern=0;pattern<3;++pattern){fill(tokens,used,pattern);run_serial(l,tokens,used);
        for(int maskcase=0;maskcase<8;++maskcase){std::vector<uint8_t> mask(ids.size());for(size_t a=0;a<mask.size();++a){const size_t token=a/size_t(used),rank=a%size_t(used);mask[a]=maskcase==0?1:maskcase==1?0:maskcase==2?a%2:maskcase==3?a==0:maskcase==4?rank+1!=size_t(used):maskcase==5?token!=0:maskcase==6?ids[a]%3!=0:(token+rank)%3==0;}run_masked(masked,l,mask);ensure(actual.size()==expected_terms.size(),"masked terms geometry");
         for(size_t a=0;a<mask.size();++a){const size_t offset=a*size_t(g.embd);if(mask[a])ensure(!std::memcmp(actual.data()+offset,expected_terms.data()+offset,size_t(g.embd)*sizeof(float)),"CPU miss term differs bitwise");else for(size_t row=0;row<size_t(g.embd);++row)ensure(std::bit_cast<uint32_t>(actual[offset+row])==0,"GPU hit leaked a CPU term");}
         std::vector<float> merged;merge_original_rank(tokens,used,size_t(g.embd),mask,actual,expected_terms,merged);ensure(!std::memcmp(merged.data(),expected.data(),merged.size()*sizeof(float)),"original-rank CPU/GPU merge differs bitwise");++batches;tokenrows+=tokens;assignmentrows+=ids.size();
         run_unmasked(masked,l);ensure(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"default unmasked path changed after a mask");++unmasked_transitions;
        }
       }
      }
      // Adversarial reduction examples: do not start with+0 or pre-sum subsets.
      float negative_zero=-0.f;std::vector<float> zero_terms(4,negative_zero),adversarial;std::vector<uint8_t> mixed_mask{1,0,1,0};merge_original_rank(1,4,1,mixed_mask,zero_terms,zero_terms,adversarial);ensure(std::signbit(adversarial[0]),"signed-zero original-rank sum changed");
      const float terms[]={1e20f,-1e20f,3.25f,0.f};std::vector<float> adversarial_terms(terms,terms+4);merge_original_rank(1,4,1,mixed_mask,adversarial_terms,adversarial_terms,adversarial);const float sequential=adversarial[0];const float wrong=(terms[0]+terms[2])+(terms[1]+terms[3]);ensure(sequential==3.25f&&wrong!=sequential,"non-associative reduction fixture ineffective");
      std::printf("{\"pass\":true,\"masked_batches\":%llu,\"token_rows\":%llu,\"weighted_term_rows\":%llu,\"unmasked_transitions\":%llu,\"allhit_bypasses_abort_armed_graph\":true,\"pair_modes\":[0,2],\"mask_patterns\":8,\"batch_sizes\":[1,2,4,8],\"route_counts\":[1,2,4],\"original_rank_merge\":true,\"default_unmasked_bitwise\":true,\"scope\":\"CPU component only; GPU terms supplied by independent stock CPU reference, no actual GPU overlap/model quality claim\"}\n",(unsigned long long)batches,(unsigned long long)tokenrows,(unsigned long long)assignmentrows,(unsigned long long)unmasked_transitions);
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);ggml_free(context);
 }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
