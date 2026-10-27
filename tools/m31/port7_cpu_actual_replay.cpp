#include "cpu_batched_verify.hpp"
#include "cpu_fused_decode.hpp"
#include "m31_geometry.hpp"
#include <chrono>
#include <cstdio>
#include <memory>
#include <fstream>
#include <map>
using namespace strata::m31;
using Clock=std::chrono::steady_clock;
void ensure(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
struct Layer{int index;std::array<ggml_tensor *,3>w{};std::array<std::vector<float>,3>s;std::array<bool,3>has{};size_t bytes_per_expert=0;};
int main(int argc,char ** argv) {
 try {
    std::setvbuf(stdout,nullptr,_IOLBF,0);ensure(argc>=2,"usage: port7_cpu_actual_replay MODEL THREADS UNUSED routes.bin");
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
      ensure(argc>=5,"actual replay requires routes.bin as argument4");
      std::ifstream file(argv[4],std::ios::binary);ensure(bool(file),"route payload open");char magic[8];file.read(magic,8);ensure(!std::memcmp(magic,"M31RPL01",8),"route payload magic");
      auto u32=[&](){uint32_t v=0;file.read(reinterpret_cast<char*>(&v),4);ensure(bool(file),"truncated route payload");return v;};
      struct Route{std::array<int32_t,4>ids;std::array<float,4>weights;};
      const auto records=u32();ensure(records>0&&records<1000000,"route record count");std::map<std::pair<int,int>,std::vector<Route>> routes;int cases=0;
      for(uint32_t r=0;r<records;++r){const auto c=u32(),step=u32(),layer=u32();ensure(c<1000&&layer<60,"route case/layer range");auto & v=routes[{int(c),int(layer)}];ensure(step==v.size(),"non-contiguous route order");Route row;file.read(reinterpret_cast<char*>(row.ids.data()),16);file.read(reinterpret_cast<char*>(row.weights.data()),16);ensure(bool(file),"truncated route row");for(size_t i=0;i<4;++i){ensure(row.ids[i]>=0&&row.ids[i]<g.experts&&std::isfinite(row.weights[i]),"invalid route id/weight");for(size_t j=0;j<i;++j)ensure(row.ids[i]!=row.ids[j],"duplicate route ID");}v.push_back(row);cases=std::max(cases,int(c+1));}
      ensure(file.peek()==std::char_traits<char>::eof(),"trailing route data");
      for(int c=0;c<cases;++c)for(const auto & l:layers)ensure(routes.count({c,l.index})&&routes.at({c,l.index}).size()==routes.at({c,layers[0].index}).size(),"incomplete case/layer route sequence");
      std::vector<float>x,weights,actual,expected,one_x,one_weights,one_output;std::vector<int32_t> ids,one_ids;
      auto fill=[&](int tokens,int c,int first,int layer){x.resize(size_t(tokens)*g.embd);ids.resize(size_t(tokens)*4);weights.resize(ids.size());const auto & sequence=routes.at({c,layer});for(int token=0;token<tokens;++token){for(int i=0;i<g.embd;++i)x[size_t(token)*g.embd+i]=std::sin(float(i)*.017f+float(first+token)*.731f+float(c)*.13f);const auto & r=sequence.at(size_t(first+token));std::copy(r.ids.begin(),r.ids.end(),ids.begin()+size_t(token)*4);std::copy(r.weights.begin(),r.weights.end(),weights.begin()+size_t(token)*4);}};
      auto run_batch=[&](const Layer & l){batch.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,x,ids,weights,actual);};
      auto run_serial=[&](const Layer & l,int tokens){expected.clear();for(int token=0;token<tokens;++token){one_x.assign(x.begin()+size_t(token)*g.embd,x.begin()+size_t(token+1)*g.embd);one_ids.assign(ids.begin()+size_t(token)*4,ids.begin()+size_t(token+1)*4);one_weights.assign(weights.begin()+size_t(token)*4,weights.begin()+size_t(token+1)*4);serial.forward(l.w[0],l.w[1],l.w[2],l.has[0]?l.s[0].data():nullptr,l.has[1]?l.s[1].data():nullptr,l.has[2]?l.s[2].data():nullptr,one_x,one_ids,one_weights,one_output);expected.insert(expected.end(),one_output.begin(),one_output.end());}};
      // Prefault all expert weights through actual trait dots, outside both timers.
      for(int round=0;round<128;++round)for(const auto & l:layers){fill(1,0,0,l.index);for(int rank=0;rank<4;++rank)ids[rank]=(round+rank*29)%128;run_batch(l);}
      for(int tokens:{2,4,8}) {
        double serial_s=0,batch_s=0;uint64_t serial_bytes=0,unique_bytes=0,unique_expert_count=0;size_t windows=0,token_rows=0;
        std::array<double,5> layer_serial{},layer_batch{};std::array<uint64_t,5> layer_serial_bytes{},layer_unique_bytes{},layer_unique_experts{};std::array<size_t,5> layer_windows{};
        for(int c=0;c<cases;++c){const int count=int(routes.at({c,layers[0].index}).size());for(int first=0;first+tokens<=count;first+=tokens)for(size_t j=0;j<layers.size();++j){const auto & l=layers[j];fill(tokens,c,first,l.index);auto ts=[&]{auto t=Clock::now();run_serial(l,tokens);double s=std::chrono::duration<double>(Clock::now()-t).count();serial_s+=s;layer_serial[j]+=s;};auto tb=[&]{auto t=Clock::now();run_batch(l);double s=std::chrono::duration<double>(Clock::now()-t).count();batch_s+=s;layer_batch[j]+=s;};if(windows%2){tb();ts();}else{ts();tb();}ensure(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"actual-route batch differs bitwise");const uint64_t baseline=uint64_t(tokens)*4*l.bytes_per_expert,unique=uint64_t(batch.unique_experts())*l.bytes_per_expert;serial_bytes+=baseline;unique_bytes+=unique;unique_expert_count+=batch.unique_experts();++windows;token_rows+=tokens;layer_serial_bytes[j]+=baseline;layer_unique_bytes[j]+=unique;layer_unique_experts[j]+=batch.unique_experts();++layer_windows[j];}}
        for(size_t j=0;j<layers.size();++j)std::printf("{\"kind\":\"layer\",\"tokens\":%d,\"layer\":%d,\"windows\":%zu,\"serial_s\":%.9f,\"grouped_s\":%.9f,\"serial_weight_bytes\":%llu,\"unique_weight_bytes\":%llu,\"unique_experts\":%llu,\"speedup\":%.6f,\"bitwise\":true}\n",tokens,layers[j].index,layer_windows[j],layer_serial[j],layer_batch[j],(unsigned long long)layer_serial_bytes[j],(unsigned long long)layer_unique_bytes[j],(unsigned long long)layer_unique_experts[j],layer_serial[j]/layer_batch[j]);
        std::printf("{\"kind\":\"aggregate\",\"tokens\":%d,\"cases\":%d,\"windows\":%zu,\"token_rows\":%zu,\"serial_s\":%.9f,\"grouped_s\":%.9f,\"serial_weight_bytes\":%llu,\"unique_weight_bytes\":%llu,\"unique_experts\":%llu,\"weight_reuse\":%.6f,\"speedup\":%.6f,\"bitwise\":true,\"synthetic_activations\":true}\n",tokens,cases,windows,token_rows,serial_s,batch_s,(unsigned long long)serial_bytes,(unsigned long long)unique_bytes,(unsigned long long)unique_expert_count,double(serial_bytes)/unique_bytes,serial_s/batch_s);
      }
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);ggml_free(context);
 }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
