#include <bits/stdc++.h>
#define private public
#include "m31_engine.hpp"
#undef private
#include "m31_engine.cpp"
using namespace strata::m31;
int main(){try{
 int checks=0;auto check=[&](bool ok,const char *why){if(!ok)throw std::runtime_error(why);++checks;};
 Engine engine;auto &p=*engine.p_;p.ready=true;p.cfg.context=256;p.cfg.max_chunk=8;
 p.g.index_heads=4;p.g.block=128;p.g.embd=8;p.g.vocab=100;
 p.backend=ggml_backend_cpu_init();p.own_backend=true;if(!p.backend)throw std::runtime_error("CPU backend");
 p.state=ggml_init({128*1024,nullptr,true});if(!p.state)throw std::runtime_error("KV metadata");
 p.cache.resize(1);auto &kv=p.cache[0];
 kv.k=ggml_new_tensor_2d(p.state,GGML_TYPE_F16,8,256);kv.v=ggml_new_tensor_2d(p.state,GGML_TYPE_F16,8,256);kv.index=ggml_new_tensor_2d(p.state,GGML_TYPE_F16,8,256);
 p.state_buffer=ggml_backend_alloc_ctx_tensors_from_buft(p.state,ggml_backend_get_default_buffer_type(p.backend));if(!p.state_buffer)throw std::runtime_error("KV buffer");
 std::vector<uint8_t> original;for(auto *t:{kv.k,kv.v,kv.index}){
  std::vector<ggml_fp16_t> data(size_t(ggml_nelements(t)));for(size_t i=0;i<data.size();++i)data[i]=ggml_fp32_to_fp16(float(i%97)/16);
  ggml_backend_tensor_set(t,data.data(),0,ggml_nbytes(t));auto *bytes=reinterpret_cast<uint8_t*>(data.data());original.insert(original.end(),bytes,bytes+ggml_nbytes(t));
 }
 auto snapshot=[&](){std::vector<uint8_t> out;for(auto *t:{kv.k,kv.v,kv.index}){size_t first=out.size();out.resize(first+ggml_nbytes(t));ggml_backend_tensor_get(t,out.data()+first,0,ggml_nbytes(t));}return out;};
 auto graph=[&](){auto b=std::make_unique<Graph>(p.backend,p.cfg,48,nullptr,1024*1024,128);auto x=b->input(GGML_TYPE_F32,8,1);b->allocate({ggml_add(b->c,x,x)});return b;};
 p.dense_decode_graphs.resize(1);p.dense_decode_graphs[0]=std::make_unique<Engine::Impl::DenseGraph>();p.dense_decode_graphs[0]->graph=graph();
 p.shared_decode_graphs.resize(1);p.shared_decode_graphs[0]=std::make_unique<Engine::Impl::HeadGraph>();p.shared_decode_graphs[0]->graph=graph();p.head_decode_graph.graph=graph();
 p.pos=48;p.last_n=1;p.hidden={7.f};p.decode_chain_active=true;p.prefill_chain_active=true;
 p.captured_rows=1;p.captured_features={3.f};
 std::string error;
 check(engine.prefill_chunk_at(0,8,error)==8&&p.pos==48&&p.last_n==1&&p.hidden==std::vector<float>({7.f}),"pure plan at0 changed state");
 check(engine.prefill_chunk_at(48,8,error)==engine.prefill_chunk(8,error),"current-position planner differs");
 check(engine.prefill_chunk_at(255,8,error)==1&&p.pos==48,"future pure plan context cap");
 check(!engine.prefill_chunk_at(-1,8,error)&&p.pos==48&&!p.poisoned,"negative logical plan changed state");
 check(!engine.prefill_chunk_at(256,8,error)&&p.pos==48&&!p.poisoned,"end logical plan accepted");
 check(!engine.prefill_chunk_at(0,0,error)&&p.pos==48&&!p.poisoned,"zero request accepted");
 check(!engine.rewind(-1,error)&&!p.poisoned&&p.pos==48,"negative rewind changed state");
 check(!engine.rewind(49,error)&&!p.poisoned&&p.pos==48,"future rewind changed state");
 check(p.dense_decode_graphs[0]&&p.shared_decode_graphs[0]&&p.head_decode_graph.graph&&p.hidden==std::vector<float>({7.f}),"invalid rewind changed graph/hidden ownership");
 p.poisoned=true;check(!engine.rewind(32,error)&&p.poisoned&&p.pos==48,"poisoned rewind accepted");p.poisoned=false;
 check(engine.rewind(32,error),"valid rewind failed");
 check(p.pos==32&&p.last_n==0&&p.hidden.empty()&&!p.poisoned,"rewind state mismatch");
 check(!p.decode_chain_active&&!p.prefill_chain_active,"rewind preserved stale chain flags");
 check(p.captured_rows==0&&p.captured_features.empty(),"rewind retained stale draft features");
 check(!p.dense_decode_graphs[0]&&!p.shared_decode_graphs[0]&&!p.head_decode_graph.graph,"rewind retained stale graph/head captures");
 check(snapshot()==original,"rewind modified KV prefix or stale future bytes");
 std::vector<float> logits;check(!engine.logits(0,1,logits,error)&&!p.poisoned,"rewind left old head accessible");
 check(engine.prefill_chunk(8,error)==8,"rewound state cannot plan next chunk");
 check(engine.rewind(32,error)&&p.pos==32&&snapshot()==original,"equal-position rewind changed KV");
 check(engine.rewind(0,error)&&p.pos==0&&snapshot()==original,"zero rewind changed KV");
 check(!engine.rewind(1,error)&&!p.poisoned&&p.pos==0,"rewind advanced beyond valid prefix");
 Engine unready;check(!unready.rewind(0,error)&&!unready.p_->poisoned,"unready rewind accepted/poisoned");
 std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"scope\":\"actual rewind/pure planner CPU graph/KV owner lifecycle; not production GPU quality\"}\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
