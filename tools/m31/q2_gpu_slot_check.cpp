// Real GGUF native CUDA FFN: compact stock tensors versus remapped cache-slot views.
#include "m31_geometry.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace strata::m31;
using Json=nlohmann::json;
static void ck(bool b,const std::string&m){if(!b)throw std::runtime_error(m);}
static size_t align_up(size_t n,size_t a){return (n+a-1)/a*a;}
struct Context{ggml_context*p=ggml_init({8*1024*1024,nullptr,true});~Context(){if(p)ggml_free(p);}};
struct Buffer{ggml_backend_buffer_t p=nullptr;~Buffer(){if(p)ggml_backend_buffer_free(p);}};
struct Allocator{ggml_gallocr_t p=nullptr;~Allocator(){if(p)ggml_gallocr_free(p);}};
struct Weights{
 Context c; Buffer b; std::array<ggml_tensor*,3>w{};
};
struct Role{ggml_type type;int64_t columns,rows;size_t bytes,allocated,offset;const uint8_t*source;};
static std::vector<std::vector<float>> compute(ggml_backend_t backend,
    const std::array<ggml_tensor*,3>&w,int n,const std::array<int32_t,4>&map){
 Context c;ck(c.p,"graph context");
 auto*graph=ggml_new_graph_custom(c.p,256,false);
 auto*x=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,6144,1,n);
 auto*ids=ggml_new_tensor_2d(c.p,GGML_TYPE_I32,4,n);
 auto*rw=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,1,4,n);
 auto*shared=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,6144,n);
 auto*residual=ggml_dup_tensor(c.p,shared);
 for(auto*t:{x,ids,rw,shared,residual})ggml_set_input(t);
 auto*gate=ggml_mul_mat_id(c.p,w[0],x,ids);
 auto*up=ggml_mul_mat_id(c.p,w[1],x,ids);
 auto*activation=ggml_swiglu_oai(c.p,gate,up,1.702f,7.f);
 auto*down=ggml_mul_mat_id(c.p,w[2],activation,ids);
 auto*weighted=ggml_mul(c.p,down,rw);
 ggml_tensor*sum=nullptr;
 for(int rank=0;rank<4;++rank){
  auto*term=ggml_view_2d(c.p,weighted,6144,n,weighted->nb[2],size_t(rank)*weighted->nb[1]);
  sum=sum?ggml_add(c.p,sum,term):term;
 }
 auto*ff=ggml_add(c.p,ggml_cont(c.p,sum),shared);
 auto*result=ggml_add(c.p,ff,residual);
 std::array<ggml_tensor*,4>outputs{gate,up,down,result};
 for(auto*t:outputs)ggml_set_output(t);
 ggml_build_forward_expand(graph,result);
 const auto dev=ggml_backend_get_device(backend);
 for(int i=0;i<ggml_graph_n_nodes(graph);++i)
  ck(ggml_backend_dev_supports_op(dev,ggml_graph_node(graph,i)),"unsupported native GPU op");
 Allocator a;a.p=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
 ck(a.p&&ggml_gallocr_alloc_graph(a.p,graph),"graph allocation");
 std::vector<float>xx(size_t(n)*6144),ww(size_t(n)*4),ss(xx.size()),rr(xx.size());
 std::vector<int32_t>ii(size_t(n)*4);
 for(size_t i=0;i<xx.size();++i){
  xx[i]=std::sin(float(i%16381)*.013f)*.75f+std::cos(float(i%8191)*.021f)*.25f;
  ss[i]=float(int(i%97)-48)*.0001f;rr[i]=float(int(i%67)-33)*.001f;
 }
 for(int token=0;token<n;++token)for(int rank=0;rank<4;++rank){
  ii[size_t(token)*4+rank]=map[size_t((rank+token)%4)];
  ww[size_t(token)*4+rank]=float(rank+1)*.07f;
 }
 ggml_backend_tensor_set(x,xx.data(),0,xx.size()*4);
 ggml_backend_tensor_set(ids,ii.data(),0,ii.size()*4);
 ggml_backend_tensor_set(rw,ww.data(),0,ww.size()*4);
 ggml_backend_tensor_set(shared,ss.data(),0,ss.size()*4);
 ggml_backend_tensor_set(residual,rr.data(),0,rr.size()*4);
 ck(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
 ggml_backend_synchronize(backend);
 std::vector<std::vector<float>>out;
 for(auto*t:outputs){out.emplace_back(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,out.back().data(),0,out.back().size()*4);}
 return out;
}
int main(int argc,char**argv){
 ggml_backend_t backend=nullptr;
 try{
  ck(argc==3,"usage: q2_gpu_slot_check MODEL REPORT.json");
  ck(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","aictl card session required");
  ck(std::getenv("CUDA_VISIBLE_DEVICES")&&std::string(std::getenv("CUDA_VISIBLE_DEVICES"))=="0","GPU0 visibility required");
  strata::GgufModel model(strata::gguf_split_paths(argv[1]));const auto g=read_geometry(model);
  ck(g.embd==6144&&g.expert_ff==3072&&g.experts==128&&g.used==4,"geometry");
  backend=ggml_backend_cuda_init(0);ck(backend,"CUDA backend");
  const auto buft=ggml_backend_get_default_buffer_type(backend);
  const size_t alignment=ggml_backend_buft_get_alignment(buft);
  size_t multiple=alignment,maximum_span=0;std::array<std::array<Role,3>,60>roles{};
  const std::array<const char*,3>names{"gate","up","down"};
  Context metadata;
  for(int layer=3;layer<60;++layer){size_t span=0;
   for(size_t r=0;r<3;++r){auto&v=roles[size_t(layer)][r];size_t shard=0;
    const auto*t=model.find(layer_name(layer,std::string("ffn_")+names[r]+"_exps.weight"),&shard);ck(t,"weights missing");
    v.type=ggml_type(t->type);v.columns=r==2?3072:6144;v.rows=r==2?6144:3072;
    auto*w=ggml_new_tensor_2d(metadata.p,v.type,v.columns,v.rows);
    v.bytes=ggml_nbytes(w);v.allocated=ggml_backend_buft_get_alloc_size(buft,w);
    v.offset=align_up(span,alignment);span=v.offset+align_up(v.allocated,alignment);
    v.source=model.shard(shard).tensor_data(*t);multiple=std::lcm(multiple,ggml_type_size(v.type));
   }maximum_span=std::max(maximum_span,span);
  }
  const size_t stride=align_up(maximum_span,multiple);
  const std::array<int32_t,4>selected{0,17,67,127},permutation{2,0,3,1},identity{0,1,2,3};
  Json cases=Json::array();size_t total=0;
  for(int layer:{3,52,59}){
   const auto&role=roles[size_t(layer)];Weights compact,padded;
   for(size_t r=0;r<3;++r)compact.w[r]=ggml_new_tensor_3d(compact.c.p,role[r].type,role[r].columns,role[r].rows,4);
   compact.b.p=ggml_backend_alloc_ctx_tensors_from_buft(compact.c.p,buft);ck(compact.b.p,"compact allocation");
   ggml_backend_buffer_set_usage(compact.b.p,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
   auto*root=ggml_new_tensor_1d(padded.c.p,GGML_TYPE_I8,int64_t(4*stride));
   padded.b.p=ggml_backend_alloc_ctx_tensors_from_buft(padded.c.p,buft);ck(padded.b.p,"padded allocation");
   ggml_backend_buffer_set_usage(padded.b.p,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
   for(size_t r=0;r<3;++r){
    ck(stride%ggml_type_size(role[r].type)==0,"block stride");
    auto*w=ggml_new_tensor_3d(padded.c.p,role[r].type,role[r].columns,role[r].rows,4);
    w->nb[2]=stride;w->nb[3]=4*stride;w->view_src=root;w->view_offs=role[r].offset;
    ck(ggml_backend_view_init(w)==GGML_STATUS_SUCCESS,"slot view");padded.w[r]=w;
    for(size_t logical=0;logical<4;++logical){
     const auto*src=role[r].source+size_t(selected[logical])*role[r].bytes;
     ggml_backend_tensor_set(compact.w[r],src,logical*role[r].bytes,role[r].bytes);
     ggml_backend_tensor_set(root,src,size_t(permutation[logical])*stride+role[r].offset,role[r].bytes);
    }
   }
   for(int n:{1,32,256,1024}){
    const auto ref=compute(backend,compact.w,n,identity);
    const auto got=compute(backend,padded.w,n,permutation);
    Json checks=Json::array();
    for(size_t r=0;r<ref.size();++r){ck(ref[r].size()==got[r].size(),"output shape");
     size_t mismatches=0;double maxabs=0;
     for(size_t i=0;i<ref[r].size();++i){ck(std::isfinite(ref[r][i])&&std::isfinite(got[r][i]),"nonfinite output");
      mismatches+=std::memcmp(&ref[r][i],&got[r][i],4)!=0;maxabs=std::max(maxabs,double(std::abs(ref[r][i]-got[r][i])));}
     checks.push_back({{"stage",r==0?"gate":r==1?"up":r==2?"down":"final"},{"values",ref[r].size()},{"bit_mismatches",mismatches},{"max_abs",maxabs}});
     ck(mismatches==0,"native slot remap bit mismatch layer"+std::to_string(layer)+" n"+std::to_string(n));total+=ref[r].size();
    }
    cases.push_back({{"layer",layer},{"n",n},{"types",{int(role[0].type),int(role[1].type),int(role[2].type)}},{"checks",checks}});
    std::cout<<"PASS layer="<<layer<<" n="<<n<<std::endl;
   }
  }
  Json report{{"format","m31-q2-native-slot-component-v1"},{"pass",true},{"slot_stride",stride},{"stride_lcm",multiple},{"maximum_role_span",maximum_span},{"selected_experts",selected},{"physical_permutation",permutation},{"cases",cases},{"total_f32_values",total},{"scope","Stock GGML CUDA full FFN compact vs padded/remapped weights; gate/up/down/final F32 exact, device graph with host readback only for output proof; synthetic bounded inputs and actual Q2 weights. Not full-model quality or speed."}};
  std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';ck(bool(out),"report write");
  ggml_backend_free(backend);backend=nullptr;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}
}
