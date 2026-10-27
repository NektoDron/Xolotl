// Independent graph-only coalesce proposal. No Engine/common CMake integration.
#include "m31_dspark_coalesced_rows.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
using namespace strata::m31;
static void need(bool ok,const std::string &why){if(!ok)throw std::runtime_error(why);}
static void same(const std::vector<float>&a,const std::vector<float>&b){need(a.size()==b.size(),"raw dimensions");if(std::memcmp(a.data(),b.data(),a.size()*4)){size_t i=0;while(i<a.size()&&!std::memcmp(&a[i],&b[i],4))++i;std::fprintf(stderr,"coalesce mismatch at%zu old=%a new=%a\n",i,a.at(i),b.at(i));throw std::runtime_error("coalesced raw bit mismatch");}}
static float raw_value(size_t i){const std::array<uint32_t,12>patterns{0,0x80000000u,1,0x80000001u,0x3f800000u,0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7fa12345u,0x00800000u,0x7f7fffffu};const auto bits=patterns[i%patterns.size()];float f;std::memcpy(&f,&bits,4);return f;}
struct Graph {
 ggml_backend_t backend;ggml_context*c=nullptr;ggml_cgraph*graph=nullptr;ggml_gallocr_t allocator=nullptr;
 ggml_tensor*input=nullptr,*weight=nullptr,*root=nullptr,*output=nullptr;
 int concat_nodes=0,cont_nodes=0;
 Graph(ggml_backend_t b,int64_t d,bool root3d,bool arithmetic,bool nested,int first,int count,bool coalesce):backend(b){
  c=ggml_init({2*1024*1024,nullptr,true});need(c,"metadata context");graph=ggml_new_graph(c);
  input=root3d?ggml_new_tensor_3d(c,GGML_TYPE_F32,d,1,8):ggml_new_tensor_2d(c,GGML_TYPE_F32,d,8);ggml_set_input(input);
  if(arithmetic){weight=ggml_new_tensor_1d(c,GGML_TYPE_F32,d);ggml_set_input(weight);root=ggml_mul(c,ggml_rms_norm(c,input,1e-5f),weight);}else root=ggml_cont(c,input);
  auto*parent=root;if(nested)parent=root3d?ggml_reshape_2d(c,root,d,8):ggml_reshape_3d(c,root,d,1,8);
  std::vector<ggml_tensor*>rows;for(int row=0;row<8;++row){auto*t=parent->ne[1]==1?ggml_view_3d(c,parent,d,1,1,parent->nb[1],parent->nb[2],size_t(row)*size_t(d)*4):ggml_view_2d(c,parent,d,1,parent->nb[1],size_t(row)*size_t(d)*4);need(!t->data&&!root->data,"predicate must run before tensor allocation");rows.push_back(t);}
  if(coalesce){output=dspark_try_coalesce_rows(c,rows,first,count);need(output,"eligible rows were rejected");need(output->op==GGML_OP_VIEW&&output->src[0]==root&&output->view_src==root&&output->view_offs==size_t(first)*size_t(d)*4,"source producer edge/layout lost");}
  else for(int i=0;i<count;++i){auto*row=ggml_cont(c,rows[size_t(first+i)]);output=output?ggml_concat(c,output,row,1):row;}
  need(output->type==GGML_TYPE_F32&&output->ne[0]==d&&output->ne[1]==count&&output->ne[2]==1&&output->ne[3]==1&&output->nb[1]==size_t(d)*4&&ggml_is_contiguous(output),"joined shape differs");
  ggml_set_output(output);ggml_build_forward_expand(graph,output);int rms=0,mul=0;const auto device=ggml_backend_get_device(backend);
  for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*t=ggml_graph_node(graph,i);need(ggml_backend_dev_supports_op(device,t),"backend does not support node");concat_nodes+=t->op==GGML_OP_CONCAT;cont_nodes+=t->op==GGML_OP_CONT;rms+=t->op==GGML_OP_RMS_NORM;mul+=t->op==GGML_OP_MUL;}
  if(arithmetic)need(rms==1&&mul==1,"RMS/MUL producer is missing from output graph");
  allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));need(allocator&&ggml_gallocr_alloc_graph(allocator,graph),"graph allocation");if(coalesce)need(output->data==static_cast<uint8_t*>(root->data)+size_t(first)*size_t(d)*4,"coalesced view does not alias root");
 }
 ~Graph(){ggml_backend_synchronize(backend);
#ifdef M31_ENGINE_CUDA
 if(ggml_backend_is_cuda(backend))ggml_backend_cuda_graph_cache_clear(backend);
#endif
 if(allocator)ggml_gallocr_free(allocator);if(c)ggml_free(c);}
 std::vector<float>run(const std::vector<float>&x,const std::vector<float>&w){ggml_backend_tensor_set(input,x.data(),0,x.size()*4);if(weight)ggml_backend_tensor_set(weight,w.data(),0,w.size()*4);need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");std::vector<float>out(size_t(ggml_nelements(output)));ggml_backend_tensor_get(output,out.data(),0,out.size()*4);return out;}
};
static size_t negative_predicates(){
 auto*c=ggml_init({512*1024,nullptr,true});need(c,"negative metadata context");size_t checks=0;try{
 const int64_t d=16;const size_t stride=64;auto*owner=ggml_new_tensor_3d(c,GGML_TYPE_F32,d,1,8);std::vector<ggml_tensor*>rows;for(int i=0;i<8;++i)rows.push_back(ggml_view_2d(c,owner,d,1,stride,size_t(i)*stride));
 auto reject=[&](const std::vector<ggml_tensor*>&r,int first,int count,const char*name){const size_t before=ggml_used_mem(c);need(dspark_try_coalesce_rows(c,r,first,count)==nullptr,std::string("unsafe coalesce accepted: ")+name);need(ggml_used_mem(c)==before,std::string("rejected predicate allocated metadata: ")+name);++checks;};
 reject(rows,-1,2,"negative first");reject(rows,0,1,"one row fallback");reject(rows,0,0,"empty span");reject(rows,0,5,"count>4");reject(rows,8,2,"first past end");reject(rows,7,2,"range past end");reject({},0,2,"empty vector");need(dspark_try_coalesce_rows(nullptr,rows,0,2)==nullptr,"null context");++checks;
 reject({rows[0],nullptr},0,2,"null row");reject({rows[0],rows[2]},0,2,"gap");reject({rows[1],rows[0]},0,2,"reverse");reject({rows[0],rows[0]},0,2,"duplicate row");
 auto*other=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,8);auto*otherrow=ggml_view_2d(c,other,d,1,stride,stride);reject({rows[0],otherrow},0,2,"different roots");
 auto*wide=ggml_new_tensor_2d(c,GGML_TYPE_F32,3*d,8);auto*q=ggml_view_2d(c,wide,d,1,wide->nb[1],0);auto*k=ggml_view_2d(c,wide,d,1,wide->nb[1],stride);reject({q,k},0,2,"partial packed QKV");
 auto*h=ggml_new_tensor_2d(c,GGML_TYPE_F16,d,8);reject({ggml_view_2d(c,h,d,1,size_t(d)*2,0),ggml_view_2d(c,h,d,1,size_t(d)*2,size_t(d)*2)},0,2,"non F32");
 auto*pair=ggml_view_2d(c,owner,d,2,stride,0);reject({pair,rows[2]},0,2,"non n1 row");
 for(int bad=0;bad<7;++bad){auto a=*rows[0],b=*rows[1],parent=*owner;
  if(bad==0)a.nb[0]=8;if(bad==1)a.nb[1]=stride+4;
  if(bad==2){parent.ne[1]=8;parent.ne[2]=1;parent.nb[1]=stride+4;parent.nb[2]=(stride+4)*8;parent.nb[3]=parent.nb[2];a.view_src=b.view_src=&parent;a.src[0]=b.src[0]=&parent;}
  if(bad==3){a.view_offs=ggml_nbytes(owner);b.view_offs=a.view_offs+stride;}
  if(bad==4){a.view_offs=4;b.view_offs=stride+4;}
  if(bad==5){a.src[0]=&a;}
  if(bad==6){parent.view_src=other;a.view_src=b.view_src=&parent;a.src[0]=b.src[0]=&parent;}
  reject({&a,&b},0,2,"malformed metadata/stride/bounds/cycle");
 }
 auto*s0=ggml_scale_inplace(c,rows[0],2.f);auto*s1=ggml_scale_inplace(c,rows[1],2.f);reject({s0,s1},0,2,"inplace arithmetic aliases");auto*v0=ggml_view_2d(c,s0,d,1,stride,0);auto*v1=ggml_view_2d(c,s1,d,1,stride,0);reject({v0,v1},0,2,"views over inplace arithmetic");
 auto*foreign=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,1);auto*copy=ggml_cpy(c,foreign,rows[0]);reject({copy,rows[1]},0,2,"inplace CPY writer");
 ggml_free(c);return checks;
 }catch(...){ggml_free(c);throw;}
}
int main(int argc,char**argv){ggml_backend_t backend=nullptr;try{
 need(argc==3,"usage: coalesced_rows_check cpu|cuda REPORT.json");const std::string kind=argv[1];if(kind=="cpu")backend=ggml_backend_cpu_init();
#ifdef M31_ENGINE_CUDA
 else if(kind=="cuda"){need(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","CUDA requires aictl session");backend=ggml_backend_cuda_init(0);}
#endif
 need(backend,"missing backend");const size_t rejected=negative_predicates();size_t cases=0,values=0,removed_concat=0,removed_cont=0;
 for(int64_t d:{1,31,256,3072,6144})for(bool root3d:{false,true})for(bool arithmetic:{false,true})for(bool nested:{false,true})for(int count:{2,3,4})for(int first:{0,1,3}){
  Graph old(backend,d,root3d,arithmetic,nested,first,count,false),coalesced(backend,d,root3d,arithmetic,nested,first,count,true);std::vector<float>x(size_t(d)*8),w(size_t(d),1.f);for(size_t i=0;i<x.size();++i)x[i]=arithmetic?.13f+.5f*std::sin(float(i%997)*.017f):raw_value(i);for(size_t i=0;i<w.size();++i)w[i]=1.f+.03125f*float(i%7);
  const auto reference=old.run(x,w),actual=coalesced.run(x,w);same(reference,actual);if(!arithmetic){const std::vector<float>expected(x.begin()+size_t(first)*size_t(d),x.begin()+size_t(first+count)*size_t(d));same(expected,actual);}
  need(old.concat_nodes==count-1&&coalesced.concat_nodes==0&&old.cont_nodes-coalesced.cont_nodes==count,"copy/concat nodes were not removed");removed_concat+=size_t(count-1);removed_cont+=size_t(count);values+=actual.size();++cases;
 }
 std::ofstream out(argv[2]);out<<"{\"pass\":true,\"backend\":\""<<kind<<"\",\"cases\":"<<cases<<",\"f32_values\":"<<values<<",\"metadata_reject_checks\":"<<rejected<<",\"removed_concat_nodes\":"<<removed_concat<<",\"removed_cont_nodes\":"<<removed_cont<<",\"checks_before_allocation\":true,\"RMS_MUL_root_producer_preserved\":true,\"root_2D_and_D1N\":true,\"nested_metadata_views\":true,\"scope\":\"Independent full-width adjacent view proposal, raw bit parity incl signedzero/NaN/subnormal and actual RMS/MUL roots; no Engine/model/performance acceptance\"}\n";need(bool(out),"report write");ggml_backend_free(backend);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}}
