// Isolated proposal; no live Engine, no common CMake changes.
#include "m31_dspark_packed_routes.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace strata::m31;
static void need(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class T>static void same(const std::vector<T>&a,const std::vector<T>&b){need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)),"packed metadata bit mismatch");}
struct Graph {
 ggml_backend_t backend;ggml_context *c=nullptr;ggml_cgraph *graph=nullptr;ggml_gallocr_t allocator=nullptr;
 explicit Graph(ggml_backend_t b):backend(b){c=ggml_init({2*1024*1024,nullptr,true});need(c,"metadata context");graph=ggml_new_graph(c);}
 ~Graph(){ggml_backend_synchronize(backend);
#ifdef M31_ENGINE_CUDA
 if(ggml_backend_is_cuda(backend))ggml_backend_cuda_graph_cache_clear(backend);
#endif
 if(allocator)ggml_gallocr_free(allocator);if(c)ggml_free(c);}
 void allocate(const std::vector<ggml_tensor*>&out){for(auto*t:out){ggml_set_output(t);ggml_build_forward_expand(graph,t);}auto dev=ggml_backend_get_device(backend);for(int i=0;i<ggml_graph_n_nodes(graph);++i)need(ggml_backend_dev_supports_op(dev,ggml_graph_node(graph,i)),"backend does not support packing node");allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));need(allocator&&ggml_gallocr_alloc_graph(allocator,graph),"packing graph allocation");}
 void compute(){need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"packing graph compute");}
};
static float sample(size_t i){const std::array<uint32_t,12>bits{0,0x80000000u,0x3f800000u,0xbf800000u,1,0x80000001u,0x7f7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0x7fa12345u,0x00800000u};float f;const uint32_t v=bits[i%bits.size()];std::memcpy(&f,&v,4);return f;}
int main(int argc,char**argv){ggml_backend_t backend=nullptr;try{
 need(argc==3,"usage: packed_routes_check cpu|cuda REPORT.json");const std::string kind=argv[1];
 if(kind=="cpu")backend=ggml_backend_cpu_init();
#ifdef M31_ENGINE_CUDA
 else if(kind=="cuda"){need(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","CUDA requires authorized aictl session");backend=ggml_backend_cuda_init(0);}
#endif
 need(backend,"unsupported/missing backend");size_t cases=0,float_values=0,id_values=0,invalid_checks=0;std::array<bool,128>seen{};
 for(int64_t d:{1,31,256,257,6144})for(int n:{1,2,3,4,5,6,7,8})for(int used:{1,2,4})for(bool strided:{false,true}){
  Graph b(backend);std::array<ggml_tensor*,4>base{},input{};std::array<std::vector<float>,3>host,expected;std::vector<int32_t>host_ids,expected_ids;
  for(int role=0;role<4;++role){const int64_t width=role<2?d:used;const int64_t physical=width+(strided?3:0);base[size_t(role)]=ggml_new_tensor_2d(b.c,role==3?GGML_TYPE_I32:GGML_TYPE_F32,physical,n);ggml_set_input(base[size_t(role)]);input[size_t(role)]=strided?ggml_view_2d(b.c,base[size_t(role)],width,n,size_t(physical)*4,4):base[size_t(role)];
   if(role==3){host_ids.assign(size_t(physical)*size_t(n),-997);for(int token=0;token<n;++token)for(int rank=0;rank<used;++rank){const int32_t id=int32_t((cases*37+size_t(token)*17+size_t(rank)*31)%128);host_ids[size_t(token)*size_t(physical)+size_t(rank)+(strided?1:0)]=id;expected_ids.push_back(id);seen[size_t(id)]=true;}}
   else{host[size_t(role)].resize(size_t(physical)*size_t(n));for(size_t i=0;i<host[size_t(role)].size();++i)host[size_t(role)][i]=sample(i+size_t(role)*3+cases);for(int token=0;token<n;++token)expected[size_t(role)].insert(expected[size_t(role)].end(),host[size_t(role)].begin()+size_t(token)*size_t(physical)+(strided?1:0),host[size_t(role)].begin()+size_t(token)*size_t(physical)+(strided?1:0)+size_t(width));}
  }
  // Mimic real main graph: norm is produced and retained for the separate shared graph.
  auto norm=ggml_cont(b.c,input[0]);auto packed=dspark_pack_routes(b.c,norm,input[1],input[2],input[3]);b.allocate({packed.tensor,norm});
  for(int r=0;r<3;++r)ggml_backend_tensor_set(base[size_t(r)],host[size_t(r)].data(),0,host[size_t(r)].size()*4);ggml_backend_tensor_set(base[3],host_ids.data(),0,host_ids.size()*4);b.compute();
  std::vector<float>scratch;DSparkPackedRoutesHost out;dspark_read_packed_routes(packed,scratch,out);same(expected[0],out.norm);same(expected[1],out.residual);same(expected[2],out.weights);same(expected_ids,out.ids);
  std::vector<float>retained(size_t(d)*size_t(n));ggml_backend_tensor_get(norm,retained.data(),0,retained.size()*4);same(expected[0],retained);
  float_values+=out.norm.size()+out.residual.size()+out.weights.size();id_values+=out.ids.size();++cases;
  if(d==6144&&n==8&&used==4&&!strided){const auto old=out;for(int bad=0;bad<5;++bad){auto malformed=scratch;if(bad<4)malformed[size_t(2*d+used)]=bad==0?-1.f:bad==1?128.f:bad==2?1.5f:sample(7);else malformed.pop_back();bool rejected=false;try{dspark_unpack_routes(packed.layout,malformed,out);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"malformed packed ids accepted");same(old.norm,out.norm);same(old.residual,out.residual);same(old.weights,out.weights);same(old.ids,out.ids);++invalid_checks;}}
 }
 need(std::all_of(seen.begin(),seen.end(),[](bool value){return value;}),"did not cover every expert id0..127");
 std::ofstream report(argv[2]);report<<"{\"pass\":true,\"backend\":\""<<kind<<"\",\"cases\":"<<cases<<",\"f32_values\":"<<float_values<<",\"id_values\":"<<id_values<<",\"all_ids0to127\":true,\"invalid_unchanged_checks\":"<<invalid_checks<<",\"norm_retained_output\":true,\"metadata_gets_per_case\":1,\"reference_metadata_gets\":4,\"scope\":\"Isolated stock concat+cast proposal: exact bit unpack, contiguous/strided, all N1..8, no Engine or performance acceptance\"}\n";need(bool(report),"result write");ggml_backend_free(backend);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}}
