// Large real-weight numerical proof before any pair-kernel performance claim.
#include "cpu_batched_dot.hpp"
#include "m31_geometry.hpp"
#include "ggml-cpu.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace strata::m31;
void ensure(bool ok,const char * text){if(!ok)throw std::runtime_error(text);}
int main(int argc,char ** argv){try{
 ensure(argc==2,"usage: port7_cpu_pair_check MODEL");ggml_cpu_init();if(!cpu_pair_detail::available())return 77;
 strata::GgufModel model(strata::gguf_split_paths(argv[1]));const auto g=read_geometry(model);
 std::array<uint64_t,2> pair_count{},full_rows{},block_count{};uint64_t input_cases=0;
 for(int layer:{3,10,52,58,59})for(const char * role:{"gate","up","down"}){
  const auto name=layer_name(layer,std::string("ffn_")+role+"_exps.weight");size_t shard=0;const auto * info=model.find(name,&shard);ensure(info,"weight");const auto type=ggml_type(info->type);
  if(type!=GGML_TYPE_IQ3_XXS&&type!=GGML_TYPE_IQ4_XS)continue;const int slot=type==GGML_TYPE_IQ3_XXS?0:1;
  const int n=int(info->shape[0]),rows=int(info->shape[1]),experts=int(info->shape[2]);const auto * trait=ggml_get_type_traits_cpu(type);const auto quant=ggml_get_type_traits_cpu(trait->vec_dot_type)->from_float;
  const size_t row_bytes=ggml_row_size(type,n),expert_bytes=row_bytes*size_t(rows),qbytes=ggml_row_size(trait->vec_dot_type,n);const size_t weight_block=ggml_type_size(type),qblock=ggml_type_size(trait->vec_dot_type);
  const auto * weights=model.shard(shard).tensor_data(*info);std::vector<float> x(n),y(n);std::vector<uint8_t> qx(qbytes),qy(qbytes);
  for(int pattern=0;pattern<8;++pattern){const float amplitudes[]={0,1,1e-10f,1e10f,7,127,.125f,3.5f};for(int i=0;i<n;++i){x[size_t(i)]=amplitudes[pattern]*(pattern==4?(i%2?1.f:-1.f):std::sin(float(i)*(.013f+.001f*pattern)));y[size_t(i)]=amplitudes[pattern]*(pattern==5?(i%3?-.75f:1.f):std::cos(float(i)*(.019f+.002f*pattern)+.123f));}quant(x.data(),qx.data(),n);quant(y.data(),qy.data(),n);++input_cases;
   for(int expert=0;expert<experts;++expert)for(int sample=0;sample<16;++sample){const int row=(sample*377+expert*13)%rows;const auto * w=weights+size_t(expert)*expert_bytes+size_t(row)*row_bytes;
    auto verify=[&](int columns,const void * source,const void * a,const void * b,bool full){float expected[2],actual[2];trait->vec_dot(columns,&expected[0],0,source,0,a,0,1);trait->vec_dot(columns,&expected[1],0,source,0,b,0,1);ensure(cpu_pair_detail::dot_pair(type,columns,source,a,b,actual[0],actual[1]),"pair dispatch unavailable");if(std::memcmp(expected,actual,sizeof expected)){std::fprintf(stderr,"pair mismatch layer=%d role=%s type=%d expert=%d row=%d pattern=%d n=%d exp=%.9g/%.9g got=%.9g/%.9g\n",layer,role,int(type),expert,row,pattern,columns,expected[0],expected[1],actual[0],actual[1]);throw std::runtime_error("pair arithmetic differs bitwise");}pair_count[slot]+=2;if(full)++full_rows[slot];else ++block_count[slot];};
    verify(n,w,qx.data(),qy.data(),true);const int nb=n/256;for(int block:{0,nb/3,2*nb/3,nb-1})verify(256,w+size_t(block)*weight_block,qx.data()+size_t(block)*qblock,qy.data()+size_t(block)*qblock,false);
   }
  }
 }
 std::printf("{\"pass\":true,\"IQ3_XXS\":{\"dot_results\":%llu,\"full_row_pairs\":%llu,\"single_block_pairs\":%llu},\"IQ4_XS\":{\"dot_results\":%llu,\"full_row_pairs\":%llu,\"single_block_pairs\":%llu},\"input_cases\":%llu,\"rhs_patterns\":8,\"weight_experts\":128,\"rows_per_expert\":16,\"source\":\"unchanged GGUF layers3/10/52/58/59\",\"stock_reference\":\"CPU vec_dot nrc1 independently for both RHS\",\"max_abs\":0}\n",(unsigned long long)pair_count[0],(unsigned long long)full_rows[0],(unsigned long long)block_count[0],(unsigned long long)pair_count[1],(unsigned long long)full_rows[1],(unsigned long long)block_count[1],(unsigned long long)input_cases);
 }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
