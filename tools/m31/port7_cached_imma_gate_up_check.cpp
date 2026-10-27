// Isolated proposal A/B: existing CUDA countM dot API, real hot290 held assignments.
#include "gpu_cached_exact.hpp"
#include "cpu_fused_decode.hpp"
#include "m31_geometry.hpp"
#include "json.hpp"
#include <cuda_runtime.h>
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
using namespace strata::m31;
using Json=nlohmann::json;
static void ck(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void cuda_ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
struct Device {uint8_t*p=nullptr;explicit Device(size_t n){cuda_ck(cudaMalloc(reinterpret_cast<void**>(&p),n));}~Device(){cudaFree(p);}};
static void same(const std::vector<float>&a,const std::vector<float>&b,const std::string&name){ck(a.size()==b.size(),"grouped term geometry");if(std::memcmp(a.data(),b.data(),a.size()*4)){size_t i=0;while(i<a.size()&&!std::memcmp(&a[i],&b[i],4))++i;std::fprintf(stderr,"%s term%zu ref=%a got=%a\n",name.c_str(),i,a.at(i),b.at(i));throw std::runtime_error("grouped weighted bit mismatch");}}
static Json timing(const std::vector<double>&wall,const GpuCachedExact::Stats&a,const GpuCachedExact::Stats&b){auto sorted=wall;std::sort(sorted.begin(),sorted.end());const double n=double(wall.size());double sum=0;for(auto x:wall)sum+=x;return Json{{"jobs",wall.size()},{"wall_median_us",sorted[sorted.size()/2]},{"wall_mean_us",sum/n},{"worker_mean_us",(b.worker_seconds-a.worker_seconds)*1e6/n},{"gate_up_mean_us",(b.gate_up_seconds-a.gate_up_seconds)*1e6/n},{"oai_mean_us",(b.oai_seconds-a.oai_seconds)*1e6/n},{"down_mean_us",(b.down_seconds-a.down_seconds)*1e6/n},{"joins_per_job",double(b.joins-a.joins)/n},{"multiply_calls_per_job",double(b.multiply_calls-a.multiply_calls)/n},{"grouped_jobs",b.grouped_jobs-a.grouped_jobs},{"grouped_assignments",b.grouped_assignments-a.grouped_assignments},{"grouped_unique_experts",b.grouped_unique_experts-a.grouped_unique_experts},{"device_bytes",b.device_bytes},{"pinned_bytes",b.pinned_bytes},{"iq3_imma_calls",b.iq3_imma_calls-a.iq3_imma_calls},{"joined_gu_jobs",b.joined_gu_jobs-a.joined_gu_jobs},{"joined_gu_assignments",b.joined_gu_assignments-a.joined_gu_assignments},{"joined_gu_groups",b.joined_gu_groups-a.joined_gu_groups}};}
int main(int argc,char**argv){ggml_backend_t cpu=nullptr;ggml_threadpool_t pool=nullptr;ggml_context*ctx=nullptr;try{
 ck(argc==6,"usage: grouped_check MODEL REAL_L3_X FIXTURE.json REPORT.json REPEATS");ck(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","aictl required");const int repeats=std::stoi(argv[5]);ck(repeats>=4&&repeats<=100,"repeats range");Json fixture;std::ifstream jf(argv[3]);jf>>fixture;ck(fixture.at("format")=="m31-port7-cached-grouped-actual-fixture-v1","fixture format");
 cuda_ck(cudaSetDevice(0));cpu=ggml_backend_cpu_init();auto params=ggml_threadpool_params_default(8);pool=ggml_threadpool_new(&params);ggml_backend_cpu_set_threadpool(cpu,pool);ggml_backend_cpu_set_n_threads(cpu,8);ctx=ggml_init({1024*1024,nullptr,true});
 strata::GgufModel model(strata::gguf_split_paths(argv[1]));const auto g=read_geometry(model);ck(g.embd==6144&&g.expert_ff==3072&&g.used==4,"real geometry");std::vector<float>real(6144*128);std::ifstream f(argv[2],std::ios::binary);f.read(reinterpret_cast<char*>(real.data()),std::streamsize(real.size()*4));ck(bool(f),"real activation fixture");
 constexpr size_t replicas=8;size_t pointer_epoch=0,peak_pools_bytes=0;cudaDeviceProp prop{};cuda_ck(cudaGetDeviceProperties(&prop,0));ck(prop.l2CacheSize>0,"device L2 size unavailable");
 Json results=Json::array();size_t values=0,cases=0;const bool candidate_gu=std::getenv("M31_COMPONENT_GU")&&std::string(std::getenv("M31_COMPONENT_GU"))=="1";
 {

 for(int group_mode:{0,1}){setenv("M31_GPU_CACHED_IQ3_IMMA","0",1);setenv("M31_GPU_CACHED_GROUPED",group_mode?"1":"0",1);setenv("M31_GPU_CACHED_GATE_UP","0",1);GpuCachedExact baseline(int(g.embd),int(g.expert_ff),4,8);setenv("M31_GPU_CACHED_IQ3_IMMA","1",1);setenv("M31_GPU_CACHED_GATE_UP",candidate_gu?"1":"0",1);GpuCachedExact grouped(int(g.embd),int(g.expert_ff),4,8);
 for(int layer:{3,10,52})for(size_t up_gap:{size_t(0),size_t(128)}){
  std::set<int32_t>resident;for(const auto&c:fixture.at("cases"))if(c.at("layer").get<int>()==layer)for(const auto&hit:c.at("hits"))resident.insert(hit.at("expert").get<int32_t>());
  std::map<int32_t,int32_t>slot;for(auto id:resident)slot[id]=int32_t(slot.size());std::array<ggml_tensor*,3>wt{};std::array<const uint8_t*,3>source{};std::array<std::vector<float>,3>all_scales;std::array<size_t,3>raw{};std::array<int,3>types{};const std::array<const char*,3>roles{"gate","up","down"};
  for(size_t r=0;r<3;++r){const auto name=layer_name(layer,std::string("ffn_")+roles[r]+"_exps.weight");size_t sh=0;const auto*t=model.find(name,&sh);ck(t,"missing original weights");types[r]=int(t->type);source[r]=model.shard(sh).tensor_data(*t);const int columns=r==2?int(g.expert_ff):int(g.embd),rows=r==2?int(g.embd):int(g.expert_ff);raw[r]=ggml_row_size(ggml_type(types[r]),columns)*size_t(rows);wt[r]=ggml_new_tensor_3d(ctx,ggml_type(types[r]),columns,rows,g.experts);wt[r]->data=const_cast<uint8_t*>(source[r]);all_scales[r].assign(size_t(g.experts),1.f);const auto sn=name.substr(0,name.size()-6)+"scale";if(const auto*s=model.find(sn,&sh))std::memcpy(all_scales[r].data(),model.shard(sh).tensor_data(*s),size_t(g.experts)*4);}
  const auto buft=ggml_backend_cuda_buffer_type(0);const size_t alignment=ggml_backend_buft_get_alignment(buft);const size_t stride=25909632;std::array<size_t,3>offsets{};size_t span=0;
  for(size_t r=0;r<3;++r){const int columns=r==2?int(g.expert_ff):int(g.embd),rows=r==2?int(g.embd):int(g.expert_ff);auto*t=ggml_new_tensor_2d(ctx,ggml_type(types[r]),columns,rows);const size_t allocated=ggml_backend_buft_get_alloc_size(buft,t);offsets[r]=(span+alignment-1)/alignment*alignment;span=offsets[r]+(allocated+alignment-1)/alignment*alignment;}
  offsets[1]+=up_gap;offsets[2]+=up_gap;span+=up_gap;ck(span<=stride&&offsets[1]-offsets[0]==raw[0]+up_gap&&types[0]==types[1],"canonical contiguous GU layout differs");
  auto device=std::make_unique<Device>(replicas*slot.size()*stride);std::vector<uint8_t>packed_weights(replicas*slot.size()*stride);std::array<const void*,3>pools{};
  for(size_t copy=0;copy<replicas;++copy)for(const auto&kv:slot)for(size_t r=0;r<3;++r)std::memcpy(packed_weights.data()+(copy*slot.size()+size_t(kv.second))*stride+offsets[r],source[r]+size_t(kv.first)*raw[r],raw[r]);
  cuda_ck(cudaMemcpy(device->p,packed_weights.data(),packed_weights.size(),cudaMemcpyHostToDevice));for(size_t r=0;r<3;++r)pools[r]=device->p+offsets[r];
  peak_pools_bytes=std::max(peak_pools_bytes,replicas*slot.size()*stride);
  CpuFusedDecode reference(cpu,pool,8);
  for(const auto&c:fixture.at("cases")){if(c.at("layer").get<int>()!=layer)continue;const int n=c.at("n");if(up_gap&&n!=1)continue;const std::string name=c.at("name");const auto ids=c.at("full_ids").get<std::vector<int32_t>>();const auto full_weights=c.at("full_weights").get<std::vector<float>>();const int first=c.at("first");std::vector<float>x(size_t(n)*size_t(g.embd));for(int token=0;token<n;++token)std::copy_n(real.data()+size_t((first+token*17+layer)%128)*size_t(g.embd),size_t(g.embd),x.data()+size_t(token)*size_t(g.embd));
   std::vector<GpuCachedExact::Assignment>assignments;std::vector<float>weights;std::array<std::vector<float>,3>scales;std::set<int32_t>groups;
   for(const auto&hit:c.at("hits")){const int token=hit.at("token"),rank=hit.at("rank");const int32_t expert=hit.at("expert");ck(ids[size_t(token)*4+size_t(rank)]==expert,"fixture route identity");assignments.push_back({token,rank,slot.at(expert)});weights.push_back(full_weights[size_t(token)*4+size_t(rank)]);for(size_t r=0;r<3;++r)scales[r].push_back(all_scales[r][size_t(expert)]);groups.insert(expert);}
   ck(assignments.size()==c.at("H").get<size_t>()&&groups.size()==c.at("G").get<size_t>(),"fixture H/G");std::vector<float>full_terms(size_t(n)*4*size_t(g.embd));
   for(int token=0;token<n;++token){std::vector<float>xx(x.begin()+size_t(token)*size_t(g.embd),x.begin()+size_t(token+1)*size_t(g.embd)),ww(full_weights.begin()+token*4,full_weights.begin()+(token+1)*4),unused;std::vector<int32_t>ii(ids.begin()+token*4,ids.begin()+(token+1)*4);reference.forward(wt[0],wt[1],wt[2],all_scales[0].data(),all_scales[1].data(),all_scales[2].data(),xx,ii,ww,unused);const auto&terms=reference.contributions();std::copy(terms.begin(),terms.end(),full_terms.begin()+size_t(token)*4*size_t(g.embd));}
   std::vector<float>expected;for(const auto&a:assignments){const auto*start=full_terms.data()+(size_t(a.token)*4+size_t(a.rank))*size_t(g.embd);expected.insert(expected.end(),start,start+g.embd);}
   const size_t working_set=groups.size()*(raw[0]+raw[1]+raw[2]);ck((replicas-2)*working_set>size_t(prop.l2CacheSize),"rotating reuse distance does not exceed actual L2");
   auto run=[&](GpuCachedExact&helper){const size_t ticket=pointer_epoch++;const size_t copy=(ticket+ticket/replicas)%replicas;std::array<const void*,3>physical{};for(size_t role=0;role<3;++role)physical[role]=static_cast<const uint8_t*>(pools[role])+copy*slot.size()*stride;helper.begin_batched(types,physical,stride,x,n,assignments,weights,scales);std::vector<float>out;helper.finish(out);return out;};
   same(expected,run(baseline),name+"-baseline");same(expected,run(grouped),name+"-grouped");values+=expected.size();++cases;
   for(int warm=0;warm<4;++warm){run(baseline);run(grouped);}const auto a=baseline.stats(),b=grouped.stats();std::vector<double>ta,tb;
   auto timed=[&](GpuCachedExact&helper,std::vector<double>&times){const auto started=std::chrono::steady_clock::now();run(helper);times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count());};
   for(int repeat=0;repeat<repeats;++repeat){if(repeat%2){timed(grouped,tb);timed(baseline,ta);}else{timed(baseline,ta);timed(grouped,tb);}}
   const auto aa=baseline.stats(),bb=grouped.stats();const auto old=timing(ta,a,aa),candidate=timing(tb,b,bb);ck(bb.grouped_jobs-b.grouped_jobs==uint64_t(group_mode&&assignments.size()>groups.size()?repeats:0),"grouped path coverage differs");ck(bb.joined_gu_jobs-b.joined_gu_jobs==uint64_t(!up_gap&&!group_mode&&candidate_gu?repeats:0),"joined GU eligibility/fallback coverage differs");ck(bb.iq3_imma_calls>b.iq3_imma_calls||group_mode,"IMMA path not covered");ck(aa.joined_gu_jobs==0,"baseline unexpectedly joined GU");
   results.push_back(Json{{"name",name},{"group_mode",group_mode},{"up_gap_bytes",up_gap},{"offsets",offsets},{"GU_raw_bytes",raw[0]},{"GU_eligible",up_gap==0},{"layer",layer},{"n",n},{"hits",assignments.size()},{"unique_experts",groups.size()},{"types",types},{"weight_working_set_bytes",working_set},{"min_reuse_distance_bytes",(replicas-2)*working_set},{"l2_bytes",prop.l2CacheSize},{"replicas",replicas},{"requested_layer_pool_bytes",replicas*slot.size()*stride},{"baseline",old},{"combined",candidate},{"wall_median_speedup",old.at("wall_median_us").get<double>()/candidate.at("wall_median_us").get<double>()},{"worker_speedup",old.at("worker_mean_us").get<double>()/candidate.at("worker_mean_us").get<double>()}});
  }
 }
 }
 }
 Json report{{"pass",true},{"candidate_GU",candidate_gu},{"physical_replicas",replicas},{"actual_l2_bytes",prop.l2CacheSize},{"peak_requested_pool_bytes",peak_pools_bytes},{"cache_policy","8 physical replicas with rotated starting copy every8 jobs for A/B address balance; minimum6 fresh working sets exceed actual L2; capacity thrash, no direct L2 hit/miss measurement"},{"cases",cases},{"weighted_f32_values",values},{"repeats",repeats},{"results",results},{"scope","Isolated adjacent gate+up A/B on canonical slot layout, group0/1 and padding-gap fallback; real held hot290 assignments and unchanged quants; raw CPU bitproof and capacity-thrash helper timing, not Engine/TG acceptance"}};std::ofstream out(argv[4]);out<<report.dump(2)<<'\n';ck(bool(out),"report write");ggml_free(ctx);ctx=nullptr;ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);pool=nullptr;ggml_backend_free(cpu);cpu=nullptr;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';if(ctx)ggml_free(ctx);if(cpu)ggml_backend_cpu_set_threadpool(cpu,nullptr);if(pool)ggml_threadpool_free(pool);if(cpu)ggml_backend_free(cpu);return 1;}}
