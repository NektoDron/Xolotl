#include "port7_iq3_lane_probe.hpp"
#include "port7_iq3_lane_reference.hpp"
#include "gpu_resident_dot.hpp"
#include "m31_geometry.hpp"
#include "ggml-cpu.h"
#include "json.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <vector>

using namespace strata;
using namespace strata::m31;
namespace probe=iq3_lane_probe;
namespace ref=iq3_lane_reference;
using Json=nlohmann::json;
static_assert(sizeof(block_iq3_xxs)==98&&sizeof(block_q8_K)==292,"original host ggml block layouts");
static void need(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
struct Device {void * p=nullptr;explicit Device(size_t bytes){ck(cudaMalloc(&p,bytes));}~Device(){cudaFree(p);}};
struct Stream {cudaStream_t p=nullptr;Stream(){ck(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking));}~Stream(){cudaStreamSynchronize(p);cudaStreamDestroy(p);}};
static uint64_t compared_i32=0,compared_f32=0;
static void equal_float(const std::vector<float> & a,const std::vector<float> & b,const char * label){
    need(a.size()==b.size(),"float comparison shape");compared_f32+=a.size();
    if(std::memcmp(a.data(),b.data(),a.size()*4)){size_t i=0;while(i<a.size()&&!std::memcmp(&a[i],&b[i],4))++i;
        std::fprintf(stderr,"%s mismatch index=%zu ref=%a actual=%a\n",label,i,a.at(i),b.at(i));throw std::runtime_error(label);}
}
static std::vector<float> cpu_dots(const block_iq3_xxs * w,const block_q8_K * q,int columns,int rows){
    const auto * traits=ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS);std::vector<float> out(size_t(rows),0);
    for(int row=0;row<rows;++row)traits->vec_dot(columns,&out[size_t(row)],0,w+size_t(row)*(columns/256),0,q,0,1);
    return out;
}
static void check_lanes(const block_iq3_xxs * w,const block_q8_K * q,int columns,int rows,
        const std::vector<int32_t> & actual,bool all_rows){
    const int blocks=columns/256;std::set<int> chosen;
    if(all_rows)for(int row=0;row<rows;++row)chosen.insert(row);
    else for(int row:{0,1,7,8,15,16,rows-1})if(row>=0&&row<rows)chosen.insert(row);
    for(int row:chosen){std::vector<int32_t> integers(size_t(blocks)*8);std::vector<float> d(size_t(blocks),0);
        for(int b=0;b<blocks;++b){const auto * x=w+size_t(row)*blocks+b;const auto lanes=ref::block(*x,q[b]);d[size_t(b)]=ggml_fp16_to_fp32(x->d)*q[b].d;
            for(int lane=0;lane<8;++lane){const int32_t got=actual[(size_t(row)*blocks+b)*8+lane];++compared_i32;
                if(got!=lanes.sum[size_t(lane)]){std::fprintf(stderr,"IQ3 partial mismatch row=%d block=%d lane=%d ref=%d got=%d\n",row,b,lane,lanes.sum[size_t(lane)],got);throw std::runtime_error("IMMA integer lane mismatch");}
                integers[size_t(b)*8+lane]=lanes.sum[size_t(lane)];}
        }
        float stock=0;ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot(columns,&stock,0,w+size_t(row)*blocks,0,q,0,1);
        const float reference=ref::finish(integers.data(),d.data(),blocks);++compared_f32;
        need(std::memcmp(&stock,&reference,4)==0,"host lane/FMA reference differs from stock CPU nrc1");
    }
}
static void fill_bsums(block_q8_K & q){
    for(int group=0;group<16;++group){int sum=0;for(int j=0;j<16;++j)sum+=q.qs[group*16+j];q.bsums[group]=int16_t(sum);}
}
static void raw_case(cudaStream_t stream,const std::vector<block_iq3_xxs> & w,const std::vector<block_q8_K> & q,int rows){
    const int columns=int(q.size())*256;need(w.size()==size_t(rows)*q.size(),"raw matrix shape");
    Device dw(w.size()*sizeof(w[0])),dq(q.size()*sizeof(q[0])),scratch(probe::scratch_bytes(columns,rows)),out(size_t(rows)*4);
    ck(cudaMemcpyAsync(dw.p,w.data(),w.size()*sizeof(w[0]),cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(dq.p,q.data(),q.size()*sizeof(q[0]),cudaMemcpyHostToDevice,stream));
    probe::multiply(dw.p,dq.p,columns,rows,scratch.p,static_cast<float *>(out.p),stream);
    std::vector<int32_t> integers(probe::scratch_bytes(columns,rows)/4);std::vector<float> actual(size_t(rows),0);
    ck(cudaMemcpyAsync(integers.data(),scratch.p,integers.size()*4,cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(actual.data(),out.p,actual.size()*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    check_lanes(w.data(),q.data(),columns,rows,integers,true);equal_float(cpu_dots(w.data(),q.data(),columns,rows),actual,"raw CPU/IMMA FMA mismatch");
}
static double event_us(cudaStream_t stream,const std::function<void(int)> & call,int repeats){
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));ck(cudaEventRecord(begin,stream));
    for(int i=0;i<repeats;++i)call(i);
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,begin,end));ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return ms*1000/repeats;
}
static double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}

int main(int argc,char ** argv){ggml_backend_t cpu=nullptr;ggml_context * metadata=nullptr;bool write_allowed=false;Json report={{"pass",false},{"cases",Json::array()},{"timings",Json::array()}};try{
    need(argc==4,"usage: authorized GPU0 IQ3 lane probe MODEL REAL_L3_INPUT REPORT.json");
    need(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1"&&std::getenv("CUDA_VISIBLE_DEVICES")&&std::string(std::getenv("CUDA_VISIBLE_DEVICES"))=="0","authorized GPU0 card session required");
    need(!std::filesystem::exists(argv[3]),"output already exists");write_allowed=true;
    cpu=ggml_backend_cpu_init();metadata=ggml_init({1024*1024,nullptr,true});need(cpu&&metadata,"CPU reference initialization");
    need(ggml_cpu_has_avx2()&&ggml_cpu_has_fma(),"stock AVX2/FMA reference required");
    const auto model=GgufModel::open(argv[1]);const auto g=read_geometry(model);need(g.embd==6144&&g.expert_ff==3072&&g.layers==60&&g.dense_lead==3&&g.experts==128,"M3 matrix geometry");
    std::vector<float> real(6144*128);std::ifstream input(argv[2],std::ios::binary);input.read(reinterpret_cast<char *>(real.data()),std::streamsize(real.size()*4));need(bool(input),"short real activation fixture");
    Stream stream;GpuDeviceDots old(stream.p);old.reserve(6144,6144,1);
    const auto * traits=ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS);need(traits->vec_dot_type==GGML_TYPE_Q8_K,"original Q8_K activation contract");const auto * quant=ggml_get_type_traits_cpu(traits->vec_dot_type);
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));report["device"]={{"name",prop.name},{"l2_bytes",prop.l2CacheSize},{"multiprocessors",prop.multiProcessorCount}};
    bool raw_done=false;size_t raw_cases=0,matrix_cases=0;
    for(int layer=3;layer<60;++layer)for(const auto & role:{std::string("gate"),std::string("up")}){
        size_t shard=0;const std::string name=layer_name(layer,"ffn_"+role+"_exps.weight");const auto * info=model.find(name,&shard);need(info&&info->type==GGML_TYPE_IQ3_XXS,"all routed gate/up must retain IQ3");
        const auto * source=reinterpret_cast<const block_iq3_xxs *>(model.shard(shard).tensor_data(*info));
        const int columns=6144,blocks=columns/256,full_rows=3072;const size_t bytes=size_t(full_rows)*blocks*sizeof(block_iq3_xxs);
        const bool full=layer==3||layer==10||layer==52||layer==59;const int rows=full?full_rows:64;
        for(int expert:{0,127}){const auto * weight=source+size_t(expert)*full_rows*blocks;
            Device dw(bytes),dx(size_t(columns)*4),dq(size_t(blocks)*sizeof(block_q8_K)),gpuq(size_t(blocks)*sizeof(block_q8_K)),scratch(probe::scratch_bytes(columns,rows)),candidate(size_t(rows)*4),baseline(size_t(rows)*4);
            ck(cudaMemcpyAsync(dw.p,weight,bytes,cudaMemcpyHostToDevice,stream.p));
            if(!raw_done){
                std::vector<block_iq3_xxs> basis_weights(16);for(int r=0;r<16;++r)basis_weights[size_t(r)]=weight[size_t(r)*blocks+(7*r)%blocks];
                for(int pos=0;pos<256;++pos)for(int value:{-127,-1,1,127}){block_q8_K q{};q.d=1;q.qs[pos]=int8_t(value);fill_bsums(q);raw_case(stream.p,basis_weights,{q},16);++raw_cases;}
                for(int r:{1,7,15,17,31})for(int nb:{1,2,3,4,5,12,24}){
                    std::vector<block_iq3_xxs> weights(size_t(r)*nb);std::vector<block_q8_K> qs(size_t(nb),block_q8_K{});
                    for(int row=0;row<r;++row)for(int b=0;b<nb;++b)weights[size_t(row)*nb+b]=weight[size_t((row*13)%full_rows)*blocks+b];
                    for(int b=0;b<nb;++b){qs[size_t(b)].d=(b&1?-1.f:1.f)*std::ldexp(1.f,(b%3==0?-80:b%3==1?0:50));
                        for(int p=0;p<256;++p)qs[size_t(b)].qs[p]=int8_t((p*73+b*31)%255-127);fill_bsums(qs[size_t(b)]);}
                    raw_case(stream.p,weights,qs,r);++raw_cases;
                }
                raw_done=true;report["raw_cases"]=raw_cases;
            }
            for(int token:{0,63,127}){
                std::vector<float> x(real.begin()+size_t(token)*columns,real.begin()+size_t(token+1)*columns);std::vector<block_q8_K> q(size_t(blocks),block_q8_K{}),actual_quant(q.size());
                quant->from_float(x.data(),q.data(),columns);for(const auto & block:q)for(auto value:block.qs)need(value!=-128,"production quant unexpectedly contains raw-128");
                ck(cudaMemcpyAsync(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream.p));ck(cudaMemcpyAsync(dq.p,q.data(),q.size()*sizeof(q[0]),cudaMemcpyHostToDevice,stream.p));
                gpu_quantize_cpu_compatible(GGML_TYPE_Q8_K,static_cast<float *>(dx.p),gpuq.p,columns,1,stream.p);
                ck(cudaMemcpyAsync(actual_quant.data(),gpuq.p,q.size()*sizeof(q[0]),cudaMemcpyDeviceToHost,stream.p));ck(cudaStreamSynchronize(stream.p));need(std::memcmp(q.data(),actual_quant.data(),q.size()*sizeof(q[0]))==0,"old GPU-compatible quant differs from CPU quant");
                probe::multiply(dw.p,dq.p,columns,rows,scratch.p,static_cast<float *>(candidate.p),stream.p);old.multiply(GGML_TYPE_IQ3_XXS,dw.p,columns,rows,1,static_cast<float *>(dx.p),static_cast<float *>(baseline.p),false);
                std::vector<int32_t> integers(probe::scratch_bytes(columns,rows)/4);std::vector<float> got(size_t(rows),0),previous(size_t(rows),0);
                ck(cudaMemcpyAsync(integers.data(),scratch.p,integers.size()*4,cudaMemcpyDeviceToHost,stream.p));ck(cudaMemcpyAsync(got.data(),candidate.p,got.size()*4,cudaMemcpyDeviceToHost,stream.p));ck(cudaMemcpyAsync(previous.data(),baseline.p,previous.size()*4,cudaMemcpyDeviceToHost,stream.p));ck(cudaStreamSynchronize(stream.p));
                check_lanes(weight,q.data(),columns,rows,integers,false);const auto expected=cpu_dots(weight,q.data(),columns,rows);equal_float(expected,got,"CPU/new IMMA output");equal_float(previous,got,"old GpuDeviceDots/new IMMA output");++matrix_cases;
                report["cases"].push_back({{"layer",layer},{"role",role},{"expert",expert},{"input_token",token},{"columns",columns},{"rows",rows},{"full_rows",full},{"pass",true}});
            }
            if(layer==10&&expert==0){
                const auto * x=real.data()+63*columns;ck(cudaMemcpyAsync(dx.p,x,size_t(columns)*4,cudaMemcpyHostToDevice,stream.p));
                auto prior=[&](const void * p){old.multiply(GGML_TYPE_IQ3_XXS,p,columns,full_rows,1,static_cast<float *>(dx.p),static_cast<float *>(baseline.p),false);};
                auto next=[&](const void * p){gpu_quantize_cpu_compatible(GGML_TYPE_Q8_K,static_cast<float *>(dx.p),gpuq.p,columns,1,stream.p);probe::multiply(p,gpuq.p,columns,full_rows,scratch.p,static_cast<float *>(candidate.p),stream.p);};
                for(int warm=0;warm<5;++warm){prior(dw.p);next(dw.p);}ck(cudaStreamSynchronize(stream.p));
                const double warm_old=event_us(stream.p,[&](int){prior(dw.p);},32),warm_new=event_us(stream.p,[&](int){next(dw.p);},32);
                const double one=event_us(stream.p,[&](int){probe::stage1(dw.p,gpuq.p,columns,full_rows,scratch.p,stream.p);},32),two=event_us(stream.p,[&](int){probe::stage2(dw.p,gpuq.p,columns,full_rows,scratch.p,static_cast<float *>(candidate.p),stream.p);},32);
                const size_t replicas=size_t(prop.l2CacheSize)/bytes+2;need(replicas>=2&&replicas<=64&&(replicas-1)*bytes>size_t(prop.l2CacheSize),"cold rotation distance");Device cold(bytes*replicas);
                for(size_t i=0;i<replicas;++i)ck(cudaMemcpyAsync(static_cast<uint8_t *>(cold.p)+i*bytes,weight,bytes,cudaMemcpyHostToDevice,stream.p));ck(cudaStreamSynchronize(stream.p));
                std::vector<double> cold_old,cold_new;size_t epoch=0;
                for(int repeat=0;repeat<32;++repeat){for(int arm=0;arm<2;++arm){const bool candidate_arm=(arm+repeat)%2;const auto * p=static_cast<const uint8_t *>(cold.p)+(epoch++%replicas)*bytes;
                    const double us=event_us(stream.p,[&](int){if(candidate_arm)next(p);else prior(p);},1);(candidate_arm?cold_new:cold_old).push_back(us);}}
                report["timings"].push_back({{"layer",layer},{"role",role},{"weight_bytes",bytes},{"scratch_bytes",probe::scratch_bytes(columns,full_rows)},
                    {"warm_old_pipeline_us",warm_old},{"warm_new_pipeline_us",warm_new},{"warm_stage1_us",one},{"warm_stage2_us",two},
                    {"cold_old_median_us",median(cold_old)},{"cold_new_median_us",median(cold_new)},{"cold_replicas",replicas},{"min_reuse_distance_bytes",(replicas-1)*bytes},{"actual_l2_bytes",prop.l2CacheSize}});
            }
        }
    }
    report["pass"]=true;report["raw_cases"]=raw_cases;report["matrix_cases"]=matrix_cases;report["integer_values"]=compared_i32;report["float_values"]=compared_f32;
    report["scope"]="Isolated IQ3n1 CPU-compatible masked-int8IMMA Kblockpartials + original8laneFMA finish. Original57layers gate/up, expert0/127, real128x6144L3activationrows; full3072rows on4layers,64rows elsewhere. Basis256positions and validraw[-127,127], no productionintegration/quality/TGclaim. Timings quant+dot pipelines; cold capacity rotation not measuredL2miss counters.";
    ggml_free(metadata);metadata=nullptr;ggml_backend_free(cpu);cpu=nullptr;std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';out.flush();need(bool(out),"IQ3 report write failed");return 0;
}catch(const std::exception & e){std::fprintf(stderr,"IQ3 lane component: %s\n",e.what());report["pass"]=false;report["error"]=e.what();report["integer_values"]=compared_i32;report["float_values"]=compared_f32;
    if(metadata)ggml_free(metadata);if(cpu)ggml_backend_free(cpu);if(write_allowed){std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';}return 1;}}
