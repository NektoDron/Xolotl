// Q2 CPU component proof: original GGUF bytes against independent per-token
// stock ggml graphs. No model quality or throughput inference from this fixture.
#include "cpu_batched_verify.hpp"
#include "cpu_fused_decode.hpp"
#include "m31_geometry.hpp"
#include <bit>
#include <cstdio>
#include <memory>
#include <string>
using namespace strata::m31;
namespace {
void check(bool pass, const char * why) { if (!pass) throw std::runtime_error(why); }
void equal(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    if (actual.size() != expected.size() || std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)))
        throw std::runtime_error(std::string("Q2 CPU bit mismatch: ")+label);
}
struct Layer { int index; std::array<ggml_tensor *,3> w{}; };
// This deliberately does not call any custom wrapper or CPU dot helper.
struct StockGraph {
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * terms = nullptr, * output = nullptr;
    ggml_backend_t backend;
    StockGraph(ggml_backend_t cpu, const Layer & layer, int used): backend(cpu) {
        context=ggml_init({1024*1024,nullptr,true}); check(context,"stock context");
        graph=ggml_new_graph_custom(context,128,false);
        const int64_t embd=layer.w[0]->ne[0];
        input=ggml_new_tensor_3d(context,GGML_TYPE_F32,embd,1,1);
        ids=ggml_new_tensor_2d(context,GGML_TYPE_I32,used,1);
        weights=ggml_new_tensor_3d(context,GGML_TYPE_F32,1,used,1);
        ggml_set_input(input); ggml_set_input(ids); ggml_set_input(weights);
        auto gate=ggml_mul_mat_id(context,layer.w[0],input,ids);
        auto up=ggml_mul_mat_id(context,layer.w[1],input,ids);
        auto h=ggml_swiglu_oai(context,gate,up,1.702f,7.f);
        auto down=ggml_mul_mat_id(context,layer.w[2],h,ids);
        terms=ggml_mul(context,down,weights); ggml_set_output(terms);
        ggml_tensor * sum=nullptr;
        for(int rank=0;rank<used;++rank) {
            auto term=ggml_view_2d(context,terms,embd,1,terms->nb[2],size_t(rank)*terms->nb[1]);
            sum=sum?ggml_add(context,sum,term):term;
        }
        output=ggml_cont(context,sum); ggml_set_output(output);
        ggml_build_forward_expand(graph,output);
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        check(allocator&&ggml_gallocr_alloc_graph(allocator,graph),"stock graph allocation");
    }
    ~StockGraph(){if(allocator)ggml_gallocr_free(allocator);if(context)ggml_free(context);}
    void run(const float * x,const int32_t * route,const float * rw,std::vector<float> & out,std::vector<float> & weighted) {
        ggml_backend_tensor_set(input,x,0,ggml_nbytes(input));
        ggml_backend_tensor_set(ids,route,0,ggml_nbytes(ids));
        ggml_backend_tensor_set(weights,rw,0,ggml_nbytes(weights));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"stock compute");
        out.resize(size_t(ggml_nelements(output))); weighted.resize(size_t(ggml_nelements(terms)));
        ggml_backend_tensor_get(output,out.data(),0,ggml_nbytes(output));
        ggml_backend_tensor_get(terms,weighted.data(),0,ggml_nbytes(terms));
    }
};
}
int main(int argc,char ** argv) {
 try {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    check(argc==3,"usage: q2_cpu_check MODEL THREADS");
    const int threads=std::stoi(argv[2]);check(threads>0&&threads<=64,"threads1..64");
    strata::GgufModel model(strata::gguf_split_paths(argv[1]));const auto g=read_geometry(model);
    auto context=ggml_init({1024*1024,nullptr,true});check(context,"tensor context");
    std::vector<Layer> layers;
    for(int index:{3,52,59}) {
        Layer layer;layer.index=index;const char * roles[]={"gate","up","down"};
        for(int role=0;role<3;++role) {
            const auto name=layer_name(index,std::string("ffn_")+roles[role]+"_exps.weight");
            size_t shard=0;auto info=model.find(name,&shard);check(info&&model.in_bounds(*info,shard),"real weight");
            auto * w=ggml_new_tensor_3d(context,ggml_type(info->type),info->shape[0],info->shape[1],info->shape[2]);
            w->data=const_cast<uint8_t *>(model.shard(shard).tensor_data(*info));layer.w[role]=w;
        }
        layers.push_back(layer);
    }
    check(layers[0].w[0]->type==GGML_TYPE_IQ2_XS&&layers[0].w[1]->type==GGML_TYPE_IQ2_XS&&layers[0].w[2]->type==GGML_TYPE_IQ3_XXS,"mixed IQ2/IQ3 fixture");
    check(layers[1].w[2]->type==GGML_TYPE_IQ4_XS&&layers[2].w[0]->type==GGML_TYPE_IQ3_XXS,"protected type fixtures");
    unsetenv("M31_CPU_STOCK_EXPERTS");
    check(!CpuDirectDecode::supported(layers[0].w[0],layers[0].w[1],layers[0].w[2]),"IQ2 default must remain rejected");
    check(CpuDirectDecode::supported(layers[2].w[0],layers[2].w[1],layers[2].w[2]),"legacy support default");
    for(const char * flag:{"0","01","1"}) {
        setenv("M31_CPU_STOCK_EXPERTS",flag,1);
        for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS,GGML_TYPE_NVFP4})
            check(CpuDirectDecode::supported_type(type),"legacy formats must ignore IQ2 optin");
        check(CpuDirectDecode::supported_type(GGML_TYPE_IQ2_XS)==(std::strcmp(flag,"1")==0),"IQ2 literal1 eligibility");
    }
    setenv("M31_CPU_STOCK_EXPERTS","01",1);check(!CpuDirectDecode::supported_type(GGML_TYPE_IQ2_XS),"only literal1 enables IQ2");
    setenv("M31_CPU_STOCK_EXPERTS","1",1);
    check(CpuDirectDecode::supported(layers[0].w[0],layers[0].w[1],layers[0].w[2]),"IQ2 optin support");
    check(ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XS)->from_float==nullptr,"weight from_float must not be required");
    unsigned guards=0;
    auto original=*layers[0].w[0];
    for(int fault=0;fault<8;++fault) {
        auto bad=original;
        if(fault==0)bad.ne[0]=0;if(fault==1)bad.ne[0]-=1;if(fault==2)bad.ne[1]=65537;
        if(fault==3)bad.ne[2]=129;if(fault==4)bad.ne[3]=2;if(fault==5)bad.nb[0]+=1;
        if(fault==6)bad.nb[1]+=1;if(fault==7)bad.nb[2]=bad.nb[1]*size_t(bad.ne[1])-1;
        check(!CpuDirectDecode::supported_weight(&bad),"IQ2 malformed geometry accepted");++guards;
    }
    auto padded=original;padded.nb[2]+=4096;check(CpuDirectDecode::supported_weight(&padded),"padded expert stride rejected");
    auto cpu=ggml_backend_cpu_init();check(cpu,"CPU backend");ggml_backend_cpu_set_n_threads(cpu,threads);
    auto params=ggml_threadpool_params_default(threads);apply_cpu_pool_environment(params);
    auto pool=ggml_threadpool_new(&params);check(pool,"persistent pool");ggml_backend_cpu_set_threadpool(cpu,pool);
    uint64_t reference_rows=0,comparisons=0,float_values=0,masked_cases=0;
    {
        CpuDirectDecode direct{unsigned(threads)};
        for(const auto & layer:layers) {
            StockGraph stock(cpu,layer,4);
            for(int tokens:{1,2,4,8})for(int pattern=0;pattern<3;++pattern) {
                std::vector<float> x(size_t(tokens)*size_t(g.embd)),rw(size_t(tokens)*4),expected,expected_terms,one,terms;
                std::vector<int32_t> ids(size_t(tokens)*4);
                for(int token=0;token<tokens;++token) {
                    for(int i=0;i<g.embd;++i)x[size_t(token)*g.embd+i]=std::sin(float(i)*.017f+float(token)*.731f)*(pattern==0?.03125f:pattern==1?1.f:8.f);
                    for(int rank=0;rank<4;++rank){const int shift=pattern==0?0:pattern==1?token:token*11;ids[size_t(token)*4+rank]=(7+shift+rank*29)%int(g.experts);rw[size_t(token)*4+rank]=(rank==1?-.125f:1.f/float(5+token+rank));}
                    stock.run(x.data()+size_t(token)*g.embd,ids.data()+size_t(token)*4,rw.data()+size_t(token)*4,one,terms);
                    expected.insert(expected.end(),one.begin(),one.end());expected_terms.insert(expected_terms.end(),terms.begin(),terms.end());++reference_rows;
                }
                auto compare=[&](const std::vector<float> & actual,const std::vector<float> & ref,const char * label){equal(actual,ref,label);++comparisons;float_values+=actual.size();};
                std::vector<float> actual;
                for(int token=0;token<tokens;++token) {
                    std::vector<float> ox(x.begin()+size_t(token)*g.embd,x.begin()+size_t(token+1)*g.embd);
                    std::vector<int32_t> oi(ids.begin()+size_t(token)*4,ids.begin()+size_t(token+1)*4);
                    std::vector<float> ow(rw.begin()+size_t(token)*4,rw.begin()+size_t(token+1)*4);
                    direct.forward(layer.w[0],layer.w[1],layer.w[2],nullptr,nullptr,nullptr,ox,oi,ow,one);
                    actual.insert(actual.end(),one.begin(),one.end());
                }
                compare(actual,expected,"direct versus stock graph");
                for(int mode:{0,1,2}) {
                    setenv("M31_CPU_FUSED_PAIRED_QUANT",std::to_string(mode).c_str(),1);CpuFusedDecode fused(cpu,pool,unsigned(threads));
                    actual.clear();
                    for(int token=0;token<tokens;++token) {
                        std::vector<float> ox(x.begin()+size_t(token)*g.embd,x.begin()+size_t(token+1)*g.embd);
                        std::vector<int32_t> oi(ids.begin()+size_t(token)*4,ids.begin()+size_t(token+1)*4);
                        std::vector<float> ow(rw.begin()+size_t(token)*4,rw.begin()+size_t(token+1)*4);
                        fused.forward(layer.w[0],layer.w[1],layer.w[2],nullptr,nullptr,nullptr,ox,oi,ow,one);
                        actual.insert(actual.end(),one.begin(),one.end());
                    }
                    compare(actual,expected,"fused versus stock graph");
                }
                for(int mode:{0,1,2}) {
                    setenv("M31_CPU_BATCH_PAIR",std::to_string(mode).c_str(),1);CpuBatchedVerify batch(cpu,pool,unsigned(threads));
                    batch.forward(layer.w[0],layer.w[1],layer.w[2],nullptr,nullptr,nullptr,x,ids,rw,actual);
                    compare(actual,expected,"batched versus stock graph");
                    compare(batch.contributions(),expected_terms,"weighted terms versus stock graph");
                    for(int maskcase=0;maskcase<3;++maskcase) {
                        std::vector<uint8_t> mask(ids.size());for(size_t i=0;i<mask.size();++i)mask[i]=maskcase==0?1:maskcase==1?0:i%2;
                        batch.forward_masked(layer.w[0],layer.w[1],layer.w[2],nullptr,nullptr,nullptr,x,ids,rw,mask,actual);
                        auto masked_ref=expected_terms;
                        for(size_t i=0;i<mask.size();++i)if(!mask[i])std::fill_n(masked_ref.data()+i*size_t(g.embd),size_t(g.embd),0.f);
                        compare(actual,masked_ref,"masked terms versus stock graph");++masked_cases;
                        batch.forward(layer.w[0],layer.w[1],layer.w[2],nullptr,nullptr,nullptr,x,ids,rw,actual);
                        compare(actual,expected,"unmasked state recovery");
                    }
                }
                std::printf("{\"kind\":\"case\",\"layer\":%d,\"tokens\":%d,\"pattern\":%d,\"bitwise\":true}\n",layer.index,tokens,pattern);
            }
        }
    }
    std::printf("{\"kind\":\"summary\",\"pass\":true,\"layers\":[3,52,59],\"batch_sizes\":[1,2,4,8],\"routes\":4,\"patterns\":3,\"reference_token_rows\":%llu,\"comparisons\":%llu,\"float_values\":%llu,\"masked_cases\":%llu,\"geometry_guards\":%u,\"threads\":%d,\"gpu_used\":false,\"reference\":\"stock per-token ggml mul_mat_id/swiglu_oai/down/weighted original-rank sum\"}\n",(unsigned long long)reference_rows,(unsigned long long)comparisons,(unsigned long long)float_values,(unsigned long long)masked_cases,guards,threads);
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);ggml_free(context);
 }catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
