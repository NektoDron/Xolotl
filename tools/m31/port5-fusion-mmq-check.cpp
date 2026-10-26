// Minimal original-weight Q6_K gate/up+OAI fusion diagnostic; no LLM/arena/KV.
#include "m31_geometry.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "json.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
static void require(bool ok,const char * reason){if(!ok)throw std::runtime_error(reason);}
static std::vector<float> compute(ggml_backend_t backend,ggml_tensor * gate_weight,
        ggml_tensor * up_weight,const std::vector<float> & input,int n,bool fence) {
    auto ctx=ggml_init({4*1024*1024,nullptr,true}); require(ctx,"graph metadata");
    auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    struct Cleanup {ggml_backend_t backend;ggml_context *ctx;ggml_gallocr_t alloc;
        ~Cleanup(){ggml_backend_synchronize(backend);ggml_backend_cuda_graph_cache_clear(backend);ggml_gallocr_free(alloc);ggml_free(ctx);}}
        cleanup{backend,ctx,allocator};
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,gate_weight->ne[0],n);ggml_set_input(x);
    auto gate=ggml_mul_mat(ctx,gate_weight,x),up=ggml_mul_mat(ctx,up_weight,x);
    if(fence){ggml_set_output(gate);ggml_set_output(up);}
    auto output=ggml_swiglu_oai(ctx,gate,up,1.702f,7.0f);ggml_set_output(output);
    auto graph=ggml_new_graph_custom(ctx,256,false);ggml_build_forward_expand(graph,output);
    require(ggml_gallocr_alloc_graph(allocator,graph),"graph allocation");
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
    return result;
}
int main(int argc,char **argv) {
    ggml_backend_t backend=nullptr;ggml_context *weights=nullptr;ggml_backend_buffer_t storage=nullptr;
    try {
        require(std::getenv("M31_CARD_SESSION") && !std::strcmp(std::getenv("M31_CARD_SESSION"),"1"),"aictl/card_session required");
        const auto *visible=std::getenv("CUDA_VISIBLE_DEVICES");require(visible && !std::strcmp(visible,"0"),"only GPU0 allowed");
        const char *model=argc>1?argv[1]:"/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf";
        const auto file=strata::GgufModel::open(model);
        backend=ggml_backend_cuda_init(0);require(backend,"CUDA backend");
        weights=ggml_init({1024*1024,nullptr,true});require(weights,"weight metadata");
        ggml_tensor *tensor[2]{};const void *source[2]{};
        for(int i=0;i<2;++i){size_t shard=0;const auto *info=file.find(i?"blk.0.ffn_up.weight":"blk.0.ffn_gate.weight",&shard);
            require(info && info->shape.size()==2,"gate/up descriptor");
            tensor[i]=ggml_new_tensor_2d(weights,ggml_type(info->type),info->shape[0],info->shape[1]);source[i]=file.shard(shard).tensor_data(*info);}
        storage=ggml_backend_alloc_ctx_tensors(weights,backend);require(storage,"weight allocation");
        ggml_backend_buffer_set_usage(storage,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(int i=0;i<2;++i)ggml_backend_tensor_set(tensor[i],source[i],0,ggml_nbytes(tensor[i]));
        json cases=json::array();
        for(int n:{128,2968}){
            std::vector<float> input(size_t(tensor[0]->ne[0])*n);
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%100003)*.013f)*.2f;
            auto a=compute(backend,tensor[0],tensor[1],input,n,false);
            auto b=compute(backend,tensor[0],tensor[1],input,n,true);
            double squared=0,reference=0;float maximum=0;size_t different=0;
            for(size_t i=0;i<a.size();++i){maximum=std::max(maximum,std::abs(a[i]-b[i]));double d=double(a[i])-b[i];squared+=d*d;reference+=double(b[i])*b[i];different+=std::memcmp(&a[i],&b[i],sizeof(float))!=0;}
            cases.push_back({{"n",n},{"columns",tensor[0]->ne[0]},{"rows",tensor[0]->ne[1]},
                {"type",ggml_type_name(tensor[0]->type)},{"values",a.size()},{"different",different},
                {"maxabs",maximum},{"relative_l2",std::sqrt(squared/std::max(reference,1e-300))}});
        }
        ggml_backend_buffer_free(storage);storage=nullptr;ggml_free(weights);weights=nullptr;ggml_backend_free(backend);backend=nullptr;
        std::cout<<json({{"scope","Original blk0 Q6_K gate/up+OAI, normal fusion vs fenced matmul outputs; diagnostic, NOT speed"},{"cases",cases}}).dump(2)<<'\n';
    } catch(const std::exception &e){if(backend)ggml_backend_synchronize(backend);if(storage)ggml_backend_buffer_free(storage);if(weights)ggml_free(weights);if(backend)ggml_backend_free(backend);std::cerr<<e.what()<<'\n';return 1;}
}
