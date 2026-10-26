#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<float> run(ggml_backend_t backend,int width,int heads,int tokens,bool rope,bool fence) {
    auto * ctx=ggml_init({4*1024*1024,nullptr,true});
    auto * graph=ggml_new_graph_custom(ctx,256,false);
    auto * x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,heads,tokens);
    auto * w=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
    auto * p=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tokens);
    for(auto * t:{x,w,p})ggml_set_input(t);
    auto * norm=ggml_rms_norm(ctx,x,1e-5f);
    if(fence)ggml_set_output(norm);
    auto * out=ggml_mul(ctx,norm,w);
    if(rope)out=ggml_rope_ext(ctx,out,p,nullptr,64,GGML_ROPE_TYPE_NEOX,65536,1000000.0f,1,0,1,32,1);
    ggml_set_output(out);ggml_build_forward_expand(graph,out);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(alloc,graph))throw std::runtime_error("allocate");
    std::vector<float> input(size_t(width)*heads*tokens),weight(width);
    std::vector<int32_t> positions(tokens);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.23f)*(.3f+float(i%19));
    for(int i=0;i<width;++i)weight[i]=.7f+std::sin(float(i)*.07f);
    for(int i=0;i<tokens;++i)positions[i]=i+2700;
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    ggml_backend_tensor_set(w,weight.data(),0,weight.size()*4);
    if(rope)ggml_backend_tensor_set(p,positions.data(),0,positions.size()*4);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");
    ggml_backend_synchronize(backend);
    std::vector<float> output(input.size());ggml_backend_tensor_get(out,output.data(),0,output.size()*4);
    ggml_backend_cuda_graph_cache_clear(backend);ggml_gallocr_free(alloc);ggml_free(ctx);
    return output;
}
int main() {
    auto backend=ggml_backend_cuda_init(0);
    for(int tokens:{1,64})for(int width:{128,6144})for(bool rope:{false,true}){
        const int heads=width==128?8:1;
        auto a=run(backend,width,heads,tokens,rope,false),b=run(backend,width,heads,tokens,rope,true);
        float maximum=0;size_t different=0;
        for(size_t i=0;i<a.size();++i){maximum=std::max(maximum,std::abs(a[i]-b[i]));different+=std::memcmp(&a[i],&b[i],4)!=0;}
        std::cout<<"width="<<width<<" tokens="<<tokens<<" rope="<<rope<<" different="<<different<<" maxabs="<<maximum<<'\n';
    }
    ggml_backend_free(backend);
}
