#include "gpu_attention_fill.hpp"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::m31 {
namespace {
void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string("M31 GPU attention fill: ")+cudaGetErrorString(error));}
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(std::string("M31 GPU attention fill: ")+message);}
template<class Mask> __global__ void fill_kernel(int32_t *positions,Mask *mask,float *bias,int32_t *blocks,
    int64_t pos,int64_t keys,int n,int block,int local_blocks,uint64_t total,uint64_t bias_total){
    const uint64_t stride=uint64_t(blockDim.x)*gridDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=stride){
        const int64_t query=int64_t(i/uint64_t(keys)),key=int64_t(i%uint64_t(keys));
        if constexpr(sizeof(Mask)==2)mask[i]=key<=pos+query?uint16_t(0):uint16_t(0xfc00);
        else mask[i]=key<=pos+query?0.f:__int_as_float(int(0xff800000u));
        if(i<bias_total){
            const int64_t count=keys/block,b=int64_t(i%uint64_t(count));
            const int64_t current=(pos+int64_t(i/uint64_t(count)))/block;
            bias[i]=(b<=current&&current-b<local_blocks)?1e30f:0.f;
        }
        if(i<uint64_t(n))positions[i]=int32_t(pos+int64_t(i));
        if(blocks&&i<uint64_t(keys))blocks[i]=int32_t(i/uint64_t(block));
    }
}
}
struct GpuAttentionFill::Impl {
    ggml_backend_t backend=nullptr;
    cudaStream_t stream=nullptr;
    ~Impl(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
GpuAttentionFill::GpuAttentionFill(ggml_backend_t backend):p_(std::make_unique<Impl>()){
    const char *session=std::getenv("M31_CARD_SESSION"),*visible=std::getenv("CUDA_VISIBLE_DEVICES");
    require(session&&std::strcmp(session,"1")==0&&visible&&std::strcmp(visible,"0")==0,"requires aictl/flock GPU0 session");
    require(backend&&ggml_backend_is_cuda(backend),"requires CUDA backend");
    int device=-1;ck(cudaGetDevice(&device));require(device==0,"requires current device0");
    p_->backend=backend;ck(cudaStreamCreateWithFlags(&p_->stream,cudaStreamNonBlocking));
}
GpuAttentionFill::~GpuAttentionFill()=default;
void GpuAttentionFill::fill(ggml_tensor *positions,ggml_tensor *mask,ggml_tensor *bias,
    ggml_tensor *cell_blocks,int64_t position,int block,int local_blocks){
    require(positions&&mask&&bias,"missing attention tensors");
    const int64_t n=positions->ne[0],keys=mask->ne[0];
    require(n>0&&n<=32768&&position>=0&&position<=INT32_MAX-n&&keys>=position+n&&keys<=INT32_MAX,
            "invalid position/shape");
    require(block>0&&block<=256&&256%block==0&&keys%block==0&&local_blocks>=0,"invalid block geometry");
    require(positions->type==GGML_TYPE_I32&&ggml_nelements(positions)==n&&
            (mask->type==GGML_TYPE_F16||mask->type==GGML_TYPE_F32)&&mask->ne[1]==1&&mask->ne[2]==n&&mask->ne[3]==1&&
            bias->type==GGML_TYPE_F32&&bias->ne[0]==keys/block&&bias->ne[1]==1&&bias->ne[2]==n&&bias->ne[3]==1,
            "unexpected tensor shape/type");
    if(cell_blocks)require(cell_blocks->type==GGML_TYPE_I32&&ggml_nelements(cell_blocks)==keys,"unexpected cell block shape");
    for(auto *tensor:{positions,mask,bias,cell_blocks})if(tensor){
        auto *buffer=tensor->view_src?tensor->view_src->buffer:tensor->buffer;
        require(ggml_is_contiguous(tensor)&&tensor->data&&buffer&&
                ggml_backend_dev_supports_buft(ggml_backend_get_device(p_->backend),ggml_backend_buffer_get_type(buffer))&&
                !ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer)),"tensor must be contiguous GPU data");
    }
    ggml_backend_synchronize(p_->backend);
    const uint64_t total=uint64_t(keys)*uint64_t(n),bias_total=uint64_t(keys/block)*uint64_t(n);
    const unsigned grid=unsigned(std::min<uint64_t>((total+255)/256,4096));
    auto *out_positions=static_cast<int32_t*>(positions->data),*out_blocks=cell_blocks?static_cast<int32_t*>(cell_blocks->data):nullptr;
    if(mask->type==GGML_TYPE_F16)fill_kernel<<<grid,256,0,p_->stream>>>(out_positions,static_cast<uint16_t*>(mask->data),static_cast<float*>(bias->data),out_blocks,position,keys,int(n),block,local_blocks,total,bias_total);
    else fill_kernel<<<grid,256,0,p_->stream>>>(out_positions,static_cast<float*>(mask->data),static_cast<float*>(bias->data),out_blocks,position,keys,int(n),block,local_blocks,total,bias_total);
    const auto launch=cudaGetLastError();
    // Join even on launch failure before the caller can discard/reuse tensors.
    const auto completed=cudaStreamSynchronize(p_->stream);
    ck(launch);ck(completed);
}
}
