// Isolated CPU-compatible IQ3 n1 prototype. Existing weights/activation layouts
// and production dispatch are unchanged. Integer work is parallel over K256;
// the epilogue retains the original eight AVX2 F32 accumulation lanes.
// Fragment layout: NVIDIA PTX ISA9.4, Matrix Fragments for mma.m16n8k32
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832
#include "port7_iq3_lane_probe.hpp"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::m31::iq3_lane_probe {
namespace {
static_assert(sizeof(block_iq3_xxs)==98&&sizeof(block_q8_K)==292,"original ggml block layouts");
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(std::string("IQ3 lane IMMA: ")+cudaGetErrorString(rc));}
__device__ __forceinline__ uint32_t packed_scale_signs(const block_iq3_xxs & x,int sub){
    const auto * p=x.qs+64+sub*4;
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
__device__ __forceinline__ uint32_t weight_word(const block_iq3_xxs & x,int sub,int group,uint32_t packed){
    const uint8_t signs=ksigns_iq2xs[(packed>>(7*(group/2)))&127];
    const uint32_t grid=iq3xxs_grid[x.qs[sub*8+group]];uint32_t result=0;
    #pragma unroll
    for(int j=0;j<4;++j){const int magnitude=int((grid>>(8*j))&255);
        const int value=(signs&(1<<((group*4+j)%8)))?-magnitude:magnitude;
        result|=uint32_t(uint8_t(value))<<(8*j);
    }
    return result;
}
__device__ __forceinline__ void imma32(int (&dot)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    dot[0]=dot[1]=dot[2]=dot[3]=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__global__ void partials_kernel(const block_iq3_xxs * weights,const block_q8_K * rhs,
        int blocks,int rows,int32_t * partials){
    // Four independent warps per CTA, each handles16weight rows and one K256.
    // No shared memory/barriers. Tail return is whole-warp, never partial MMA.
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,b=int(blockIdx.y)*4+warp;
    if(b>=blocks)return;
    const int group=lane>>2,word=lane&3,row0=int(blockIdx.x)*16+group,row1=row0+8;
    const auto * x0=row0<rows?weights+size_t(row0)*blocks+b:nullptr;
    const auto * x1=row1<rows?weights+size_t(row1)*blocks+b:nullptr;
    const auto * qwords=reinterpret_cast<const uint32_t *>(rhs[b].qs);
    int32_t total[4]={};
    #pragma unroll
    for(int sub=0;sub<8;++sub){
        const uint32_t bits0=x0?packed_scale_signs(*x0,sub):0,bits1=x1?packed_scale_signs(*x1,sub):0;
        const uint32_t a0=x0?weight_word(*x0,sub,word,bits0):0,a1=x1?weight_word(*x1,sub,word,bits1):0;
        const uint32_t a2=x0?weight_word(*x0,sub,word+4,bits0):0,a3=x1?weight_word(*x1,sub,word+4,bits1):0;
        // NVIDIA PTX m16n8k32 s8 B column = lane/4; its two words cover
        // K[word*4..+3] and K[16+word*4..+3]. Mask all other CPU lanes.
        const uint32_t b0=group==word?qwords[sub*8+word]:0;
        const uint32_t b1=group==word+4?qwords[sub*8+word+4]:0;
        int dot[4];imma32(dot,a0,a1,a2,a3,b0,b1);
        const int scale0=x0?2*int(bits0>>28)+1:0,scale1=x1?2*int(bits1>>28)+1:0;
        total[0]+=dot[0]*scale0;total[1]+=dot[1]*scale0;
        total[2]+=dot[2]*scale1;total[3]+=dot[3]*scale1;
    }
    // PTX accumulator D: row=(lane/4)+(i/2)*8, col=2*(lane%4)+i%2.
    #pragma unroll
    for(int i=0;i<4;++i){const int row=row0+(i/2)*8,cpu_lane=2*word+i%2;
        if(row<rows)partials[(size_t(row)*blocks+b)*8+cpu_lane]=total[i];}
}
__global__ void finish_kernel(const block_iq3_xxs * weights,const block_q8_K * rhs,
        int blocks,int rows,const int32_t * partials,float * output){
    const int cpu_lane=threadIdx.x&7,row=int((size_t(blockIdx.x)*blockDim.x+threadIdx.x)/8);
    if(row>=rows)return;
    const unsigned mask=0xffu<<(threadIdx.x&24);float acc=0;
    for(int b=0;b<blocks;++b){
        const float d=__fmul_rn(__half2float(weights[size_t(row)*blocks+b].d),rhs[b].d);
        const int32_t isum=partials[(size_t(row)*blocks+b)*8+cpu_lane];
        acc=__fmaf_rn(d,__int2float_rn(isum),acc);
    }
    acc=__fadd_rn(acc,__shfl_xor_sync(mask,acc,4,8));
    acc=__fadd_rn(acc,__shfl_xor_sync(mask,acc,2,8));
    acc=__fadd_rn(acc,__shfl_xor_sync(mask,acc,1,8));
    if(!cpu_lane)output[row]=__fmul_rn(acc,0.25f);
}
void inputs(const void * w,const void * q,int columns,int rows,const void * scratch){
    if(!w||!q||!scratch||columns<=0||columns%256||rows<=0||rows>1048576)
        throw std::invalid_argument("IQ3 lane IMMA input/shape");
    (void)scratch_bytes(columns,rows);
}
}
size_t scratch_bytes(int columns,int rows){
    if(columns<=0||columns%256||rows<=0||rows>1048576)throw std::invalid_argument("IQ3 lane IMMA shape");
    const uint64_t bytes=uint64_t(columns/256)*uint64_t(rows)*8*sizeof(int32_t);
    if(bytes>std::numeric_limits<size_t>::max())throw std::invalid_argument("IQ3 lane IMMA scratch overflow");
    return size_t(bytes);
}
void stage1(const void * w,const void * q,int columns,int rows,void * scratch,void * stream){
    inputs(w,q,columns,rows,scratch);const int blocks=columns/256;
    partials_kernel<<<dim3(unsigned((rows+15)/16),unsigned((blocks+3)/4)),128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const block_iq3_xxs *>(w),static_cast<const block_q8_K *>(q),blocks,rows,static_cast<int32_t *>(scratch));ck(cudaGetLastError());
}
void stage2(const void * w,const void * q,int columns,int rows,const void * scratch,float * output,void * stream){
    inputs(w,q,columns,rows,scratch);if(!output)throw std::invalid_argument("IQ3 lane IMMA output");
    finish_kernel<<<unsigned((size_t(rows)*8+127)/128),128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const block_iq3_xxs *>(w),static_cast<const block_q8_K *>(q),columns/256,rows,static_cast<const int32_t *>(scratch),output);ck(cudaGetLastError());
}
void multiply(const void * w,const void * q,int columns,int rows,void * scratch,float * output,void * stream){
    stage1(w,q,columns,rows,scratch,stream);stage2(w,q,columns,rows,scratch,output,stream);
}
}
