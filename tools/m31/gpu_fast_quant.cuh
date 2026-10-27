#pragma once
// Fast FFN activation conversion. The quantization lattice is the CPU ggml
// Q8_K/Q8_0 lattice; conversion to half is deliberate for tensor-core GEMM.
// This is a tolerance-contract primitive, not the exact verification path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <climits>
#include <stdexcept>

namespace strata::m31 {
// Activation-only compact Q8_K block. bsums are unnecessary for half GEMM.
// Scale and signed quants are exactly the ordinary CPU Q8_K lattice.
struct FastQ8KBlock { float d; int8_t qs[256]; };
static_assert(sizeof(FastQ8KBlock)==260);
namespace fast_quant_detail {
__device__ __forceinline__ void argmax_first(float & value, int & index) {
    for (int offset = 16; offset; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu,value,offset);
        const int oi = __shfl_down_sync(0xffffffffu,index,offset);
        if (fabsf(other) > fabsf(value) || (fabsf(other) == fabsf(value) && oi < index)) {
            value = other; index = oi;
        }
    }
}
__device__ __forceinline__ float q8k_inverse(float x,float &scale) {
    __shared__ float warp_value[8];
    __shared__ int warp_index[8];
    float value=x; int index=int(threadIdx.x);
    argmax_first(value,index);
    if ((threadIdx.x&31)==0) { warp_value[threadIdx.x/32]=value; warp_index[threadIdx.x/32]=index; }
    __syncthreads();
    if (threadIdx.x<32) {
        value = threadIdx.x<8 ? warp_value[threadIdx.x] : 0.f;
        index = threadIdx.x<8 ? warp_index[threadIdx.x] : 256;
        argmax_first(value,index);
        if (!threadIdx.x) warp_value[0]=value;
    }
    __syncthreads();
    const float maximum=warp_value[0];
    const float inverse=maximum?__fdiv_rn(-127.f,maximum):0.f;
    scale=maximum?__fdiv_rn(1.f,inverse):0.f;
    return inverse;
}
template<bool Q80>
__global__ void gather_half(const float * input, half * output, half * low, int columns,
                            const int32_t * routes, int used) {
    const int row = int(blockIdx.y);
    const int source_row = routes ? routes[row]/used : row;
    const int column = int(blockIdx.x)*256+int(threadIdx.x);
    const float x = input[size_t(source_row)*columns+column];
    float scale = 0.f, inverse = 0.f;
    if constexpr (Q80) {
        float maximum = fabsf(x);
        for (int offset=16; offset; offset >>= 1)
            maximum = fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        // AVX2 uses the original maximum for the reciprocal, then stores the
        // forward scale in FP16. Do not derive reciprocal from the rounded d.
        inverse = maximum ? __fdiv_rn(127.f,maximum) : 0.f;
        scale = __half2float(__float2half_rn(__fdiv_rn(maximum,127.f)));
    } else {
        inverse=q8k_inverse(x,scale);
    }
    const int q = min(127,__float2int_rn(__fmul_rn(x,inverse)));
    const float dequant=__fmul_rn(scale,float(q));
    const half high=__float2half_rn(dequant);
    output[size_t(row)*columns+column] = high;
    if (low) low[size_t(row)*columns+column]=__float2half_rn(__fsub_rn(dequant,__half2float(high)));
}
static __global__ void pack_q8k(const float * input,FastQ8KBlock * output,int columns) {
    const size_t block=blockIdx.x;
    const float x=input[block*256+threadIdx.x];
    float scale;
    const float inverse=q8k_inverse(x,scale);
    output[block].qs[threadIdx.x]=int8_t(min(127,__float2int_rn(__fmul_rn(x,inverse))));
    if(!threadIdx.x)output[block].d=scale;
}
static __global__ void gather_packed_q8k(const FastQ8KBlock * input,half *output,half *low,
                                int columns,const int32_t *routes,int used) {
    const size_t blocks_per_row=size_t(columns)/256;
    const size_t row=blockIdx.x/blocks_per_row,column_block=blockIdx.x%blocks_per_row;
    const size_t source=routes?size_t(routes[row]/used):row;
    const auto &block=input[source*blocks_per_row+column_block];
    const float value=__fmul_rn(block.d,float(block.qs[threadIdx.x]));
    const half high=__float2half_rn(value);
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    output[index]=high;
    if(low)low[index]=__float2half_rn(__fsub_rn(value,__half2float(high)));
}
static __global__ void gather_prequant_half(const half *input,const half *input_low,
                                    half *output,half *low,int columns,
                                    const int32_t *routes,int used) {
    const size_t blocks_per_row=size_t(columns)/256;
    const size_t row=blockIdx.x/blocks_per_row,column_block=blockIdx.x%blocks_per_row;
    const size_t source=routes?size_t(routes[row]/used):row;
    const size_t source_pair=(source*size_t(columns)+column_block*256)/2+threadIdx.x;
    const size_t target_pair=size_t(blockIdx.x)*128+threadIdx.x;
    reinterpret_cast<uint32_t *>(output)[target_pair]=reinterpret_cast<const uint32_t *>(input)[source_pair];
    if(low)reinterpret_cast<uint32_t *>(low)[target_pair]=reinterpret_cast<const uint32_t *>(input_low)[source_pair];
}
inline unsigned flat_blocks(int columns,int count) {
    if(columns<=0 || columns%256 || count<=0 || size_t(columns/256)*size_t(count)>INT_MAX)
        throw std::invalid_argument("M31 fast prequant: invalid shape");
    return unsigned(size_t(columns/256)*size_t(count));
}
inline void check_launch() {
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace fast_quant_detail

// Each output row is a contiguous column-vector for column-major GEMM.
// With routes, source row is routes[row]/used (original flattened route ID).
// Without routes, input/output have the same contiguous row geometry.
// Input/output/routes remain owned by the caller through stream completion.
inline void fast_quant_gather_half(const float * input, half * output, int columns,
                                  int count, const int32_t * routes, int used,
                                  cudaStream_t stream, bool q8_0, half * low=nullptr) {
    if (!input || !output || columns<=0 || columns%256 || count<=0 || count>65535 || used<=0)
        throw std::invalid_argument("M31 fast quant: invalid shape or pointer");
    const dim3 grid(unsigned(columns/256),unsigned(count));
    if (q8_0) fast_quant_detail::gather_half<true><<<grid,256,0,stream>>>(input,output,low,columns,routes,used);
    else fast_quant_detail::gather_half<false><<<grid,256,0,stream>>>(input,output,low,columns,routes,used);
    const auto status=cudaGetLastError();
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline size_t fast_q8k_packed_bytes(int columns,int count) {
    return size_t(fast_quant_detail::flat_blocks(columns,count))*sizeof(FastQ8KBlock);
}
// Quantize each original token only once, before expert grouping. At c8192,
// columns6144 this costs 48.75MiB, versus 192MiB for high+low half storage.
inline void fast_quant_q8k_packed(const float *input,void *output,int columns,int count,cudaStream_t stream) {
    if(!input || !output)throw std::invalid_argument("M31 fast prequant: null pointer");
    fast_quant_detail::pack_q8k<<<fast_quant_detail::flat_blocks(columns,count),256,0,stream>>>(input,static_cast<FastQ8KBlock *>(output),columns);
    fast_quant_detail::check_launch();
}
inline void fast_gather_prequant_q8k_half(const void *input,half *output,int columns,
                                        int count,const int32_t *routes,int used,
                                        cudaStream_t stream,half *low=nullptr) {
    if(!input || !output || used<=0)throw std::invalid_argument("M31 fast prequant: invalid pointers/used");
    fast_quant_detail::gather_packed_q8k<<<fast_quant_detail::flat_blocks(columns,count),256,0,stream>>>(
        static_cast<const FastQ8KBlock *>(input),output,low,columns,routes,used);
    fast_quant_detail::check_launch();
}
// Optional larger-memory variant. Copies both halves without arithmetic.
// Buffers must not overlap; matching null low pointers select high-only mode.
inline void fast_gather_prequant_half(const half *input,const half *input_low,
                                    half *output,half *low,int columns,int count,
                                    const int32_t *routes,int used,cudaStream_t stream) {
    if(!input || !output || input==output || bool(input_low)!=bool(low) || used<=0)
        throw std::invalid_argument("M31 fast half gather: invalid pointers/used");
    fast_quant_detail::gather_prequant_half<<<fast_quant_detail::flat_blocks(columns,count),128,0,stream>>>(
        input,input_low,output,low,columns,routes,used);
    fast_quant_detail::check_launch();
}
} // namespace strata::m31
