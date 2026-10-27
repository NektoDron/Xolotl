#pragma once
#include <cstddef>
// Isolated research component only. Not included by Engine or GpuDeviceDots.
namespace strata::m31::iq3_lane_probe {
// Scratch I32 layout [weight_row,K256block,CPU_AVX2_lane8]. Activation Q8_K
// bytes are the unchanged ggml layout. Production finite quant domain is
// qs[-127,127]; raw -128 has different stock AVX2 psignb semantics.
size_t scratch_bytes(int columns,int rows);
void stage1(const void * weights,const void * q8k,int columns,int rows,
            void * scratch,void * borrowed_cuda_stream);
void stage2(const void * weights,const void * q8k,int columns,int rows,
            const void * scratch,float * output,void * borrowed_cuda_stream);
void multiply(const void * weights,const void * q8k,int columns,int rows,
              void * scratch,float * output,void * borrowed_cuda_stream);
}
