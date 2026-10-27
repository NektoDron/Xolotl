#pragma once
#include <cstddef>
// Exact IQ3 n1 integer MMA stages; private scratch and stream belong to GpuDeviceDots.
namespace strata::m31::iq3_lane_exact {
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
