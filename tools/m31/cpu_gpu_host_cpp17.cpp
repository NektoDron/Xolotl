// The GPU .cu target currently uses CUDA_STANDARD17; this public API must too.
#include "cpu_gpu_host_ops.hpp"
static_assert(__cplusplus >= 201703L);
void cpu_gpu_host_cpp17_header_probe(strata::m31::GpuHostExecutor & pool,
                                     const std::vector<float> & input, std::vector<uint8_t> & output) {
    strata::m31::gpu_host_quantize_rows(pool, GGML_TYPE_IQ3_XXS, 256, input, output);
}
