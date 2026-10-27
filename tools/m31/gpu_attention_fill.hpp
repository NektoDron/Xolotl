#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <memory>
namespace strata::m31 {
// Prototype: fill the existing persistent attention inputs directly on CUDA.
// No tensor allocation and no quadratic host mask. fill() joins its stream
// before returning; caller keeps these descriptors/buffers alive until return.
class GpuAttentionFill {
    struct Impl;
    std::unique_ptr<Impl> p_;
public:
    explicit GpuAttentionFill(ggml_backend_t backend);
    ~GpuAttentionFill();
    GpuAttentionFill(const GpuAttentionFill &)=delete;
    GpuAttentionFill &operator=(const GpuAttentionFill &)=delete;
    void fill(ggml_tensor *positions,ggml_tensor *mask,ggml_tensor *bias,
              ggml_tensor *cell_blocks,int64_t position,int block,int local_blocks);
};
}
