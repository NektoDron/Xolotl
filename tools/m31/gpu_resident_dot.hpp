#pragma once
#include <cstddef>
#include <memory>

namespace strata::m31 {
// CPU-compatible activation quantization entirely on the supplied CUDA stream.
// activation_type is GGML_TYPE_Q8_K or GGML_TYPE_Q8_0; output uses ordinary
// ggml block layouts (292 and 34 bytes respectively), not compact FastQ8KBlock.
// Input/output allocation and lifetimes belong to the caller.
void gpu_quantize_cpu_compatible(int activation_type,const float * device_input,
                                void * device_output,int columns,int count,
                                void * cuda_stream);

// One expert at a time, keeping activation quantization and all dot products
// on GPU. The borrowed stream must outlive this object. No host activation or
// result transfers occur. Calls enqueue work and return without synchronizing.
class GpuDeviceDots {
public:
    explicit GpuDeviceDots(void * borrowed_cuda_stream);
    ~GpuDeviceDots();
    GpuDeviceDots(const GpuDeviceDots &)=delete;
    GpuDeviceDots &operator=(const GpuDeviceDots &)=delete;
    // Reserve the largest gate/up/down part BEFORE overlapping weight DMA.
    // Accommodates activation widths up to max(columns,rows), rounding to a
    // whole Q8_K block. Subsequent multiply calls within this bound allocate
    // nothing. The caller ensures previous work has completed before growth.
    void reserve(int columns,int rows,int count_max);
    // Weights are exactly one original packed expert [columns,rows]. Input
    // and output contain count contiguous F32 vectors. original_tiled refers
    // to the FULL original layer histogram, even when count is a small part.
    // NVFP4 always uses its CPU vec-dot accumulation contract.
    void multiply(int type,const void * device_weights,int columns,int rows,
                  int count,const float * device_input,float * device_output,
                  bool original_tiled);
    void abort() noexcept;
    size_t scratch_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::m31
