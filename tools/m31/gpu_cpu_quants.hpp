// Diagnostic opt-in: original weights on CUDA with CPU-compatible activation quants.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::m31 {
class GpuHostExecutor;
class GpuCpuQuants {
public:
    GpuCpuQuants();
    ~GpuCpuQuants();
    GpuCpuQuants(const GpuCpuQuants &) = delete;
    GpuCpuQuants & operator=(const GpuCpuQuants &) = delete;
    // Matrix weights [columns,rows,slots] already reside on GPU. Input is either
    // [columns,1,n] or [columns,used,n]. ids [used,n] select the weight slot.
    std::vector<float> multiply(int type, const void * weights, size_t slot_stride,
                               int columns, int rows, int used, int n,
                               const std::vector<float> & input,
                               const std::vector<int32_t> & ids);
    // Reuse caller-owned readback storage across layers/chunks. Input and output
    // vector objects must differ. Arithmetic and output layout match multiply().
    void multiply_into(int type, const void * weights, size_t slot_stride,
                       int columns, int rows, int used, int n,
                       const std::vector<float> & input,
                       const std::vector<int32_t> & ids, std::vector<float> & output);
    size_t scratch_bytes() const;
    // Cumulative counters can be emitted while an HTTP worker is still alive.
    // Device kernel events are collected only with M31_GPU_CPU_QUANTS_TIMING=1.
    void print_profile(const char * label = "runtime") const;
    GpuHostExecutor & host_executor();
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
