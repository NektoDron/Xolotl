#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::m31 {
// Resident routed FFN. Original packed weights and CPU-compatible activation
// quants are retained; default products use original IMMA/short-lane arithmetic.
// Experimental DOTS=cublas uses compensated half Tensor Core GEMM.
// Only the input and final weighted output cross the host/device boundary.
class GpuFastFfn {
public:
    GpuFastFfn();
    ~GpuFastFfn();
    GpuFastFfn(const GpuFastFfn &) = delete;
    GpuFastFfn & operator=(const GpuFastFfn &) = delete;
    static bool supported(int gate_type, int up_type, int down_type);
    void prepare(int gate_type, int up_type, int down_type,
                 int columns, int hidden, int used, int n,
                 const std::vector<float> & input,
                 const std::vector<int32_t> & original_expert_ids,
                 const std::vector<float> & route_weights);
    // Borrow the device input through finish_device/read_into/abort. The caller
    // keeps its allocation alive; an optional CUDA ggml fence orders norm writes.
    // This variant performs no F32 activation H2D/D2D copy.
    void prepare_device(int gate_type, int up_type, int down_type,
                 int columns, int hidden, int used, int n,
                 const float * device_input,
                 const std::vector<int32_t> & original_expert_ids,
                 const std::vector<float> & route_weights,
                 ggml_backend_event_t input_ready = nullptr);
    // Pools point at each role's base; common stride separates physical slots.
    // Logical experts are original IDs. Scales are indexed by logical expert.
    // Each route is enqueued once. Per-expert subbatches are bounded by
    // M31_GPU_FAST_FFN_EXPERT_BATCH (default1024); no routes are pruned.
    // Weight slots stay valid until read/abort. RANK_PHASES=1 requires one
    // enqueue containing every active expert, so rank phases stay globally ordered.
    // The borrowed upload event is consumed by cudaStreamWaitEvent before return.
    void enqueue(const void * gate_pool, const void * up_pool, const void * down_pool,
                 size_t slot_stride,
                 const std::vector<int32_t> & logical_experts,
                 const std::vector<int32_t> & physical_slots,
                 const std::vector<float> & gate_scales = {},
                 const std::vector<float> & up_scales = {},
                 const std::vector<float> & down_scales = {},
                 ggml_backend_event_t ready = nullptr);
    void read_into(std::vector<float> & output);
    // Produce either weighted FFN output (both addends null), or the next hidden
    // state (moe+shared)+residual using separate F32 RN additions. No F32 D2H.
    // All pointers are on this CUDA device; dest may alias an addend. Completes
    // the private stream before return, so borrowed ggml graph buffers may die.
    void finish_device(float * destination, const float * shared = nullptr,
                       const float * residual = nullptr);
    void abort() noexcept;
    size_t scratch_bytes() const;
    int expert_batch_rows() const;
    bool fused_reduction() const;
    bool rank_phases() const;
    void print_profile(const char * label = "runtime") const;
private:
    void prepare_impl(int gate_type, int up_type, int down_type,
                      int columns, int hidden, int used, int n,
                      const float * host_input, const float * device_input,
                      const std::vector<int32_t> & original_expert_ids,
                      const std::vector<float> & route_weights,
                      ggml_backend_event_t input_ready);
    void finish_impl(std::vector<float> * host_output, float * destination,
                     const float * shared, const float * residual);
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
