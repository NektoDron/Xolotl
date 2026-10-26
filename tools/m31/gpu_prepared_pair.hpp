#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::m31 {
class GpuHostExecutor;
// One full original layer batch. Upload disjoint expert groups, enqueue their
// gate/up computation, then upload the next group while the current GPU kernels
// run. Original expert histograms and route positions survive every subset.
class GpuPreparedPair {
public:
    explicit GpuPreparedPair(GpuHostExecutor & executor); // borrowed, outlives this
    ~GpuPreparedPair();
    GpuPreparedPair(const GpuPreparedPair &) = delete;
    GpuPreparedPair & operator=(const GpuPreparedPair &) = delete;
    static bool supported(int gate_type, int up_type, int n);
    void prepare(int gate_type, int up_type, int columns, int rows, int used, int n,
                 const std::vector<float> & input,
                 const std::vector<int32_t> & original_expert_ids);
    // logical_experts and physical_slots are parallel arrays for this group.
    // Each logical expert must occur in exactly one enqueue. ready is a borrowed
    // CUDA ggml event recorded after this group's uploads. Enqueue its wait
    // before the uploader records/reuses that event again. nullptr is allowed
    // only if the caller has already completed these uploads synchronously.
    // Weight pool storage and all slots remain valid through read_into/abort.
    void enqueue(const void * gate_pool, const void * up_pool, size_t slot_stride,
                 const std::vector<int32_t> & logical_experts,
                 const std::vector<int32_t> & physical_slots,
                 ggml_backend_event_t ready = nullptr);
    void read_into(std::vector<float> & gate, std::vector<float> & up);
    // Experimental tolerance-contract path: CUDA expf is not bit-identical to
    // the CPU libc expf. Keep this opt-in and pass model-logit gates separately.
    // Scales are indexed by ORIGINAL logical expert IDs; empty means all ones.
    // Only hidden is transferred to the host; device gate storage is reused.
    void read_oai_into(std::vector<float> & hidden,
                       const std::vector<float> & gate_scales = {},
                       const std::vector<float> & up_scales = {});
    void abort() noexcept; // drain outstanding raw CUDA consumers before slot reuse
    size_t scratch_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
