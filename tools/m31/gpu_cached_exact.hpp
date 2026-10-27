#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::m31 {
// Opt-in exact cached subset for n1 or short verifier batches. A persistent worker owns one CUDA stream;
// old CPU-compatible dots execute without changing the native prefill backend.
class GpuCachedExact {
public:
    struct Stats {
        uint64_t jobs=0, experts=0, joins=0, h2d_bytes=0, d2h_bytes=0;
        double worker_seconds=0, gate_up_seconds=0, oai_seconds=0, down_seconds=0;
        size_t device_bytes=0, pinned_bytes=0;
        uint64_t grouped_jobs=0,grouped_assignments=0,grouped_unique_experts=0,multiply_calls=0;
        uint64_t joined_gu_jobs=0,joined_gu_assignments=0,joined_gu_groups=0;
        uint64_t iq3_imma_calls=0,iq3_imma_rows=0;
    };
    struct Assignment {int token=0,rank=0,slot=0;};
    GpuCachedExact(int embd,int ff,int maximum_used);
    GpuCachedExact(int embd,int ff,int maximum_used,int maximum_tokens);
    ~GpuCachedExact();
    GpuCachedExact(const GpuCachedExact&)=delete;
    GpuCachedExact&operator=(const GpuCachedExact&)=delete;
    // All vectors are copied into fixed owned storage before return. Weight
    // pools/slots remain borrowed through finish/drain. Only cached hits appear.
    void begin(const std::array<int,3>&types,const std::array<const void*,3>&pools,
        size_t slot_stride,const std::vector<float>&input,
        const std::vector<int32_t>&slots,const std::vector<float>&weights,
        const std::array<std::vector<float>,3>&scales);
    // Token-major input[embd,n], ordered unique (token,rank)
    // assignments. Slot IDs may repeat across tokens. No per-token joins; the
    // worker retains exact n1 quant/dot/OAI for every assignment and joins twice.
    void begin_batched(const std::array<int,3>&types,const std::array<const void*,3>&pools,
        size_t slot_stride,const std::vector<float>&input,int n,
        const std::vector<Assignment>&assignments,const std::vector<float>&weights,
        const std::array<std::vector<float>,3>&scales);
    // Already weighted [embd,hit] terms, preserving assignment order.
    void finish(std::vector<float>&terms);
    void drain() noexcept;
    Stats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
