// M2 expert execution. Original GGUF quants, one cache shared by all layers.
#pragma once
#include "m31_arena.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::m31 {
namespace cache_seed { struct ExpectedRank; }

// Borrows the backend and arena, which must outlive this object. Serialized use only.
// The VRAM budget covers one fixed expert-slot pool; graph scratch/dense/KV are separate.
// A cache miss copies one complete expert slot through a small pinned staging buffer.
class GpuExperts {
public:
    struct Stats {
        uint64_t hits = 0, misses = 0, evictions = 0, upload_bytes = 0;
        uint64_t graph_batches = 0, forwards = 0;
        // Demand across valid routed forwards, and next-layer predictions classified
        // when that layer's actual routes arrive (not bandwidth/device timings).
        uint64_t required_experts = 0, required_bytes = 0;
        uint64_t prefetch_used_experts = 0, prefetch_unused_experts = 0;
        uint64_t prefetch_used_bytes = 0, prefetch_unused_bytes = 0;
        uint64_t prefetch_history_batches = 0, prefetch_history_skipped_experts = 0;
        uint64_t short_hybrid_forwards = 0, short_cpu_experts = 0, short_gpu_experts = 0;
        uint64_t short_cached_experts = 0, short_cpu_routes = 0, short_gpu_routes = 0;
        uint64_t short_cold_gpu_bytes = 0;
        double short_cpu_seconds = 0, short_wall_seconds = 0;
        uint64_t cached_forwards = 0;
        uint64_t hybrid_forwards = 0, decode_route_lookups = 0, decode_resident_routes = 0;
        double hybrid_submit_seconds = 0, hybrid_cpu_seconds = 0, hybrid_join_seconds = 0;
        size_t decode_scratch_bytes = 0;
        uint64_t decode_seed_primes = 0, decode_seed_upload_bytes = 0;
        double decode_seed_seconds = 0;
        uint64_t cached_batch_calls=0,cached_batch_forwards=0,cached_batch_tokens=0;
        uint64_t cached_batch_routes=0,cached_batch_hits=0,cached_batch_allmiss=0;
        double cached_batch_seed_seconds=0,cached_batch_submit_seconds=0;
        double cached_batch_cpu_seconds=0,cached_batch_join_seconds=0,cached_batch_wall_seconds=0;
        uint64_t fast_forwards = 0;
        // Wall time on the calling thread. Async upload_submit can overlap prior
        // DMA; execute includes device completion and must not be called pure kernel time.
        double prepare_seconds = 0, pack_seconds = 0, upload_submit_seconds = 0;
        double upload_wait_seconds = 0, execute_seconds = 0;
        double gate_seconds = 0, up_seconds = 0, oai_seconds = 0;
        double down_seconds = 0, reduce_seconds = 0;
        size_t seeded_slots = 0;
        size_t resident_bytes = 0, peak_resident_bytes = 0, budget_bytes = 0;
        size_t scratch_bytes = 0, minimum_budget_bytes = 0;
        size_t slot_stride = 0, slot_count = 0, pool_bytes = 0, occupied_slots = 0;
        bool pinned_staging = false;
        bool q8_activations = false;
        uint64_t prefetch_batches = 0, prefetched_experts = 0, prefetch_bytes = 0;
        double prefetch_wait_seconds = 0;
        double prefetch_pack_seconds = 0, prefetch_staging_wait_seconds = 0, prefetch_submit_seconds = 0;
    };
    GpuExperts();
    ~GpuExperts();
    GpuExperts(const GpuExperts &) = delete;
    GpuExperts & operator=(const GpuExperts &) = delete;

    // Throws on invalid budget/geometry/backend. Allocates a fixed bounded pool once;
    // no weights are uploaded until forward(). A batch must fit all its distinct experts.
    void init(const GgufModel & model, const Geometry & geometry, const ExpertArena & arena,
              ggml_backend_t backend, double cache_gib, int max_chunk, bool q8_activations = false);

    // x: [embd,n]; ids and weights: [used,n], all in contiguous GGML order.
    // Runs native ggml GPU math by default; custom resident FFN and CPU-exact
    // verification (M31_GPU_EXACT=1) require explicit opt-in. Contributions use original top-k order.
    // Uses OAI SwiGLU(alpha=1.702,limit=7) and each role's external NVFP4 scale.
    std::vector<float> forward(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & ids,
                               const std::vector<float> & weights);
    // Reuse caller-owned host capacity across layers/chunks. Output must not be
    // the same vector object as x or weights; on failure the caller must discard it.
    void forward_into(int layer, int n, const std::vector<float> & x,
                      const std::vector<int32_t> & ids, const std::vector<float> & weights,
                      std::vector<float> & output);
    // Device-only prefill. All tensors are contiguous GPU F32 [embd,n]; dense
    // input/shared/residual descriptors and buffers stay alive until return.
    // The destination is separate. Only IDs/route weights/scales cross H2D.
    // Completes the borrowed-backend or private fast stream before returning.
    void forward_device(int layer,int n,ggml_tensor * normalized,
                        const std::vector<int32_t> & ids,const std::vector<float> & weights,
                        ggml_tensor * shared,ggml_tensor * residual,ggml_tensor * destination);
    // Native GPU fallback within an adaptive chunk, without speculative copies
    // for a next layer that may be assigned entirely to the CPU.
    void forward_device_no_prefetch(int layer,int n,ggml_tensor * normalized,
        const std::vector<int32_t> &ids,const std::vector<float> &weights,
        ggml_tensor *shared,ggml_tensor *residual,ggml_tensor *destination);
    // Adaptive mixed-device prefill. The CPU callback receives complete experts'
    // route assignments and returns weighted down rows in route_indices order.
    using CpuRoutes = std::function<void(int,int,const std::vector<float> &,
        const std::vector<int32_t> &,const std::vector<float> &,
        const std::vector<int32_t> &,std::vector<float> &)>;
    void forward_device_hybrid(int layer,int n,ggml_tensor * normalized,
        const std::vector<int32_t> & ids,const std::vector<float> & weights,
        ggml_tensor * shared,ggml_tensor * residual,ggml_tensor * destination,
        const CpuRoutes & cpu_routes);
    // Complete old speculative copies before switching to a CPU/hybrid chunk.
    void drain_prefetch();
    // A cache miss returns false without uploading, evicting, touching LRU or output.
    // A hit executes the complete original [used,n] graph, preserving accumulation.
    // Synchronizes the borrowed backend before exact-dot raw-stream execution.
    bool forward_cached(int layer, int n, const std::vector<float> & x,
                        const std::vector<int32_t> & ids, const std::vector<float> & weights,
                        std::vector<float> & output);
    // Native n1 partial-cache execution. After optional one-time decode seeding,
    // false means zero resident routes: no callback/output mutation or per-route
    // uploads/evictions. Otherwise GPU work is
    // enqueued before the callback computes CPU misses. Callback returns already
    // weighted [embd,missing] contributions, in submitted miss-ID order. The final
    // sum preserves original top-k order; all borrowed work drains on failure.
    using MissingDecode = std::function<void(const std::vector<int32_t> &,
        const std::vector<float> &, std::vector<float> &)>;
    bool forward_cached_hybrid(int layer, const std::vector<float> & x,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        const MissingDecode & cpu_misses, std::vector<float> & output);
    // Opt-in exact cached-only verifier, n1..8. After optional decode seed,
    // all-miss returns false without callback/output mutation or per-route DMA.
    // CPU mask is original [n,used]:1=miss,0=GPU hit. Callback supplies full-rank
    // already weighted [n,used,embd] terms with zero GPU holes; all-hit skips it.
    // Complete Moe output [n,embd] preserves original rank order. Actual private
    // GPU readers and all slot leases drain before return/exception.
    using MaskedDecode = std::function<void(const std::vector<uint8_t>&,
        std::vector<float>&)>;
    bool forward_cached_batched(int layer,int n,const std::vector<float>& x,
        const std::vector<int32_t>& ids,const std::vector<float>& weights,
        const MaskedDecode& cpu_misses,std::vector<float>& output);
    // Readable during a live HTTP worker; no destructor or signal shutdown needed.
    void print_profile(const char * label = "live") const;
    // Compact grouped-worker counters after a completed verifier round; silent
    // while grouping is unused. No public Stats/Config layout changes.
    void print_cached_batch_profile(const char * label = "verify") const;
    bool exact_cpu_math() const;
    bool fast_gpu_math() const;
    bool native_gpu_math() const;
    size_t workspace_bytes() const;
    // Native-only, between completed passes: drains prefetch/backend readers and
    // releases transient prefill scratch. Fixed weight pool and decode graphs stay.
    // A later prefill reallocates scratch normally; call before loading a drafter.
    void release_prefill_workspace();
    // Actual configured per-expert local GEMM bound, or zero outside fast mode.
    int fast_expert_batch_rows() const;
    bool fast_fused_reduction() const;
    bool fast_rank_phases() const;
    // Caller first runs cache_seed_preflight.py against current GGUF headers/stats.
    // Expected identity/content hashes come from its trusted receipt, not the rank itself.
    // Cold-cache initialization only; seed DMA is complete before returning.
    void seed_hot(const std::string & rank_path, const cache_seed::ExpectedRank & expected, size_t hot_count);
    const Stats & stats() const;
    void reset_stats(); // Keeps warm cache entries; counters restart at current residency.
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::m31
