// Opt-in M2 expert execution. Original GGUF quants, one cache shared by all layers.
#pragma once
#include "m31_arena.hpp"

#include <cstdint>
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
        uint64_t cached_forwards = 0;
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
    // Runs one layer-batched mul_mat_id graph with the original [used,n] shape and
    // slot-remapped IDs. Contributions are summed on GPU in the original top-k order.
    // Uses OAI SwiGLU(alpha=1.702,limit=7) and each role's external NVFP4 scale.
    std::vector<float> forward(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & ids,
                               const std::vector<float> & weights);
    // Reuse caller-owned host capacity across layers/chunks. Output must not be
    // the same vector object as x or weights; on failure the caller must discard it.
    void forward_into(int layer, int n, const std::vector<float> & x,
                      const std::vector<int32_t> & ids, const std::vector<float> & weights,
                      std::vector<float> & output);
    // A cache miss returns false without uploading, evicting, touching LRU or output.
    // A hit executes the complete original [used,n] graph, preserving accumulation.
    // Synchronizes the borrowed backend before exact-dot raw-stream execution.
    bool forward_cached(int layer, int n, const std::vector<float> & x,
                        const std::vector<int32_t> & ids, const std::vector<float> & weights,
                        std::vector<float> & output);
    // Readable during a live HTTP worker; no destructor or signal shutdown needed.
    void print_profile(const char * label = "live") const;
    bool exact_cpu_math() const;
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
