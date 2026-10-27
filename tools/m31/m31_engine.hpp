#pragma once

#include "m31_geometry.hpp"
#include "draft_feature_view.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::m31 {

struct Config {
    ggml_backend_t backend = nullptr; // borrowed; nullptr creates a CPU backend
    int threads = 32;
    int64_t context = 4096;
    int max_chunk = 128;
    double arena_gib = 0; // opt-in exact per-expert RAM repack; 0 retains GGUF mappings
    double gpu_experts_gib = 0; // opt-in original-type individual expert LRU, requires arena/CUDA
    int gpu_experts_min_tokens = 32; // PR's default op-offload threshold; CPU decode remains exact
    bool reuse_allocations = false; // retain scratch buffers/metadata across sequential graphs
    bool persistent_cpu_pool = false;
    bool cpu_residual = false; // same F32 addition order, avoids an extra GPU round trip
    bool cpu_decode_cache = false; // keep per-layer CPU MoE graphs for single-token steps
    bool dense_decode_cache = false; // stable dense/head graphs, indexed KV writes; requires reuse/cpu_residual
    bool profile = false;
    // Optional DSpark capture of residuals ENTERING these zero-based layer inputs.
    // [4,18,32,46,60] are outputs of blocks [3,17,31,45,59]. Empty keeps v8.
    std::vector<int> draft_feature_layers;
    int draft_feature_max_rows = 4096;
    // Optional calibration observer; IDs/weights are token-major [n,used].
    std::function<void(int layer, int n, const std::vector<int32_t> &,
                       const std::vector<float> &)> route_observer;
    // Called after compute, before the graph is freed. Tensor data may live on the GPU:
    // read with ggml_backend_tensor_get(). Names follow minimax-m3.cpp's cb names.
    std::function<void(const std::string &, int64_t pos0, const ggml_tensor *)> tap;
};

// Initial correctness implementation. Own MSA/router/shared graphs; routed experts use
// ggml-cpu mul_mat_id in their ORIGINAL formats. No rank profile, approximation or requantization.
// The CPU expert path and transient graphs are not the M2 performance implementation.
class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    bool init(const std::string & gguf, const Config & cfg, std::string & error);
    const Geometry & geometry() const;
    // Append one chunk. A preflight rejection preserves valid state. Once execution
    // starts, an error requires reset(): caches may be partially written.
    bool forward(const int32_t * tokens, int n, std::string & error);
    // Append mixed rows with the same linear decoder positions as forward().
    // visual_rows[i] == -1 selects tokens[i]; otherwise tokens[i] must be -1
    // and visual_rows[i] selects an F32 row in [visual_count, geometry().embd].
    // The whole chunk uses the ordinary layered prefill (including GPU v8).
    // Inputs are borrowed only until this synchronous call returns.
    bool forward_mixed(const int32_t * tokens, const int32_t * visual_rows,
                       const float * visual_embeddings, int visual_count,
                       int n, std::string & error);
    // Exact whole-prefix checkpoint: owned host copies of valid F16 K/V/index
    // rows only. Checkpoints belong to this initialized Engine; no model math,
    // weights, live graph pointers or expert-cache entries are serialized.
    class PrefixCheckpoint {
    public:
        ~PrefixCheckpoint() = default;
        PrefixCheckpoint(const PrefixCheckpoint &) = delete;
        PrefixCheckpoint & operator=(const PrefixCheckpoint &) = delete;
        size_t bytes() const { return bytes_.size(); }
        int64_t position() const { return position_; }
    private:
        friend class Engine;
        PrefixCheckpoint() = default;
        std::shared_ptr<const void> owner_;
        int64_t position_ = 0;
        std::vector<uint8_t> bytes_;
    };
    using PrefixCheckpointPtr = std::shared_ptr<const PrefixCheckpoint>;
    size_t prefix_checkpoint_bytes() const;
    bool checkpoint_prefix(size_t maximum_bytes, PrefixCheckpointPtr & out, std::string & error);
    // On success only forward() may consume the restored prefix; logits() has
    // no last chunk. The caller retains the exact last-prefix logits separately.
    // Rejected foreign/invalid checkpoints leave the current valid state alone.
    bool restore_prefix(const PrefixCheckpointPtr & checkpoint, std::string & error);
    // Experimental layer-major verifier: n1 dense geometry, shared CPU expert rows.
    bool verify(const int32_t * tokens, int n, std::vector<float> & logits, std::string & error);
    // Logits for a consecutive set of rows in the last chunk. Small row slices avoid vocab*n scratch.
    bool logits(int row, int count, std::vector<float> & out, std::string & error);
    void reset();
    // Borrow the immutable loaded output projection; valid until Engine destruction.
    ggml_tensor * output_weight() const;
    // Opt-in decode transition: discard retained prefill scratch after the caller
    // saved the last logits/features. KV and expert weights remain valid.
    bool release_prefill_workspace(std::string & error);
    // Caller-owned GPU allocations (e.g. retained DSpark) are subtracted from
    // the post-resident prefill budget. Caller updates this after their changes.
    bool set_external_resident_bytes(uint64_t bytes, std::string & error);
    // Captured rows of the last forward, token-major [rows, feature_layers*embd].
    bool draft_features(int64_t & first_position, int & rows,
                        std::vector<float> & values, std::string & error) const;
    // Request-boundary capture policy, after reset() and before any forward.
    // Empty layers keeps the ordinary target path free of feature readbacks.
    bool set_draft_feature_layers(const std::vector<int> & layers, int max_rows,
                                  std::string & error);
    // Opt-in reuse capture: skip prompt rows before this position. Empty windows
    // return zero valid rows; policy can change only immediately after reset.
    bool set_draft_feature_window(int64_t min_position, std::string & error);
    // Separate opt-in bounded [begin,end) host window. The older min setter
    // retains its per-chunk semantics. Requires VIEW=1 and REUSE=1.
    bool set_draft_feature_view_window(int64_t begin, int64_t end, std::string & error);
    bool draft_feature_window_view(DraftFeatureView & view, std::string & error) const;
    int64_t position() const;
    // Serialized use only, between Engine calls. Cumulative counters for
    // calibration; resets keep these counters. CPU
    // weight demand counts each selected expert's payload once per layer, not
    // measured DRAM traffic (which also depends on CPU tiling and caches).
    struct PrefillStats {
        uint64_t gpu_upload_bytes = 0, gpu_upload_experts = 0, gpu_cache_hits = 0;
        size_t gpu_resident_slots = 0;
        uint64_t cpu_weight_demand_bytes = 0, cpu_assignments = 0, cpu_layers = 0;
        double cpu_lane_seconds = 0, cpu_setup_seconds = 0, cpu_compute_seconds = 0;
        double cpu_reduce_seconds = 0; // included in cpu_compute_seconds
        double cpu_correction_seconds = 0; // included in cpu_compute_seconds
        uint64_t cpu_correction_groups = 0;
        size_t cpu_correction_bytes = 0;
        double gpu_execute_seconds = 0;
        uint64_t short_hybrid_forwards = 0, short_cpu_experts = 0, short_gpu_experts = 0;
        uint64_t short_cached_experts = 0, short_cpu_routes = 0, short_gpu_routes = 0;
        uint64_t short_cold_gpu_bytes = 0;
        double short_cpu_seconds = 0, short_wall_seconds = 0;
        uint64_t cpu_executor_graph_builds = 0, cpu_executor_graph_hits = 0;
        double cpu_executor_graph_build_seconds = 0, cpu_executor_pack_seconds = 0;
        double cpu_executor_read_seconds = 0;
        size_t cpu_executor_retained_graphs = 0, cpu_executor_retained_scratch_bytes = 0;
        uint64_t cpu_cuda_fallback_layers = 0;
        uint64_t cpu_gpu_quant_calls = 0, cpu_gpu_quant_h2d_bytes = 0, cpu_gpu_quant_d2h_bytes = 0;
        double cpu_gpu_quant_seconds = 0; // included in cpu_compute_seconds
        size_t cpu_gpu_quant_device_bytes = 0, cpu_gpu_quant_pinned_bytes = 0;
        uint64_t cpu_gpu_quant_reserve_calls = 0, cpu_gpu_quant_allocations = 0, cpu_gpu_quant_runtime_allocations = 0;
        double cpu_gpu_quant_reserve_seconds = 0; // Engine preparation, outside cpu_compute_seconds
        double cpu_gpu_quant_host_copy_seconds = 0, cpu_gpu_quant_submit_seconds = 0;
        double cpu_gpu_quant_wait_seconds = 0, cpu_gpu_quant_unpack_seconds = 0;
        size_t cpu_gpu_quant_reserved_max_values = 0, cpu_gpu_quant_reserved_max_rows = 0;
        int cpu_gpu_quant_stream_priority = 0;
    };
    PrefillStats prefill_stats() const;
    // requested is already capped by the caller's remaining prompt. With
    // M31_PREFILL_ADAPTIVE=1, choose a conservative CUDA scratch-safe chunk.
    // Returns0 with error on invalid state/insufficient budget; forward never changes n.
    int prefill_chunk(int requested, std::string & error) const;
    // Cache/speculation primitive. Preserves KV/index rows [0,position), but
    // invalidates every graph-backed hidden/head result. Caller owns raw logits.
    // Only positions produced by completed passes are eligible for reuse.
    bool rewind(int64_t position, std::string & error);
    // Pure planning at a logical position, using current retained-owner budget.
    // This does not claim that future allocation high-water marks stay constant.
    int prefill_chunk_at(int64_t position, int requested, std::string & error) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::m31
