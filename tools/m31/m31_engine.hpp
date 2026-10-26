#pragma once

#include "m31_geometry.hpp"
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
    // Append one chunk. On error, reset() is required before retrying: caches may be partially written.
    bool forward(const int32_t * tokens, int n, std::string & error);
    // Logits for a consecutive set of rows in the last chunk. Small row slices avoid vocab*n scratch.
    bool logits(int row, int count, std::vector<float> & out, std::string & error);
    void reset();
    int64_t position() const;
    // requested is already capped by the caller's remaining prompt. With
    // M31_PREFILL_ADAPTIVE=1, choose a conservative CUDA scratch-safe chunk.
    // Returns0 with error on invalid state/insufficient budget; forward never changes n.
    int prefill_chunk(int requested, std::string & error) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::m31
