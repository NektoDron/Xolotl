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
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::m31
