#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::m31 {
// A separate, opt-in drafter. Target weights and the accepted target path are never changed.
// GGUF target_layers are layer INPUT indices, i.e. [4,18,32,46,60] means the
// residual OUTPUTS of zero-based target blocks [3,17,31,45,59].
struct DSparkConfig {
    ggml_backend_t backend = nullptr; // borrowed, must outlive this object
    std::string sidecar;
    std::string target; // mmap of the unchanged target token embeddings
    ggml_tensor * shared_head = nullptr; // borrowed backend output.weight
    int max_append = 1024;
};
class DSpark {
public:
    DSpark();
    ~DSpark();
    DSpark(const DSpark &) = delete;
    DSpark & operator=(const DSpark &) = delete;
    bool init(const DSparkConfig &, std::string & error);
    // Token-major concatenated residuals in GGUF target_layers order. Only append
    // accepted rows. Draft block K/V occupies separate scratch slots, never the ring.
    bool append(const float * features, int64_t first_position, int count, std::string & error);
    // anchor is the already sampled next target token, at position(). Predicts
    // tokens AFTER it. Confidence is a probability; p_min truncates the prefix.
    bool draft(int32_t anchor, int depth, float p_min,
               std::vector<int32_t> & ids, std::vector<float> & confidence, std::string & error);
    void reset();
    int64_t position() const;
    int feature_dim() const;
    int block_size() const;
    uint64_t weight_bytes() const;
    // Owned device weights, ring and all retained graph allocators; excludes borrowed head.
    uint64_t resident_bytes() const;
    double last_ms() const;
    const std::vector<int> & target_layers() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::m31
