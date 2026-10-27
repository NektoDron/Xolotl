#pragma once
// Optional one-token host transfers. The backend owns the pinned allocation;
// all transfers use its existing stream and preserve the original tensor bytes.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::m31 {
class DensePinnedIo {
public:
    DensePinnedIo(ggml_backend_t backend, size_t hidden_bytes, size_t route_bytes)
        : backend_(backend), hidden_bytes_(hidden_bytes), route_bytes_(route_bytes) {
        if (!backend || !hidden_bytes || !route_bytes || hidden_bytes > (1u << 20) || route_bytes > 4096)
            throw std::runtime_error("M31 dense pinned I/O: invalid one-token dimensions");
        auto type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (!type || !ggml_backend_buft_is_host(type))
            throw std::runtime_error("M31 dense pinned I/O: CUDA host buffer unavailable");
        // Separate upload/readback regions avoid overwriting an in-flight H2D.
        bytes_ = 2 * hidden_bytes + 2 * route_bytes;
        buffer_ = ggml_backend_buft_alloc_buffer(type, bytes_);
        if (!buffer_ || ggml_backend_buffer_get_type(buffer_) != type) {
            if (buffer_) ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
            throw std::runtime_error("M31 dense pinned I/O: allocation fell back to pageable memory");
        }
        base_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer_));
    }
    ~DensePinnedIo() {
        if (buffer_) {
            ggml_backend_synchronize(backend_);
            ggml_backend_buffer_free(buffer_);
        }
    }
    DensePinnedIo(const DensePinnedIo &) = delete;
    DensePinnedIo & operator=(const DensePinnedIo &) = delete;
    size_t bytes() const { return bytes_; }
    void upload(ggml_tensor * tensor, const float * input) {
        check(tensor, GGML_TYPE_F32, hidden_bytes_);
        std::memcpy(base_, input, hidden_bytes_);
        ggml_backend_tensor_set_async(backend_, tensor, base_, 0, hidden_bytes_);
    }
    void read_norm(ggml_tensor * tensor) {
        check(tensor, GGML_TYPE_F32, hidden_bytes_);
        ggml_backend_tensor_get_async(backend_, tensor, base_ + hidden_bytes_, 0, hidden_bytes_);
    }
    void read_ids(ggml_tensor * tensor) {
        check(tensor, GGML_TYPE_I32, route_bytes_);
        ggml_backend_tensor_get_async(backend_, tensor, base_ + 2 * hidden_bytes_, 0, route_bytes_);
    }
    void read_weights(ggml_tensor * tensor) {
        check(tensor, GGML_TYPE_F32, route_bytes_);
        ggml_backend_tensor_get_async(backend_, tensor, base_ + 2 * hidden_bytes_ + route_bytes_, 0, route_bytes_);
    }
    // The caller MUST join backend_ before publishing these bytes to CPU work.
    void collect(float * norm, int32_t * ids, float * weights) const {
        std::memcpy(norm, base_ + hidden_bytes_, hidden_bytes_);
        std::memcpy(ids, base_ + 2 * hidden_bytes_, route_bytes_);
        std::memcpy(weights, base_ + 2 * hidden_bytes_ + route_bytes_, route_bytes_);
    }
private:
    ggml_backend_t backend_;
    ggml_backend_buffer_t buffer_ = nullptr;
    uint8_t * base_ = nullptr;
    size_t hidden_bytes_, route_bytes_, bytes_ = 0;
    static void check(ggml_tensor * tensor, ggml_type type, size_t bytes) {
        if (!tensor || tensor->type != type || !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != bytes)
            throw std::runtime_error("M31 dense pinned I/O: unexpected tensor type/shape");
    }
};
}
