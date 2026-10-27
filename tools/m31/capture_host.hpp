#pragma once
// Persistent host storage for exact F32 residual capture. No arithmetic changes.
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>

namespace strata::m31 {
class CaptureHost {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t scratch_ = nullptr;
    std::shared_ptr<float> backing_;
    size_t batch_offset_ = 0, window_rows_ = 0;
    bool accumulating_ = false;
    size_t columns_ = 0, slots_ = 0, capacity_ = 0, rows_ = 0;
    size_t scratch_bytes_ = 0, backing_bytes_ = 0;
    bool pinned_ = false, active_ = false;
    std::vector<uint8_t> written_;
    static void require(bool ok, const char * why) {
        if (!ok) throw std::runtime_error(why);
    }
    static size_t product(size_t a, size_t b) {
        require(!b || a <= std::numeric_limits<size_t>::max()/b,"capture host byte count overflow");
        return a*b;
    }
    void release() noexcept {
        if (scratch_) {
            ggml_backend_synchronize(backend_);
            ggml_backend_buffer_free(scratch_);
        }
        backing_.reset();
        scratch_ = nullptr; backend_ = nullptr;
        columns_ = slots_ = capacity_ = rows_ = scratch_bytes_ = backing_bytes_ = 0;
        pinned_ = active_ = false; written_.clear();
    }
public:
    CaptureHost() = default;
    ~CaptureHost() { release(); }
    CaptureHost(const CaptureHost &) = delete;
    CaptureHost & operator=(const CaptureHost &) = delete;
    // Allocation, registration/page-fault settling happen before the PP timer.
    // Same geometry returns without touching or zeroing the backing again.
    bool prepare(ggml_backend_t backend, size_t columns, size_t slots,
                 size_t capacity_rows, bool require_pinned) {
        require(backend && columns && slots && capacity_rows,"capture host invalid geometry/backend");
        require(!has_live_view(),"release draft feature view before capture policy change");
        if (scratch_ && backend_ == backend && columns_ == columns && slots_ == slots && capacity_ == capacity_rows) {
            require(!require_pinned || pinned_,"capture host scratch is not pinned");
            invalidate(); return false;
        }
        const size_t scratch_bytes = product(product(columns,capacity_rows),sizeof(float));
        const size_t backing_bytes = product(scratch_bytes,slots);
        require(backing_bytes <= std::numeric_limits<size_t>::max()-63,"capture host alignment overflow");
        std::vector<uint8_t> next_written(slots,0);
        auto type = require_pinned
            ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend))
            : ggml_backend_get_default_buffer_type(backend);
        require(type && ggml_backend_buft_is_host(type),"capture host allocation type unavailable");
        auto scratch = ggml_backend_buft_alloc_buffer(type,scratch_bytes);
        require(scratch != nullptr,"capture host scratch allocation failed");
        if (ggml_backend_buffer_get_type(scratch) != type) {
            ggml_backend_buffer_free(scratch);
            throw std::runtime_error("capture host scratch fell back to pageable memory");
        }
        if (!ggml_backend_buffer_get_base(scratch)) {
            ggml_backend_buffer_free(scratch);
            throw std::runtime_error("capture host scratch has no host pointer");
        }
        void * backing = std::aligned_alloc(64,(backing_bytes+63)/64*64);
        if (!backing) {
            ggml_backend_buffer_free(scratch);
            throw std::runtime_error("capture host backing allocation failed");
        }
        // First touch only once on allocation. Every valid row/slot is overwritten
        // later; no clear/resize zero fill occurs during repeated forwards.
        std::memset(ggml_backend_buffer_get_base(scratch),0,scratch_bytes);
        std::memset(backing,0,backing_bytes);
        std::shared_ptr<float> next_backing;
        try { next_backing=std::shared_ptr<float>(static_cast<float *>(backing),[](float * data){std::free(data);}); }
        catch (...) { ggml_backend_buffer_free(scratch);throw; }
        release();
        backend_ = backend; scratch_ = scratch; backing_ = std::move(next_backing);
        columns_ = columns; slots_ = slots; capacity_ = capacity_rows;
        scratch_bytes_ = scratch_bytes; backing_bytes_ = backing_bytes;
        pinned_ = require_pinned; written_ = std::move(next_written); invalidate();
        return true;
    }
    void invalidate_batch() noexcept {
        active_=false;rows_=0;batch_offset_=0;accumulating_=false;
        std::fill(written_.begin(),written_.end(),0);
    }
    void invalidate() noexcept { invalidate_batch();window_rows_=0; }
    bool has_live_view() const noexcept { return backing_ && backing_.use_count()>1; }
    std::shared_ptr<const float> share_window() const {
        require(active_ && complete(rows_),"incomplete accumulated feature window");
        return backing_;
    }
    size_t window_rows() const noexcept { return window_rows_; }
    void begin_window(size_t rows,size_t offset) {
        require(!has_live_view(),"release draft feature view before another forward");
        require(offset==window_rows_ && rows<=capacity_-offset,"non-contiguous/out-of-bounds feature window");
        begin(rows);batch_offset_=offset;accumulating_=true;
    }
    void begin(size_t rows) {
        require(!has_live_view(),"release draft feature view before another forward");
        require(scratch_ && backing_ && rows <= capacity_,"capture host batch exceeds prepared capacity");
        rows_ = rows;batch_offset_=0;accumulating_=false;active_ = true; std::fill(written_.begin(),written_.end(),0);
    }
    float * scratch() const {
        require(active_ && scratch_,"capture host scratch outside a batch");
        return static_cast<float *>(ggml_backend_buffer_get_base(scratch_));
    }
    void scatter(size_t slot, const float * source) {
        require(active_ && source && slot < slots_,"capture host invalid scatter");
        const size_t width = columns_*slots_;
        for (size_t row=0;row<rows_;++row)
            std::memcpy(backing_.get()+(batch_offset_+row)*width+slot*columns_,source+row*columns_,columns_*sizeof(float));
        written_[slot] = 1;
        if (accumulating_ && complete(rows_)) window_rows_=batch_offset_+rows_;
    }
    bool complete(size_t rows) const {
        return active_ && rows == rows_ && (!rows || std::all_of(written_.begin(),written_.end(),[](uint8_t value){return value == 1;}));
    }
    const float * values(size_t rows) const {
        require(complete(rows),"capture host contains incomplete/stale feature slots");
        return backing_.get()+batch_offset_*columns_*slots_;
    }
    size_t scratch_bytes() const { return scratch_bytes_; }
    size_t backing_bytes() const { return backing_bytes_; }
    size_t capacity_rows() const { return capacity_; }
    bool pinned() const { return pinned_; }
};
} // namespace strata::m31
