#pragma once
// Optional bounded host storage for private-stream quantization and readback.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::m31 {
// Borrowed stream must outlive this object. Both buffers belong exclusively to
// that stream; they must not be submitted to a different stream or backend.
// ensure() performs no CUDA calls when existing capacities already suffice.
class GpuPinnedScratch {
public:
    static constexpr size_t maximum_budget = size_t(1)<<30;
    struct Stats {
        size_t quant_capacity = 0, readback_capacity = 0, budget = 0;
        uint64_t growths = 0, fallbacks = 0;
        bool disabled_after_allocation_failure = false;
    };
    explicit GpuPinnedScratch(cudaStream_t stream,size_t budget = maximum_budget)
        : stream_(stream),budget_(budget) {
        if (!stream || !budget || budget > maximum_budget)
            throw std::invalid_argument("M31 pinned scratch: requires private stream and budget<=1GiB");
    }
    ~GpuPinnedScratch() {
        // Caller must destroy this before destroying the borrowed stream.
        if (quant_ || readback_) {
            cudaStreamSynchronize(stream_);
            clear_unchecked();
        }
    }
    GpuPinnedScratch(const GpuPinnedScratch &) = delete;
    GpuPinnedScratch & operator=(const GpuPinnedScratch &) = delete;

    // False means use the existing pageable quant/readback path for this call.
    // Budget refusal leaves any smaller reusable buffers intact. Allocation
    // failure releases both buffers and disables future attempts on this object.
    // Other CUDA errors propagate; an earlier kernel failure is never hidden.
    bool ensure(size_t quant_bytes,size_t readback_bytes) {
        if (disabled_) { ++fallbacks_; return false; }
        if (quant_bytes <= quant_capacity_ && readback_bytes <= readback_capacity_) return true;
        const size_t q = std::max(quant_capacity_,page_round(quant_bytes));
        const size_t r = std::max(readback_capacity_,page_round(readback_bytes));
        if (q > budget_ || r > budget_-q) { ++fallbacks_; return false; }
        check(cudaStreamSynchronize(stream_)); // drain BEFORE any grow/free
        try {
            if (!grow(quant_,quant_capacity_,q) || !grow(readback_,readback_capacity_,r)) {
                clear_unchecked(); disabled_ = true; ++fallbacks_; return false;
            }
        } catch (...) {
            clear_unchecked(); disabled_ = true;
            throw;
        }
        return true;
    }
    uint8_t * quant() const { return static_cast<uint8_t *>(quant_); }
    float * readback() const { return static_cast<float *>(readback_); }
    Stats stats() const { return {quant_capacity_,readback_capacity_,budget_,growths_,fallbacks_,disabled_}; }

private:
    cudaStream_t stream_;
    size_t budget_,quant_capacity_ = 0,readback_capacity_ = 0;
    void * quant_ = nullptr, * readback_ = nullptr;
    uint64_t growths_ = 0,fallbacks_ = 0;
    bool disabled_ = false;
    static size_t page_round(size_t n) {
        if (n > std::numeric_limits<size_t>::max()-4095)
            return std::numeric_limits<size_t>::max();
        return (n+4095)&~size_t(4095);
    }
    static void check(cudaError_t result) {
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("M31 pinned scratch: ")+cudaGetErrorString(result));
    }
    bool grow(void * & pointer,size_t & capacity,size_t bytes) {
        if (capacity >= bytes) return true;
        // Free before allocation so transient pinned usage also stays bounded.
        if (pointer) {
            check(cudaFreeHost(pointer)); pointer = nullptr; capacity = 0;
        }
        void * next = nullptr;
        const auto result = cudaHostAlloc(&next,bytes,cudaHostAllocDefault);
        if (result == cudaErrorMemoryAllocation || result == cudaErrorNotSupported || result == cudaErrorOperatingSystem) {
            // Clear precisely this expected allocation failure from the calling
            // thread's last-error state; unrelated asynchronous errors still fail.
            const auto sticky = cudaGetLastError();
            if (sticky != cudaSuccess && sticky != result) check(sticky);
            return false;
        }
        check(result);
        pointer = next; capacity = bytes; ++growths_;
        return true;
    }
    void clear_unchecked() noexcept {
        if (quant_) cudaFreeHost(quant_);
        if (readback_) cudaFreeHost(readback_);
        quant_ = readback_ = nullptr;
        quant_capacity_ = readback_capacity_ = 0;
    }
};
}
