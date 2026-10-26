// Bounded original-byte upload staging. The borrowed compute backend and every
// destination tensor must outlive the uploader (including its destructor).
#pragma once
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31 {

class PrefillUploads {
public:
    struct Part { size_t offset; const void * data; size_t bytes; };
    struct Stats {
        uint64_t batches = 0, uploads = 0, packed_bytes = 0, upload_bytes = 0;
        uint64_t staging_reuses = 0;
        size_t staging_bytes = 0, depth = 0;
        bool pinned = false, asynchronous = false;
    };
    // slot_bytes includes quant and alignment padding; each upload zeros all
    // padding before copying Parts. No host/device allocation occurs in enqueue.
    PrefillUploads(ggml_backend_t compute, size_t slot_bytes, size_t depth = 2,
                   bool allow_async = true, size_t staging_limit = 256 * 1024 * 1024)
        : compute_(compute), bytes_(slot_bytes) {
        require(compute && slot_bytes && depth && depth <= 16 &&
                slot_bytes <= staging_limit / depth, "invalid bounded staging budget");
        stat_.depth = depth;
        stat_.staging_bytes = slot_bytes * depth;
        stages_.resize(depth);
        const auto device = ggml_backend_get_device(compute);
        require(device != nullptr, "compute backend has no device");
        const auto type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            const char * session = std::getenv("M31_CARD_SESSION");
            require(session && std::strcmp(session, "1") == 0, "GPU requires the aictl/flock runner");
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const auto host = ggml_backend_dev_host_buffer_type(device);
        try {
            if (allow_async && type == GGML_BACKEND_DEVICE_TYPE_GPU &&
                props.caps.async && props.caps.events && host) {
                transfer_ = ggml_backend_dev_init(device, nullptr);
                require(transfer_ != nullptr, "transfer backend initialization failed");
                ready_ = ggml_backend_event_new(device);
                readers_ = ggml_backend_event_new(device);
                require(ready_ && readers_, "batch event allocation failed");
                for (auto & stage : stages_) {
                    stage.buffer = ggml_backend_buft_alloc_buffer(host, bytes_);
                    require(stage.buffer != nullptr, "pinned staging allocation failed");
                    stage.event = ggml_backend_event_new(device);
                    require(stage.event != nullptr, "staging event allocation failed");
                }
                stat_.pinned = stat_.asynchronous = true;
            } else {
                for (auto & stage : stages_) stage.fallback.resize(bytes_);
            }
        } catch (...) {
            release();
            throw;
        }
    }
    ~PrefillUploads() { release(); }
    PrefillUploads(const PrefillUploads &) = delete;
    PrefillUploads & operator=(const PrefillUploads &) = delete;

    // Call only after all previous-batch consumers have been queued on compute.
    // This records their completion and makes subsequent H2D writes wait for it;
    // destination slots cannot be overwritten under a previous consumer.
    void begin_batch() {
        require(!active_, "begin_batch while a batch is open");
        if (stat_.asynchronous) {
            ggml_backend_event_record(readers_, compute_);
            ggml_backend_event_wait(transfer_, readers_);
        } else {
            ggml_backend_synchronize(compute_);
        }
        active_ = true;
    }

    // The source Parts may be reused as soon as this returns. The staging ring
    // owns copies until H2D events complete. A destination is a complete slot.
    void enqueue(ggml_tensor * destination, std::initializer_list<Part> parts) {
        require(active_, "enqueue outside a batch");
        require(destination && ggml_is_contiguous(destination) && ggml_nbytes(destination) == bytes_,
                "destination must be one contiguous complete slot");
        for (auto a = parts.begin(); a != parts.end(); ++a) {
            require(a->offset <= bytes_ && a->bytes <= bytes_ - a->offset && (a->data || !a->bytes),
                    "invalid source part");
            for (auto b = parts.begin(); b != a; ++b)
                require(!a->bytes || !b->bytes || a->offset >= b->offset + b->bytes ||
                        b->offset >= a->offset + a->bytes, "overlapping source parts");
        }
        auto & stage = stages_[cursor_];
        if (stage.busy) {
            // CPU packing must not race a DMA still reading this pinned buffer.
            ggml_backend_event_synchronize(stage.event);
            ++stat_.staging_reuses;
        }
        auto * data = stage.buffer ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage.buffer))
                                  : stage.fallback.data();
        std::memset(data, 0, bytes_);
        for (const auto & part : parts) if (part.bytes) {
            std::memcpy(data + part.offset, part.data, part.bytes);
            stat_.packed_bytes += part.bytes;
        }
        if (stat_.asynchronous) {
            ggml_backend_tensor_set_async(transfer_, destination, data, 0, bytes_);
            ggml_backend_event_record(stage.event, transfer_);
            stage.busy = true;
        } else {
            ggml_backend_tensor_set(destination, data, 0, bytes_);
        }
        ++stat_.uploads;
        stat_.upload_bytes += bytes_;
        cursor_ = (cursor_ + 1) % stages_.size();
    }

    // Orders future compute after all uploads without a host synchronization.
    // It does not overlap this batch's compute with this batch's uploads: the
    // caller must supply disjoint batches/slots to implement that scheduling.
    void finish() {
        require(active_, "finish outside a batch");
        if (stat_.asynchronous) {
            ggml_backend_event_record(ready_, transfer_);
            ggml_backend_event_wait(compute_, ready_);
        }
        active_ = false;
        ++stat_.batches;
    }
    void synchronize() {
        require(!active_, "synchronize before finish");
        if (transfer_) ggml_backend_synchronize(transfer_);
        ggml_backend_synchronize(compute_);
    }
    const Stats & stats() const { return stat_; }

private:
    struct Stage {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        std::vector<uint8_t> fallback;
        bool busy = false;
    };
    static void require(bool condition, const char * message) {
        if (!condition) throw std::runtime_error(std::string("M31 prefill uploads: ") + message);
    }
    void release() noexcept {
        if (transfer_) ggml_backend_synchronize(transfer_);
        if (compute_) ggml_backend_synchronize(compute_);
        for (auto & stage : stages_) {
            if (stage.event) ggml_backend_event_free(stage.event);
            if (stage.buffer) ggml_backend_buffer_free(stage.buffer);
            stage.event = nullptr;
            stage.buffer = nullptr;
        }
        if (ready_) ggml_backend_event_free(ready_);
        if (readers_) ggml_backend_event_free(readers_);
        if (transfer_) ggml_backend_free(transfer_);
        ready_ = readers_ = nullptr;
        transfer_ = nullptr;
    }
    ggml_backend_t compute_ = nullptr, transfer_ = nullptr;
    ggml_backend_event_t ready_ = nullptr, readers_ = nullptr;
    size_t bytes_ = 0, cursor_ = 0;
    std::vector<Stage> stages_;
    Stats stat_;
    bool active_ = false;
};
} // namespace strata::m31
