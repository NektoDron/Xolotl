// M2 preparation: exact RAM repack, no requantization or routing changes.
#pragma once
#include "m31_geometry.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "prefill_thp_mapping.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace strata::m31 {

class ExpertArena {
public:
    struct Slice { uint8_t * data = nullptr; size_t expert_stride = 0; };
    struct Layer {
        size_t offset = 0, stride = 0, tail_offset = 0;
        std::array<size_t, 3> role_offsets{}, role_bytes{};
        bool routed = false;
    };
    ~ExpertArena() { if (buffer_) ggml_backend_buffer_free(buffer_); }
    ExpertArena() = default;
    ExpertArena(const ExpertArena &) = delete;
    ExpertArena & operator=(const ExpertArena &) = delete;

    void init(const GgufModel & model, const Geometry & g, double budget_gib) {
        if (buffer_ || !std::isfinite(budget_gib) || budget_gib <= 0)
            throw std::runtime_error("M31 arena: invalid/repeated initialization");
        const auto t0 = std::chrono::steady_clock::now();
        struct Source {
            std::string name;
            const uint8_t * data = nullptr, * scales = nullptr;
            bool nvfp4 = false;
        };
        std::vector<std::array<Source, 3>> sources(size_t(g.layers));
        layers_.resize(size_t(g.layers));
        const char * roles[] = {"gate", "up", "down"};
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        auto align = [alignment](size_t n) { return (n + alignment - 1) / alignment * alignment; };
        for (int l = 0; l < g.layers; ++l) {
            if (!g.routed(l)) continue;
            auto & layer = layers_[size_t(l)];
            layer.routed = true;
            layer.offset = bytes_;
            size_t role_end = 0;
            bool tail = false;
            for (int r = 0; r < 3; ++r) {
                auto & src = sources[size_t(l)][size_t(r)];
                src.name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                size_t shard = 0;
                const auto * ti = model.find(src.name, &shard);
                if (!ti || ti->shape.size() != 3 || ti->shape[2] != uint64_t(g.experts) || !model.in_bounds(*ti, shard))
                    throw std::runtime_error("M31 arena: invalid weight " + src.name);
                src.data = model.shard(shard).tensor_data(*ti);
                src.nvfp4 = ti->type == GGML_TYPE_NVFP4;
                layer.role_offsets[size_t(r)] = role_end;
                layer.role_bytes[size_t(r)] = ggml_row_size(ggml_type(ti->type), int64_t(ti->shape[0])) * ti->shape[1];
                role_end += layer.role_bytes[size_t(r)];
                if (src.nvfp4) {
                    const auto scale_name = src.name.substr(0, src.name.size() - 6) + "scale";
                    const auto * scale = model.find(scale_name, &shard);
                    if (!scale || scale->type != GGML_TYPE_F32 || scale->shape != std::vector<uint64_t>{uint64_t(g.experts)} ||
                        !model.in_bounds(*scale, shard))
                        throw std::runtime_error("M31 arena: missing/invalid scale " + scale_name);
                    src.scales = model.shard(shard).tensor_data(*scale);
                    tail = true;
                }
            }
            // The tail belongs to the packed blob, never to the source down-tensor stride.
            layer.tail_offset = tail ? role_end : 0;
            layer.stride = align(role_end + (tail ? 16 : 0));
            bytes_ += layer.stride * size_t(g.experts);
        }
        if (double(bytes_) > budget_gib * 1073741824.0)
            throw std::runtime_error("M31 arena: budget is smaller than the complete expert arena");
        // Leave space for the duty model and graphs; never rely on overcommit/swap for this copy.
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        uint64_t kb = 0;
        while (meminfo >> key >> kb >> unit) {
            if (key == "MemAvailable:") {
                if (kb * 1024 < bytes_ + 16ull * 1024 * 1024 * 1024)
                    throw std::runtime_error("M31 arena: insufficient MemAvailable with 16 GiB reserve");
                break;
            }
        }
        if (PrefillThpMapping::requested()) {
            thp_ = PrefillThpMapping::create(bytes_);
            if (thp_->data()) {
                // This wrapper borrows data: it is freed by ~ExpertArena before
                // the last-declared RAII mapping owner unmaps its reservation.
                buffer_ = ggml_backend_cpu_buffer_from_ptr(thp_->data(), bytes_);
                if (!buffer_) throw std::runtime_error("M31 arena: THP CPU buffer wrapper failed");
            }
        }
        if (!buffer_) {
        buffer_ = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes_);
        }
        if (!buffer_) throw std::runtime_error("M31 arena: CPU allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer_));
        for (int l = 0; l < g.layers; ++l) {
            const auto & layer = layers_[size_t(l)];
            if (!layer.routed) continue;
            for (int r = 0; r < 3; ++r)
                slices_.emplace(sources[size_t(l)][size_t(r)].name,
                    Slice{base + layer.offset + layer.role_offsets[size_t(r)], layer.stride});
            for (int e = 0; e < g.experts; ++e) {
                auto * dst = base + layer.offset + size_t(e) * layer.stride;
                size_t raw_end = 0;
                float tail[4] = {1, 1, 1, 0};
                for (int r = 0; r < 3; ++r) {
                    const auto & src = sources[size_t(l)][size_t(r)];
                    const size_t size = layer.role_bytes[size_t(r)];
                    std::memcpy(dst + layer.role_offsets[size_t(r)], src.data + size_t(e) * size, size);
                    raw_end += size;
                    if (src.scales) {
                        std::memcpy(&tail[r], src.scales + size_t(e) * sizeof(float), sizeof(float));
                        if (!std::isfinite(tail[r]))
                            throw std::runtime_error("M31 arena: invalid NVFP4 scale");
                    }
                }
                std::memset(dst + raw_end, 0, layer.stride - raw_end);
                if (layer.tail_offset) std::memcpy(dst + layer.tail_offset, tail, sizeof tail);
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "M31 RAM arena: bytes=%zu experts=%lld seconds=%.3f (exact packed weights, pageable)\n",
                     bytes_, (long long) (g.experts * (g.layers - g.dense_lead)), seconds);
        if (thp_) {
            const auto & stat = thp_->stats();
            const auto actual = thp_->anon_huge_pages();
            std::fprintf(stderr, "M31 arena THP: mapped=%d advised=%d forced=%d mapped_bytes=%zu AnonHugePages=%llu exact=%d; %s\n",
                         int(stat.mapped), int(stat.advised), int(stat.forced), stat.mapped_bytes,
                         (unsigned long long) actual.bytes, int(actual.exact), stat.note.c_str());
        }
    }
    const Slice * find(const std::string & name) const {
        auto it = slices_.find(name);
        return it == slices_.end() ? nullptr : &it->second;
    }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    size_t bytes() const { return bytes_; }
    const std::vector<Layer> & layers() const { return layers_; }
    const PrefillThpMapping * thp_mapping() const { return thp_.get(); }
private:
    ggml_backend_buffer_t buffer_ = nullptr;
    size_t bytes_ = 0;
    std::vector<Layer> layers_;
    std::map<std::string, Slice> slices_;
    // Keep existing field offsets unchanged; free the borrowed ggml wrapper in
    // the destructor body before this owner releases the underlying mmap.
    std::unique_ptr<PrefillThpMapping> thp_;
};
} // namespace strata::m31
