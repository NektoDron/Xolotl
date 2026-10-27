#pragma once
#include "cpu_direct_decode.hpp"
#include "ggml-alloc.h"
#include <array>
#include <memory>
#include <tuple>

// The exact barrier used by CPU graph kernels; the pool is borrowed from Engine.
// Signature is declared in the pinned vendor's ggml-cpu-impl.h.
extern "C" void ggml_barrier(struct ggml_threadpool *);

namespace strata::m31 {
// A single CPU graph node, using the existing persistent ggml pool. No second pool,
// condition-variable submission per stage, approximation, or changed dot kernel.
class CpuFusedDecode {
    ggml_backend_t backend;
    ggml_threadpool_t pool;
    unsigned threads;
    // Opt-in: pair gate/up rows, then quantize completed independent blocks.
    // Default zero retains the accepted v8 work partition and barriers.
    unsigned paired_quant = 0;
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output_tensor = nullptr;
    std::array<const ggml_tensor *, 3> weight{};
    std::array<const ggml_type_traits_cpu *, 3> trait{};
    std::array<const float *, 3> scales{};
    std::array<std::vector<uint8_t>, 3> quant;
    std::vector<float> gate, up, hidden, down;
    const float * input = nullptr;
    const int32_t * ids = nullptr;
    const float * route_weights = nullptr;
    size_t embd = 0, ff = 0, used = 0;
    static std::pair<size_t, size_t> range(size_t size, int ith, int nth) {
        return {size * size_t(ith) / size_t(nth), size * size_t(ith + 1) / size_t(nth)};
    }
    static void compute_paired(ggml_tensor * output, int ith, int nth, CpuFusedDecode & self) {
        const auto & w = self.weight;
        const auto & t = self.trait;
        if (ith == 0) {
            ggml_get_type_traits_cpu(t[0]->vec_dot_type)->from_float(self.input, self.quant[0].data(), int64_t(self.embd));
            if (t[0]->vec_dot_type != t[1]->vec_dot_type)
                ggml_get_type_traits_cpu(t[1]->vec_dot_type)->from_float(self.input, self.quant[1].data(), int64_t(self.embd));
        }
        ggml_barrier(self.pool);
        const auto * qup = t[0]->vec_dot_type == t[1]->vec_dot_type ? self.quant[0].data() : self.quant[1].data();
        const auto quantize = ggml_get_type_traits_cpu(t[2]->vec_dot_type)->from_float;
        const size_t qstride = ggml_row_size(t[2]->vec_dot_type, int64_t(self.ff));
        const size_t block = size_t(ggml_blck_size(t[2]->vec_dot_type));
        const bool block_quant = self.paired_quant > 1 && block <= 256 && self.ff % block == 0 &&
                                 self.used * (self.ff / block) >= size_t(nth);
        const size_t tile = block_quant ? block : 1;
        auto [begin, end] = range(self.used * (self.ff / tile), ith, nth);
        for (size_t task = begin; task < end; ++task) {
            const size_t rank = task / (self.ff / tile), first = task % (self.ff / tile) * tile;
            float local[256];
            for (size_t offset = 0; offset < tile; ++offset) {
                const size_t row = first + offset;
                float gate_value, up_value;
                const auto * gate_row = static_cast<const uint8_t *>(w[0]->data) + size_t(self.ids[rank]) * w[0]->nb[2] + row * w[0]->nb[1];
                const auto * up_row = static_cast<const uint8_t *>(w[1]->data) + size_t(self.ids[rank]) * w[1]->nb[2] + row * w[1]->nb[1];
                t[0]->vec_dot(int(self.embd), &gate_value, 0, gate_row, 0, self.quant[0].data(), 0, 1);
                t[1]->vec_dot(int(self.embd), &up_value, 0, up_row, 0, qup, 0, 1);
                if (self.scales[0]) gate_value *= self.scales[0][self.ids[rank]];
                if (self.scales[1]) up_value *= self.scales[1][self.ids[rank]];
                const float a = std::min(gate_value, 7.f), b = std::clamp(up_value, -7.f, 7.f);
                const float glu = a / (1.f + expf(1.702f * (-a)));
                const float value = glu * (b + 1.f);
                if (block_quant) local[offset] = value;
                else self.hidden[rank * self.ff + row] = value;
            }
            if (block_quant)
                quantize(local, self.quant[2].data() + rank * qstride + ggml_row_size(t[2]->vec_dot_type, int64_t(first)), int64_t(tile));
        }
        ggml_barrier(self.pool);
        if (!block_quant) {
            for (size_t rank = size_t(ith); rank < self.used; rank += size_t(nth))
                quantize(self.hidden.data() + rank * self.ff, self.quant[2].data() + rank * qstride, int64_t(self.ff));
            ggml_barrier(self.pool);
        }
        std::tie(begin, end) = range(self.used * self.embd, ith, nth);
        for (size_t item = begin; item < end; ++item) {
            const size_t rank = item / self.embd, row = item % self.embd;
            const auto * source = static_cast<const uint8_t *>(w[2]->data) + size_t(self.ids[rank]) * w[2]->nb[2] + row * w[2]->nb[1];
            float value;
            t[2]->vec_dot(int(self.ff), &value, 0, source, 0, self.quant[2].data() + rank * qstride, 0, 1);
            if (self.scales[2]) value *= self.scales[2][self.ids[rank]];
            self.down[item] = value * self.route_weights[rank];
        }
        ggml_barrier(self.pool);
        std::tie(begin, end) = range(self.embd, ith, nth);
        auto * destination = static_cast<float *>(output->data);
        for (size_t row = begin; row < end; ++row) {
            float sum = self.down[row];
            for (size_t rank = 1; rank < self.used; ++rank) sum += self.down[rank * self.embd + row];
            destination[row] = sum;
        }
    }
    static void compute(ggml_tensor * output, int ith, int nth, void * opaque) {
        auto & self = *static_cast<CpuFusedDecode *>(opaque);
        if (self.paired_quant) { compute_paired(output, ith, nth, self); return; }
        const auto & w = self.weight;
        const auto & t = self.trait;
        if (ith == 0) {
            ggml_get_type_traits_cpu(t[0]->vec_dot_type)->from_float(self.input, self.quant[0].data(), int64_t(self.embd));
            if (t[0]->vec_dot_type != t[1]->vec_dot_type)
                ggml_get_type_traits_cpu(t[1]->vec_dot_type)->from_float(self.input, self.quant[1].data(), int64_t(self.embd));
        }
        ggml_barrier(self.pool);
        const auto * qup = t[0]->vec_dot_type == t[1]->vec_dot_type ? self.quant[0].data() : self.quant[1].data();
        auto [begin, end] = range(self.used * self.ff * 2, ith, nth);
        for (size_t index = begin; index < end; ++index) {
            const int role = index >= self.used * self.ff;
            const size_t item = role ? index - self.used * self.ff : index;
            const size_t rank = item / self.ff, row = item % self.ff;
            const auto * source = static_cast<const uint8_t *>(w[role]->data) + size_t(self.ids[rank]) * w[role]->nb[2] + row * w[role]->nb[1];
            float value;
            t[role]->vec_dot(int(self.embd), &value, 0, source, 0, role ? qup : self.quant[0].data(), 0, 1);
            if (self.scales[role]) value *= self.scales[role][self.ids[rank]];
            (role ? self.up : self.gate)[item] = value;
        }
        ggml_barrier(self.pool);
        std::tie(begin, end) = range(self.hidden.size(), ith, nth);
        for (size_t i = begin; i < end; ++i) {
            const float a = std::min(self.gate[i], 7.f), b = std::clamp(self.up[i], -7.f, 7.f);
            const float glu = a / (1.f + expf(1.702f * (-a)));
            self.hidden[i] = glu * (b + 1.f);
        }
        ggml_barrier(self.pool);
        const size_t qstride = ggml_row_size(t[2]->vec_dot_type, int64_t(self.ff));
        for (size_t rank = size_t(ith); rank < self.used; rank += size_t(nth))
            ggml_get_type_traits_cpu(t[2]->vec_dot_type)->from_float(self.hidden.data() + rank * self.ff,
                self.quant[2].data() + rank * qstride, int64_t(self.ff));
        ggml_barrier(self.pool);
        std::tie(begin, end) = range(self.used * self.embd, ith, nth);
        for (size_t item = begin; item < end; ++item) {
            const size_t rank = item / self.embd, row = item % self.embd;
            const auto * source = static_cast<const uint8_t *>(w[2]->data) + size_t(self.ids[rank]) * w[2]->nb[2] + row * w[2]->nb[1];
            float value;
            t[2]->vec_dot(int(self.ff), &value, 0, source, 0, self.quant[2].data() + rank * qstride, 0, 1);
            if (self.scales[2]) value *= self.scales[2][self.ids[rank]];
            self.down[item] = value * self.route_weights[rank];
        }
        ggml_barrier(self.pool);
        std::tie(begin, end) = range(self.embd, ith, nth);
        auto * destination = static_cast<float *>(output->data);
        for (size_t row = begin; row < end; ++row) {
            float sum = self.down[row];
            for (size_t rank = 1; rank < self.used; ++rank) sum += self.down[rank * self.embd + row];
            destination[row] = sum;
        }
    }
public:
    CpuFusedDecode(ggml_backend_t cpu, ggml_threadpool_t threadpool, unsigned n_threads)
        : backend(cpu), pool(threadpool), threads(n_threads) {
        if (!cpu || !ggml_backend_is_cpu(cpu) || !threadpool || !threads || threads > GGML_MAX_N_THREADS)
            throw std::invalid_argument("M31 fused CPU: requires CPU backend and matching persistent pool");
        if (const char * value = std::getenv("M31_CPU_FUSED_PAIRED_QUANT")) {
            if (std::strcmp(value, "0") == 0) paired_quant = 0;
            else if (std::strcmp(value, "1") == 0) paired_quant = 1;
            else if (std::strcmp(value, "2") == 0) paired_quant = 2;
            else throw std::invalid_argument("M31 fused CPU: M31_CPU_FUSED_PAIRED_QUANT must be 0, 1 or 2");
        }
    }
    ~CpuFusedDecode() { if (allocator) ggml_gallocr_free(allocator); if (context) ggml_free(context); }
    CpuFusedDecode(const CpuFusedDecode &) = delete;
    CpuFusedDecode & operator=(const CpuFusedDecode &) = delete;
    void forward(const ggml_tensor * wg, const ggml_tensor * wu, const ggml_tensor * wd,
                 const float * gate_scale, const float * up_scale, const float * down_scale,
                 const std::vector<float> & x, const std::vector<int32_t> & expert_ids,
                 const std::vector<float> & weights, std::vector<float> & output) {
        if (!CpuDirectDecode::supported(wg, wu, wd) || !wg->data || !wu->data || !wd->data || expert_ids.empty() || expert_ids.size() > 128 ||
            x.size() != size_t(wg->ne[0]) || weights.size() != expert_ids.size() ||
            wu->ne[0] != wg->ne[0] || wu->ne[1] != wg->ne[1] || wd->ne[0] != wg->ne[1] || wd->ne[1] != wg->ne[0] ||
            wg->ne[2] != wu->ne[2] || wg->ne[2] != wd->ne[2])
            throw std::invalid_argument("M31 fused CPU: invalid expert shape");
        for (size_t k = 0; k < expert_ids.size(); ++k) {
            if (expert_ids[k] < 0 || expert_ids[k] >= wg->ne[2]) throw std::invalid_argument("M31 fused CPU: invalid expert ID");
            for (size_t j = 0; j < k; ++j) if (expert_ids[k] == expert_ids[j])
                throw std::invalid_argument("M31 fused CPU: duplicate expert ID");
        }
        if (graph && embd != size_t(wg->ne[0])) throw std::invalid_argument("M31 fused CPU: changed graph shape");
        embd = size_t(wg->ne[0]); ff = size_t(wg->ne[1]); used = expert_ids.size();
        weight = {wg, wu, wd}; scales = {gate_scale, up_scale, down_scale};
        for (int role = 0; role < 3; ++role) {
            trait[role] = ggml_get_type_traits_cpu(weight[role]->type);
            quant[role].resize(ggml_row_size(trait[role]->vec_dot_type, int64_t(role == 2 ? ff : embd)) * (role == 2 ? used : 1));
        }
        gate.resize(used * ff); up.resize(used * ff); hidden.resize(used * ff); down.resize(used * embd);
        input = x.data(); ids = expert_ids.data(); route_weights = weights.data();
        if (!graph) {
            context = ggml_init({16 * 1024, nullptr, true});
            if (!context) throw std::runtime_error("M31 fused CPU: graph context allocation failed");
            graph = ggml_new_graph_custom(context, 16, false);
            output_tensor = ggml_custom_4d(context, GGML_TYPE_F32, int64_t(embd), 1, 1, 1, nullptr, 0, compute, int(threads), this);
            ggml_set_output(output_tensor); ggml_build_forward_expand(graph, output_tensor);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("M31 fused CPU: graph allocation failed");
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("M31 fused CPU: graph compute failed");
        output.resize(embd);
        std::memcpy(output.data(), output_tensor->data, embd * sizeof(float));
    }
    // Already weighted terms, ordered by the submitted ID/weight rank. Borrowed until next forward.
    const std::vector<float> & contributions() const { return down; }
};
}
