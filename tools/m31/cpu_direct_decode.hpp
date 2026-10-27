#pragma once
#include "cpu_gpu_host_executor.hpp"
#include "ggml-cpu.h"
#include "cpu_threadpool_config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
// CPU-only n=1 path. Every dot calls the same CPU trait as ggml mul_mat_id.
// Per-expert histogram is one: the >=8-assignment tiled path cannot apply.
class CpuDirectDecode {
    GpuHostExecutor executor;
    std::vector<uint8_t> qgate, qup, qdown;
    std::vector<float> gate, up, hidden, down;
    static const ggml_type_traits_cpu * traits(const ggml_tensor * weight) {
        if (!weight || !weight->data || !supported_weight(weight))
            throw std::invalid_argument("M31 direct CPU: unsupported expert weight");
        auto result = ggml_get_type_traits_cpu(weight->type);
        if (!result || !result->vec_dot || !ggml_get_type_traits_cpu(result->vec_dot_type)->from_float)
            throw std::invalid_argument("M31 direct CPU: missing CPU traits");
        return result;
    }
    static void quantize(const ggml_type_traits_cpu * wt, const float * input,
                         int64_t columns, size_t rows, std::vector<uint8_t> & output) {
        const size_t stride = ggml_row_size(wt->vec_dot_type, columns);
        output.resize(rows * stride);
        auto quant = ggml_get_type_traits_cpu(wt->vec_dot_type)->from_float;
        for (size_t row = 0; row < rows; ++row) quant(input + row * size_t(columns), output.data() + row * stride, columns);
    }
public:
    // IQ2_XS is deliberately an opt-in to the existing stock nrc=1 CPU dots.
    // It has no weight from_float callback: only activation quantization is used.
    // Existing accepted formats return before consulting the new environment flag.
    static bool supported_type(ggml_type type) {
        if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_NVFP4)
            return true;
        if (type != GGML_TYPE_IQ2_XS) return false;
        const char * enabled = std::getenv("M31_CPU_STOCK_EXPERTS");
        if (!enabled || std::strcmp(enabled, "1") != 0) return false;
        const auto * wt = ggml_get_type_traits_cpu(type);
        const auto * activation = wt ? ggml_get_type_traits_cpu(wt->vec_dot_type) : nullptr;
        return wt && wt->vec_dot && activation && activation->from_float;
    }
    static bool supported_weight(const ggml_tensor * weight) {
        if (!weight || !supported_type(weight->type)) return false;
        if (weight->type != GGML_TYPE_IQ2_XS) return true;
        const auto * wt = ggml_get_type_traits_cpu(weight->type);
        return weight->ne[0] > 0 && weight->ne[0] <= std::numeric_limits<int>::max() &&
            weight->ne[0] % ggml_blck_size(weight->type) == 0 &&
            weight->ne[0] % ggml_blck_size(wt->vec_dot_type) == 0 &&
            weight->ne[1] > 0 && weight->ne[1] <= 65536 &&
            weight->ne[2] > 0 && weight->ne[2] <= 128 && weight->ne[3] == 1 &&
            weight->nb[0] == ggml_type_size(weight->type) &&
            weight->nb[1] == ggml_row_size(weight->type, weight->ne[0]) &&
            weight->nb[2] >= weight->nb[1] * size_t(weight->ne[1]);
    }
    static bool supported(const ggml_tensor * wg, const ggml_tensor * wu, const ggml_tensor * wd) {
        for (auto tensor : {wg, wu, wd}) {
            if (!supported_weight(tensor)) return false;
        }
        return true;
    }
    explicit CpuDirectDecode(unsigned threads) : executor([&] {
        GpuHostExecutor::Options options{threads, 32, 4};
        if (std::getenv("M31_CPU_AFFINITY")) {
            auto params = ggml_threadpool_params_default(int(threads));
            apply_cpu_pool_environment(params);
            for (unsigned cpu = 0; cpu < GGML_MAX_N_THREADS && options.affinity.size() < threads; ++cpu)
                if (params.cpumask[cpu]) options.affinity.push_back(cpu);
        }
        return options;
    }()) {}
    // Borrowed original tensors retain their original expert stride (including arena).
    void forward(const ggml_tensor * wg, const ggml_tensor * wu, const ggml_tensor * wd,
                 const float * gate_scale, const float * up_scale, const float * down_scale,
                 const std::vector<float> & input, const std::vector<int32_t> & ids,
                 const std::vector<float> & weights, std::vector<float> & output) {
        const auto * tg = traits(wg); const auto * tu = traits(wu); const auto * td = traits(wd);
        const size_t used = ids.size(), embd = size_t(wg->ne[0]), ff = size_t(wg->ne[1]);
        if (!used || used > 128 || input.size() != embd || weights.size() != used ||
            wu->ne[0] != int64_t(embd) || wu->ne[1] != int64_t(ff) || wd->ne[0] != int64_t(ff) ||
            wd->ne[1] != int64_t(embd) || wg->ne[2] != wu->ne[2] || wg->ne[2] != wd->ne[2])
            throw std::invalid_argument("M31 direct CPU: invalid shape");
        for (size_t k = 0; k < used; ++k) {
            if (ids[k] < 0 || ids[k] >= wg->ne[2]) throw std::invalid_argument("M31 direct CPU: invalid expert ID");
            for (size_t j = 0; j < k; ++j) if (ids[k] == ids[j])
                throw std::invalid_argument("M31 direct CPU: duplicate expert ID");
        }
        quantize(tg, input.data(), int64_t(embd), 1, qgate);
        if (tu->vec_dot_type != tg->vec_dot_type) quantize(tu, input.data(), int64_t(embd), 1, qup);
        const uint8_t * up_input = tu->vec_dot_type == tg->vec_dot_type ? qgate.data() : qup.data();
        gate.resize(used * ff); up.resize(used * ff); hidden.resize(used * ff); down.resize(used * embd);
        executor.parallel_count(used * ff * 2, [&](size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                const bool is_up = index >= used * ff;
                const size_t item = is_up ? index - used * ff : index;
                const size_t rank = item / ff, row = item % ff;
                const auto * w = is_up ? wu : wg;
                const auto * t = is_up ? tu : tg;
                float value;
                const auto * source = static_cast<const uint8_t *>(w->data) + size_t(ids[rank]) * w->nb[2] + row * w->nb[1];
                t->vec_dot(int(embd), &value, 0, source, 0, is_up ? up_input : qgate.data(), 0, 1);
                const float * scales = is_up ? up_scale : gate_scale;
                if (scales) value *= scales[ids[rank]];
                (is_up ? up : gate)[item] = value;
            }
        });
        // Exactly the accepted scalar ggml OAI expression, with no fast-math.
        executor.parallel_count(hidden.size(), [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const float a = std::min(gate[i], 7.f), b = std::clamp(up[i], -7.f, 7.f);
                const float glu = a / (1.f + expf(1.702f * (-a)));
                hidden[i] = glu * (b + 1.f);
            }
        });
        quantize(td, hidden.data(), int64_t(ff), used, qdown);
        const size_t qstride = ggml_row_size(td->vec_dot_type, int64_t(ff));
        executor.parallel_count(used * embd, [&](size_t begin, size_t end) {
            for (size_t item = begin; item < end; ++item) {
                const size_t rank = item / embd, row = item % embd;
                const auto * source = static_cast<const uint8_t *>(wd->data) + size_t(ids[rank]) * wd->nb[2] + row * wd->nb[1];
                float value;
                td->vec_dot(int(ff), &value, 0, source, 0, qdown.data() + rank * qstride, 0, 1);
                if (down_scale) value *= down_scale[ids[rank]];
                down[item] = value * weights[rank];
            }
        });
        output.resize(embd);
        for (size_t row = 0; row < embd; ++row) {
            float sum = down[row];
            for (size_t rank = 1; rank < used; ++rank) sum += down[rank * embd + row];
            output[row] = sum;
        }
    }
};
}
