#include "shortprefill_cpu_executor.hpp"

#include <iostream>
#include <numeric>
#include <random>
#include <string>

using namespace strata::m31;
namespace {
void check(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
struct Weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, 3> w{};
    std::array<std::vector<float>, 3> scale;
    Weights(ggml_backend_t backend, const std::array<ggml_type, 3> & types, int seed) {
        ctx = ggml_init({64 * 1024, nullptr, true});
        check(ctx, "weight context");
        for (size_t role = 0; role < 3; ++role) {
            w[role] = ggml_new_tensor_3d(ctx, types[role], 256, 256, 8);
            scale[role].resize(8);
            for (size_t expert = 0; expert < 8; ++expert)
                scale[role][expert] = .125f + float(expert + role) * .03125f;
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer, "weight buffer");
        std::mt19937 random{unsigned(seed)};
        std::uniform_real_distribution<float> distribution(-.4f, .4f);
        for (size_t role = 0; role < 3; ++role) {
            std::vector<float> values(size_t(ggml_nelements(w[role]))), imatrix(256, 1.f);
            for (auto & value : values) value = distribution(random);
            std::vector<uint8_t> quant(ggml_nbytes(w[role]));
            const auto count = ggml_quantize_chunk(types[role], values.data(), quant.data(), 0,
                int64_t(values.size() / 256), 256, imatrix.data());
            check(count == quant.size(), "weight quantization size");
            ggml_backend_tensor_set(w[role], quant.data(), 0, quant.size());
        }
    }
    ~Weights() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};
std::vector<float> reference(ggml_backend_t backend, Weights & weights,
    const std::vector<float> & x, const std::vector<int32_t> & ids,
    const std::vector<float> & routes, size_t used, bool scales) {
    auto * ctx = ggml_init({128 * 1024, nullptr, true});
    check(ctx, "reference context");
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    const size_t n = x.size() / 256;
    auto * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, int64_t(n));
    auto * ix = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, int64_t(used), int64_t(n));
    auto * rw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, int64_t(used), int64_t(n));
    for (auto * input : {inp, ix, rw}) ggml_set_input(input);
    std::array<ggml_tensor *, 3> scale{};
    auto mm = [&](size_t role, ggml_tensor * input) {
        auto * result = ggml_mul_mat_id(ctx, weights.w[role], input, ix);
        if (weights.w[role]->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(result, GGML_PREC_BF16);
        if (scales) {
            scale[role] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8, int64_t(n));
            ggml_set_input(scale[role]);
            result = ggml_mul(ctx, result, ggml_get_rows(ctx, scale[role], ix));
        }
        return result;
    };
    auto * gate = mm(0, inp), * up = mm(1, inp);
    auto * hidden = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.f);
    auto * output = ggml_mul(ctx, mm(2, hidden), rw);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(allocator && ggml_gallocr_alloc_graph(allocator, graph), "reference allocation");
    ggml_backend_tensor_set(inp, x.data(), 0, ggml_nbytes(inp));
    ggml_backend_tensor_set(ix, ids.data(), 0, ggml_nbytes(ix));
    ggml_backend_tensor_set(rw, routes.data(), 0, ggml_nbytes(rw));
    if (scales) for (size_t role = 0; role < 3; ++role) {
        std::vector<float> expanded(n * 8);
        for (size_t token = 0; token < n; ++token)
            std::copy(weights.scale[role].begin(), weights.scale[role].end(), expanded.begin() + token * 8);
        ggml_backend_tensor_set(scale[role], expanded.data(), 0, ggml_nbytes(scale[role]));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "reference compute");
    std::vector<float> values(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return values;
}
} // namespace

int main() {
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr;
    try {
        backend = ggml_backend_cpu_init(); check(backend, "CPU backend");
        ggml_backend_cpu_set_n_threads(backend, 4);
        auto params = ggml_threadpool_params_default(4);
        params.poll = 0;
        pool = ggml_threadpool_new(&params); check(pool, "CPU pool");
        ggml_backend_cpu_set_threadpool(backend, pool);
        size_t comparisons = 0;
        {
            ShortPrefillCpuExecutor executor(backend, 4);
            for (const auto & types : {std::array{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32},
                                      std::array{GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS},
                                      std::array{GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4}}) {
                Weights first(backend, types, 135), second(backend, types, 987);
                for (Weights * source : {&first, &second}) for (bool scaled : {false, true})
                    for (size_t n : {size_t(1), size_t(7), size_t(8), size_t(16), size_t(31), size_t(8)}) {
                        std::vector<float> x(256 * n), routes(2 * n);
                        std::vector<int32_t> ids(2 * n);
                        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * .013f);
                        // Expert zero's count crosses 7/8; the remaining experts
                        // have sparse counts and share input tokens with zero.
                        for (size_t token = 0; token < n; ++token) {
                            ids[token * 2] = 0;
                            ids[token * 2 + 1] = int32_t(1 + token % 7);
                            routes[token * 2] = .375f;
                            routes[token * 2 + 1] = .625f;
                        }
                        const auto expected = reference(backend, *source, x, ids, routes, 2, scaled);
                        for (int mask : {255, 1, 254, 10, 0}) {
                            std::vector<size_t> selected;
                            for (size_t route = 0; route < ids.size(); ++route)
                                if (mask & (1 << ids[route])) selected.push_back(route);
                            // Deliberately arbitrary caller order must not change
                            // grouped CPU arithmetic or output order.
                            std::reverse(selected.begin(), selected.end());
                            std::vector<float> actual;
                            executor.forward(source->w[0], source->w[1], source->w[2],
                                scaled ? source->scale[0].data() : nullptr,
                                scaled ? source->scale[1].data() : nullptr,
                                scaled ? source->scale[2].data() : nullptr,
                                x, ids, routes, selected, 2, actual);
                            check(actual.size() == selected.size() * 256, "compact output size");
                            for (size_t row = 0; row < selected.size(); ++row) {
                                check(!std::memcmp(actual.data() + row * 256,
                                    expected.data() + selected[row] * 256, 256 * sizeof(float)),
                                    "compact CPU differs from full CPU: n=" + std::to_string(n) +
                                    " type=" + std::to_string(types[0]) + " mask=" + std::to_string(mask));
                                ++comparisons;
                            }
                        }
                    }
            }
            // Inputs that would silently change the grouped kernel are rejected.
            Weights w(backend, {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32}, 55);
            std::vector<float> x(256 * 8, .1f), routes(16, .5f), output;
            std::vector<int32_t> ids(16);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(i % 2);
            for (const std::vector<size_t> selected : {std::vector<size_t>{0}, {0, 0}, {16}}) {
                bool rejected = false;
                try { executor.forward(w.w[0], w.w[1], w.w[2], nullptr, nullptr, nullptr,
                    x, ids, routes, selected, 2, output); }
                catch (const std::invalid_argument &) { rejected = true; }
                check(rejected, "malformed/partial selection accepted");
            }
            const auto & stats = executor.stats();
            check(stats.retained_graphs <= 4 && stats.graph_builds && stats.graph_hits,
                  "bounded/reused graph cache not exercised");
            std::cout << "PASS compact CPU grouped parity: rows=" << comparisons
                      << " graph_builds=" << stats.graph_builds << " graph_hits=" << stats.graph_hits
                      << " retained=" << stats.retained_graphs << " scratch=" << stats.retained_scratch_bytes << '\n';
        }
        ggml_backend_cpu_set_threadpool(backend, nullptr);
        ggml_threadpool_free(pool); pool = nullptr;
        ggml_backend_free(backend); backend = nullptr;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL " << e.what() << '\n';
        if (backend) ggml_backend_cpu_set_threadpool(backend, nullptr);
        if (pool) ggml_threadpool_free(pool);
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
