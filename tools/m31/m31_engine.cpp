// MiniMax-M3.1 correctness path. Architecture source: llama.cpp PR #31192,
// src/models/minimax-m3.cpp at 6b95a9908d2c7b242817a447708aec38e937e2c0 (MIT).
// Append-only, one sequence, F16 caches. This deliberately starts with stock ggml ops.
#include "m31_engine.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace strata::m31 {
namespace {

void require(bool ok, const std::string & reason) {
    if (!ok) throw std::runtime_error("M31: " + reason);
}
int64_t padded(int64_t n, int64_t align) { return (n + align - 1) / align * align; }

struct Weights {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    std::map<std::string, ggml_tensor *> tensors;
    ~Weights() {
        for (auto b : buffers) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        require(it != tensors.end(), "missing loaded tensor " + name);
        return it->second;
    }
    ggml_tensor * maybe(const std::string & name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
    void load(const GgufModel & model, ggml_backend_t backend, bool dense_gpu) {
        ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "weight metadata allocation failed");
        for (size_t sh = 0; sh < model.size(); ++sh) {
            const auto & file = model.shard(sh);
            ggml_backend_buffer_t buffer = nullptr;
            if (!dense_gpu && !file.tensors().empty()) {
                const auto & first = file.tensors().front();
                uint8_t * base = const_cast<uint8_t *>(file.tensor_data(first)) - file.data_start() - first.offset;
                buffer = ggml_backend_cpu_buffer_from_ptr(base, size_t(file.file_size()));
                require(buffer != nullptr, "cannot wrap GGUF mapping");
                buffers.push_back(buffer);
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            for (const auto & ti : file.tensors()) {
                if (dense_gpu && (expert_tensor(ti.name) || ti.name == "token_embd.weight")) continue;
                require(model.in_bounds(ti, sh), "out-of-bounds weight " + ti.name);
                int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
                require(ti.shape.size() <= GGML_MAX_DIMS, "weight rank too large");
                for (size_t d = 0; d < ti.shape.size(); ++d) ne[d] = int64_t(ti.shape[d]);
                require(ti.type < GGML_TYPE_COUNT, "ggml does not know type " + ti.name);
                auto t = ggml_new_tensor(ctx, ggml_type(ti.type), int(ti.shape.size()), ne);
                ggml_set_name(t, ti.name.c_str());
                tensors.emplace(ti.name, t);
                if (!dense_gpu)
                    require(ggml_backend_tensor_alloc(buffer, t, const_cast<uint8_t *>(file.tensor_data(ti))) == GGML_STATUS_SUCCESS,
                            "cannot bind " + ti.name);
            }
        }
        if (dense_gpu) {
            auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
            require(buffer != nullptr, "cannot allocate GPU dense weights");
            buffers.push_back(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (size_t sh = 0; sh < model.size(); ++sh)
                for (const auto & ti : model.shard(sh).tensors()) {
                    auto it = tensors.find(ti.name);
                    if (it != tensors.end())
                        ggml_backend_tensor_set(it->second, model.shard(sh).tensor_data(ti), 0, ggml_nbytes(it->second));
                }
        }
    }
};

struct Graph {
    ggml_context * c = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;
    const Config & config;
    int64_t pos0 = 0;
    std::vector<std::pair<std::string, ggml_tensor *>> taps;
    Graph(ggml_backend_t b, const Config & cfg, int64_t pos) : backend(b), config(cfg), pos0(pos) {
        c = ggml_init({32 * 1024 * 1024, nullptr, true});
        require(c != nullptr, "graph metadata allocation failed");
        graph = ggml_new_graph_custom(c, 8192, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator != nullptr, "graph allocator failed");
    }
    ~Graph() {
#ifdef M31_ENGINE_CUDA
        // The patched backend keys captured graphs by node addresses. Those addresses can be
        // reused by the next transient context: discard captures BEFORE freeing their buffers.
        ggml_backend_cuda_graph_cache_clear(backend);
#endif
        if (allocator) ggml_gallocr_free(allocator);
        if (c) ggml_free(c);
    }
    ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        auto t = ggml_new_tensor_3d(c, type, n0, n1, n2);
        ggml_set_input(t);
        return t;
    }
    ggml_tensor * tap(ggml_tensor * t, const std::string & base, int l = -1) {
        if (config.tap) {
            const std::string name = l < 0 ? base : base + "-" + std::to_string(l);
            ggml_set_output(t);
            // A view's source can later be overwritten/reused; keep the view as an explicit graph output.
            ggml_build_forward_expand(graph, t);
            taps.emplace_back(name, t);
        }
        return t;
    }
    void allocate(std::initializer_list<ggml_tensor *> outputs) {
        for (auto t : outputs) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
        require(ggml_gallocr_alloc_graph(allocator, graph), "compute allocation failed");
    }
    void compute() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
        for (auto & t : taps) config.tap(t.first, pos0, t.second);
    }
    static std::vector<float> read(ggml_tensor * t) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "expected contiguous F32 readback");
        std::vector<float> result(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
        return result;
    }
};

ggml_tensor * rms(Graph & b, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(b.c, ggml_rms_norm(b.c, x, eps), w);
}
ggml_tensor * ffn(Graph & b, const Weights & w, ggml_tensor * x, const std::string & prefix,
                  const std::string & suffix) {
    auto gate = ggml_mul_mat(b.c, w.get(prefix + "ffn_gate" + suffix + ".weight"), x);
    auto up = ggml_mul_mat(b.c, w.get(prefix + "ffn_up" + suffix + ".weight"), x);
    auto h = ggml_swiglu_oai(b.c, gate, up, 1.702f, 7.0f);
    return ggml_mul_mat(b.c, w.get(prefix + "ffn_down" + suffix + ".weight"), h);
}

// minimax-m3.cpp::build_attn_msa_fa, single sequence, channel=GQA group.
ggml_tensor * grouped_fa(Graph & b, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                         ggml_tensor * mask, const Geometry & g, int n) {
    const int64_t group = g.heads / g.kv_heads;
    auto q4 = ggml_reshape_4d(b.c, q, g.head_dim, group, g.kv_heads, n);
    q4 = ggml_permute(b.c, q4, 0, 2, 3, 1);
    auto out = ggml_flash_attn_ext(b.c, q4, k, v, mask, 1.0f / std::sqrt(float(g.head_dim)), 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    out = ggml_cont(b.c, ggml_permute(b.c, out, 0, 1, 3, 2));
    return ggml_reshape_2d(b.c, out, g.head_dim * g.heads, n);
}

} // namespace

struct Engine::Impl {
    Config cfg;
    Geometry g;
    // Declaration order: model mappings outlive the borrowed tensor buffers in the stores.
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    bool own_backend = false;
    Weights host, device;
    const Weights * dense = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    struct Cache { ggml_tensor * k = nullptr, * v = nullptr, * index = nullptr; };
    std::vector<Cache> cache;
    int64_t capacity = 0, pos = 0;
    int last_n = 0;
    bool poisoned = false, ready = false;
    std::vector<float> hidden;

    ~Impl() {
        close_weights();
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
        if (own_backend && backend) ggml_backend_free(backend);
        if (cpu) ggml_backend_free(cpu);
    }
    void close_weights() {
        for (Weights * w : {&device, &host}) {
            for (auto b : w->buffers) ggml_backend_buffer_free(b);
            w->buffers.clear();
            if (w->ctx) { ggml_free(w->ctx); w->ctx = nullptr; }
            w->tensors.clear();
        }
    }

    ggml_tensor * attention(Graph & b, int l, int64_t start, int n, ggml_tensor * x,
                             ggml_tensor * positions, ggml_tensor * pos_mask, ggml_tensor * bias) {
        auto c = b.c;
        const std::string prefix = "blk." + std::to_string(l) + ".";
        auto wt = [&](const char * name) { return dense->get(prefix + name); };
        auto norm = b.tap(rms(b, x, wt("attn_norm.weight"), g.rms_eps), "attn_norm", l);
        auto q = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_q.weight"), norm), g.head_dim, g.heads, n);
        auto k = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_k.weight"), norm), g.head_dim, g.kv_heads, n);
        auto v = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_v.weight"), norm), g.head_dim, g.kv_heads, n);
        q = b.tap(rms(b, q, wt("attn_q_norm.weight"), g.rms_eps), "Qcur_normed", l);
        k = b.tap(rms(b, k, wt("attn_k_norm.weight"), g.rms_eps), "Kcur_normed", l);
        auto rope = [&](ggml_tensor * t) {
            return ggml_rope_ext(c, t, positions, nullptr, int(g.rope_dim), GGML_ROPE_TYPE_NEOX,
                                 int(g.context), g.rope_base, 1, 0, 1, 32, 1);
        };
        q = b.tap(rope(q), "Qcur", l);
        k = b.tap(rope(k), "Kcur", l);
        b.tap(v, "Vcur", l);
        auto iq = ggml_reshape_3d(c, ggml_mul_mat(c, wt("indexer.q_proj.weight"), norm), g.index_dim, g.index_heads, n);
        auto ik = ggml_reshape_3d(c, ggml_mul_mat(c, wt("indexer.k_proj.weight"), norm), g.index_dim, 1, n);
        iq = rope(rms(b, iq, wt("indexer.q_norm.weight"), g.rms_eps));
        ik = rope(rms(b, ik, wt("indexer.k_norm.weight"), g.rms_eps));

        auto & kv = cache[size_t(l)];
        auto kdst = ggml_view_3d(c, kv.k, g.head_dim, g.kv_heads, n, kv.k->nb[1], kv.k->nb[2], size_t(start) * kv.k->nb[2]);
        auto vdst = ggml_view_3d(c, kv.v, g.head_dim, g.kv_heads, n, kv.v->nb[1], kv.v->nb[2], size_t(start) * kv.v->nb[2]);
        auto idst = ggml_view_2d(c, kv.index, g.index_dim, n, kv.index->nb[1], size_t(start) * kv.index->nb[1]);
        ggml_build_forward_expand(b.graph, ggml_cpy(c, k, kdst));
        ggml_build_forward_expand(b.graph, ggml_cpy(c, v, vdst));
        ggml_build_forward_expand(b.graph, ggml_cpy(c, ggml_reshape_2d(c, ik, g.index_dim, n), idst));

        const int64_t nk = padded(start + n, g.block), blocks = nk / g.block;
        const int64_t top = std::min(g.top_blocks, blocks);
        require(nk * g.index_heads * n < INT32_MAX, "MSA intermediate exceeds int32 kernel indexing; reduce chunk");
        auto ikv = ggml_view_2d(c, kv.index, g.index_dim, nk, kv.index->nb[1], 0);
        auto score = ggml_mul_mat(c, ikv, ggml_reshape_2d(c, iq, g.index_dim, g.index_heads * n));
        ggml_prec_set_acc(score, GGML_PREC_F32);
        score = ggml_reshape_3d(c, score, nk, g.index_heads, n);
        score = ggml_add(c, score, pos_mask);
        auto bs = b.tap(ggml_pool_2d(c, score, GGML_OP_POOL_MAX, int(g.block), 1, int(g.block), 1, 0, 0), "msa_bs", l);
        auto idx = b.tap(ggml_top_k(c, ggml_add(c, bs, bias), int(top)), "msa_topk", l);
        ggml_tensor * result = nullptr;
        if (n == 1) {
            // Match the oracle's score-sorted gather order, including future/padded rows masked at -inf.
            auto a = ggml_reshape_3d(c, ggml_scale(c, ggml_cast(c, idx, GGML_TYPE_F32), float(g.block)), 1, top, g.index_heads);
            auto tj = ggml_add(c, ggml_repeat_4d(c, a, g.block, top, g.index_heads, 1),
                              ggml_reshape_3d(c, ggml_arange(c, 0, float(g.block), 1), g.block, 1, 1));
            auto tr = ggml_add(c, ggml_scale(c, tj, float(g.kv_heads)),
                              ggml_reshape_3d(c, ggml_arange(c, 0, float(g.kv_heads), 1), 1, 1, g.kv_heads));
            auto tok = ggml_cast(c, ggml_reshape_1d(c, tj, g.block * top * g.index_heads), GGML_TYPE_I32);
            auto row = ggml_cast(c, ggml_reshape_1d(c, tr, g.block * top * g.index_heads), GGML_TYPE_I32);
            auto k3 = ggml_view_2d(c, kv.k, g.head_dim, nk * g.kv_heads, kv.k->nb[1], 0);
            auto v3 = ggml_view_2d(c, kv.v, g.head_dim, nk * g.kv_heads, kv.v->nb[1], 0);
            auto kg = ggml_cast(c, ggml_get_rows(c, k3, row), GGML_TYPE_F16);
            auto vg = ggml_cast(c, ggml_get_rows(c, v3, row), GGML_TYPE_F16);
            auto mg = ggml_get_rows(c, ggml_reshape_2d(c, pos_mask, 1, nk), tok);
            auto mask = ggml_cast(c, ggml_reshape_4d(c, mg, g.block * top, 1, 1, g.kv_heads), GGML_TYPE_F16);
            result = grouped_fa(b, q,
                ggml_reshape_4d(c, kg, g.head_dim, g.block * top, 1, g.kv_heads),
                ggml_reshape_4d(c, vg, g.head_dim, g.block * top, 1, g.kv_heads), mask, g, n);
        } else {
            // Scatter the chosen blocks into a mask, then expand each block to its 128 token positions.
            auto ninf = ggml_cast(c, ggml_scale_bias(c, bias, 0, -1e30f), GGML_TYPE_F16);
            ninf = ggml_repeat_4d(c, ninf, blocks, g.index_heads, n, 1);
            auto zero = ggml_scale(c, ggml_cast(c, idx, GGML_TYPE_F32), 0);
            auto bm = ggml_set_rows(c,
                ggml_reshape_3d(c, ninf, 1, blocks, g.index_heads * n),
                ggml_reshape_3d(c, zero, 1, top, g.index_heads * n),
                ggml_reshape_2d(c, idx, top, g.index_heads * n));
            bm = ggml_reshape_3d(c, bm, blocks, g.index_heads, n);
            bm = ggml_cont(c, ggml_permute(c, bm, 0, 2, 1, 3)); // [blocks,n,heads]
            b.tap(bm, "msa_block_mask", l);
            auto rows = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, bm, blocks, n * g.index_heads)));
            auto cell_block = b.input(GGML_TYPE_I32, nk);
            // This extra input is uploaded by the caller after graph allocation.
            ggml_set_name(cell_block, "m31.cell_block");
            cell_blocks_input = cell_block;
            auto selected = ggml_get_rows(c, rows, cell_block);
            selected = ggml_cont(c, ggml_transpose(c, selected));
            selected = ggml_reshape_3d(c, selected, nk, n, g.index_heads);
            auto causal = ggml_reshape_3d(c, pos_mask, nk, n, 1);
            auto mask = ggml_cast(c, ggml_add(c, selected, causal), GGML_TYPE_F16);
            mask = ggml_reshape_4d(c, mask, nk, n, 1, g.index_heads);
            auto k4 = ggml_view_4d(c, kv.k, g.head_dim, g.kv_heads, nk, 1,
                                  kv.k->nb[1], kv.k->nb[2], kv.k->nb[3], 0);
            auto v4 = ggml_view_4d(c, kv.v, g.head_dim, g.kv_heads, nk, 1,
                                  kv.v->nb[1], kv.v->nb[2], kv.v->nb[3], 0);
            result = grouped_fa(b, q, ggml_permute(c, k4, 0, 3, 1, 2),
                                 ggml_permute(c, v4, 0, 3, 1, 2), mask, g, n);
        }
        result = b.tap(result, "kqv_out", l);
        return ggml_mul_mat(c, wt("attn_output.weight"), result);
    }
    ggml_tensor * cell_blocks_input = nullptr; // per transient graph; never retained across a compute

    std::vector<float> routed(int l, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & ids, const std::vector<float> & weights) {
        Graph b(cpu, cfg, pos);
        auto c = b.c;
        auto inp = b.input(GGML_TYPE_F32, g.embd, 1, n);
        auto id = b.input(GGML_TYPE_I32, g.used, n);
        auto rw = b.input(GGML_TYPE_F32, 1, g.used, n);
        const std::string prefix = "blk." + std::to_string(l) + ".ffn_";
        auto mm = [&](const char * role, ggml_tensor * activation) {
            auto w = host.get(prefix + role + "_exps.weight");
            auto out = ggml_mul_mat_id(c, w, activation, id);
            if (w->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(out, GGML_PREC_BF16);
            if (auto scale = host.maybe(prefix + role + "_exps.scale")) {
                auto s = ggml_repeat_4d(c, ggml_reshape_3d(c, scale, 1, g.experts, 1), 1, g.experts, n, 1);
                out = ggml_mul(c, out, ggml_get_rows(c, s, id));
            }
            return out;
        };
        auto gate = b.tap(mm("gate", inp), "ffn_moe_gate", l);
        auto up = b.tap(mm("up", inp), "ffn_moe_up", l);
        auto h = ggml_swiglu_oai(c, gate, up, 1.702f, 7.0f);
        auto down = b.tap(mm("down", h), "ffn_moe_down_scaled", l);
        auto weighted = ggml_mul(c, down, rw);
        // Match llama.cpp's sequential accumulation order, rather than sum_rows/tree reduction.
        ggml_tensor * result = nullptr;
        for (int64_t k = 0; k < g.used; ++k) {
            auto v = ggml_view_2d(c, weighted, g.embd, n, weighted->nb[2], size_t(k) * weighted->nb[1]);
            result = result ? ggml_add(c, result, v) : v;
        }
        result = b.tap(ggml_cont(c, result), "ffn_moe_out", l);
        b.allocate({result});
        ggml_backend_tensor_set(inp, x.data(), 0, ggml_nbytes(inp));
        ggml_backend_tensor_set(id, ids.data(), 0, ggml_nbytes(id));
        ggml_backend_tensor_set(rw, weights.data(), 0, ggml_nbytes(rw));
        b.compute();
        return Graph::read(result);
    }

    void layer(int l, int n) {
        std::vector<float> ffn_input, residual, shared;
        std::vector<int32_t> ids(size_t(g.used * n));
        std::vector<float> route_weights;
        {
            Graph b(backend, cfg, pos);
            auto c = b.c;
            auto x = b.input(GGML_TYPE_F32, g.embd, n);
            auto positions = b.input(GGML_TYPE_I32, n);
            const int64_t nk = padded(pos + n, g.block), nb = nk / g.block;
            auto mask = b.input(GGML_TYPE_F32, nk, 1, n);
            auto bias = b.input(GGML_TYPE_F32, nb, 1, n);
            cell_blocks_input = nullptr;
            auto a = attention(b, l, pos, n, x, positions, mask, bias);
            auto ffn_inp = b.tap(ggml_add(c, a, x), "ffn_inp", l);
            const std::string prefix = "blk." + std::to_string(l) + ".";
            auto norm = b.tap(rms(b, ffn_inp, dense->get(prefix + "ffn_norm.weight"), g.rms_eps), "ffn_norm", l);
            if (!g.routed(l)) {
                auto ff = b.tap(ffn(b, *dense, norm, prefix, ""), "ffn_out", l);
                auto out = b.tap(ggml_add(c, ff, ffn_inp), "l_out", l);
                b.allocate({out});
                upload_attention(b, x, positions, mask, bias, n, nk, nb);
                b.compute();
                hidden = Graph::read(out);
                return;
            }
            auto shexp = b.tap(ffn(b, *dense, norm, prefix, "_shexp"), "ffn_shexp", l);
            auto probs = ggml_sigmoid(c, ggml_mul_mat(c, dense->get(prefix + "ffn_gate_inp.weight"), norm));
            auto selected = b.tap(ggml_top_k(c, ggml_add(c, probs, dense->get(prefix + "exp_probs_b.bias")), int(g.used)), "ffn_moe_topk", l);
            auto weight = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, g.experts, n), selected);
            weight = ggml_reshape_2d(c, weight, g.used, n);
            if (g.norm_routes) {
                auto sum = ggml_clamp(c, ggml_sum_rows(c, weight), 6.103515625e-5f, INFINITY);
                weight = ggml_div(c, weight, sum);
            }
            if (g.route_scale != 1) weight = ggml_scale(c, weight, g.route_scale);
            b.allocate({ffn_inp, norm, shexp, selected, weight});
            upload_attention(b, x, positions, mask, bias, n, nk, nb);
            b.compute();
            residual = Graph::read(ffn_inp);
            ffn_input = Graph::read(norm);
            shared = Graph::read(shexp);
            // top_k can be a non-contiguous view; read each token column explicitly.
            for (int t = 0; t < n; ++t)
                ggml_backend_tensor_get(selected, ids.data() + t * g.used, size_t(t) * selected->nb[1], size_t(g.used) * sizeof(int32_t));
            route_weights = Graph::read(weight);
        }
        const auto moe = routed(l, n, ffn_input, ids, route_weights);
        Graph b(backend, cfg, pos);
        auto r = b.input(GGML_TYPE_F32, g.embd, n);
        auto s = b.input(GGML_TYPE_F32, g.embd, n);
        auto m = b.input(GGML_TYPE_F32, g.embd, n);
        auto ff = b.tap(ggml_add(b.c, m, s), "ffn_out", l);
        auto out = b.tap(ggml_add(b.c, ff, r), "l_out", l);
        b.allocate({out});
        ggml_backend_tensor_set(r, residual.data(), 0, ggml_nbytes(r));
        ggml_backend_tensor_set(s, shared.data(), 0, ggml_nbytes(s));
        ggml_backend_tensor_set(m, moe.data(), 0, ggml_nbytes(m));
        b.compute();
        hidden = Graph::read(out);
    }

    void upload_attention(Graph &, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask,
                          ggml_tensor * bias, int n, int64_t nk, int64_t nb) {
        ggml_backend_tensor_set(x, hidden.data(), 0, ggml_nbytes(x));
        std::vector<int32_t> positions_h(size_t(n), 0);
        std::vector<float> mask_h(size_t(nk * n), -INFINITY), bias_h(size_t(nb * n), 0);
        for (int t = 0; t < n; ++t) {
            positions_h[size_t(t)] = int32_t(pos + t);
            std::fill_n(mask_h.data() + size_t(t) * nk, size_t(pos + t + 1), 0);
            for (int64_t b = (pos + t) / g.block, local = 0; local < g.local_blocks && b >= 0; ++local, --b)
                bias_h[size_t(t) * nb + b] = 1e30f;
        }
        ggml_backend_tensor_set(positions, positions_h.data(), 0, ggml_nbytes(positions));
        ggml_backend_tensor_set(mask, mask_h.data(), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(bias, bias_h.data(), 0, ggml_nbytes(bias));
        if (cell_blocks_input) {
            std::vector<int32_t> blocks(size_t(nk), 0);
            for (int64_t i = 0; i < nk; ++i) blocks[size_t(i)] = int32_t(i / g.block);
            ggml_backend_tensor_set(cell_blocks_input, blocks.data(), 0, ggml_nbytes(cell_blocks_input));
        }
    }
};

Engine::Engine() : p_(new Impl) {}
Engine::~Engine() = default;

bool Engine::init(const std::string & path, const Config & cfg, std::string & error) {
    try {
        require(!p_->model, "init may only be called once");
        require(cfg.threads > 0 && cfg.context > 0 && cfg.max_chunk > 0, "invalid configuration");
        p_->cfg = cfg;
        p_->model = std::make_unique<GgufModel>(gguf_split_paths(path));
        p_->g = read_geometry(*p_->model);
        require(cfg.context <= p_->g.context && cfg.context <= INT32_MAX, "context exceeds model maximum");
        p_->capacity = padded(cfg.context, p_->g.block);
        p_->cpu = ggml_backend_cpu_init();
        require(p_->cpu != nullptr, "cannot create CPU expert backend");
        ggml_backend_cpu_set_n_threads(p_->cpu, cfg.threads);
        p_->backend = cfg.backend;
        if (!p_->backend) {
            p_->backend = ggml_backend_cpu_init();
            p_->own_backend = true;
        }
        require(p_->backend != nullptr, "cannot create dense backend");
        if (ggml_backend_is_cpu(p_->backend)) ggml_backend_cpu_set_n_threads(p_->backend, cfg.threads);
        p_->host.load(*p_->model, p_->cpu, false);
        if (ggml_backend_is_cpu(p_->backend)) p_->dense = &p_->host;
        else { p_->device.load(*p_->model, p_->backend, true); p_->dense = &p_->device; }
        p_->state = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(p_->state != nullptr, "cache metadata allocation failed");
        const auto & g = p_->g;
        for (int64_t l = 0; l < g.layers; ++l) {
            Impl::Cache kv;
            kv.k = ggml_new_tensor_3d(p_->state, GGML_TYPE_F16, g.head_dim, g.kv_heads, p_->capacity);
            kv.v = ggml_new_tensor_3d(p_->state, GGML_TYPE_F16, g.head_dim, g.kv_heads, p_->capacity);
            kv.index = ggml_new_tensor_2d(p_->state, GGML_TYPE_F16, g.index_dim, p_->capacity);
            p_->cache.push_back(kv);
        }
        p_->state_buffer = ggml_backend_alloc_ctx_tensors_from_buft(p_->state, ggml_backend_get_default_buffer_type(p_->backend));
        require(p_->state_buffer != nullptr, "cannot allocate KV/index caches");
        ggml_backend_buffer_clear(p_->state_buffer, 0); // padded rows must contain finite values
        p_->ready = true;
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

const Geometry & Engine::geometry() const { return p_->g; }
int64_t Engine::position() const { return p_->pos; }
void Engine::reset() { p_->pos = 0; p_->last_n = 0; p_->hidden.clear(); p_->poisoned = false; }

bool Engine::forward(const int32_t * tokens, int n, std::string & error) {
    try {
        require(p_->ready && !p_->poisoned, "not initialized or reset required after a failed pass");
        require(tokens && n > 0 && n <= p_->cfg.max_chunk && p_->pos + n <= p_->cfg.context, "invalid pass size/context");
        const auto & g = p_->g;
        for (int i = 0; i < n; ++i) require(tokens[i] >= 0 && tokens[i] < g.vocab, "token id out of range");
        auto emb = p_->host.get("token_embd.weight");
        p_->hidden.resize(size_t(g.embd * n));
        const auto * traits = ggml_get_type_traits(emb->type);
        require(emb->type == GGML_TYPE_F32 || traits->to_float, "no embedding dequantizer");
        for (int i = 0; i < n; ++i) {
            const auto * row = static_cast<const uint8_t *>(emb->data) + size_t(tokens[i]) * emb->nb[1];
            auto * dst = p_->hidden.data() + size_t(i) * g.embd;
            if (emb->type == GGML_TYPE_F32) std::memcpy(dst, row, size_t(g.embd) * sizeof(float));
            else traits->to_float(row, dst, g.embd);
        }
        p_->poisoned = true;
        for (int l = 0; l < g.layers; ++l) p_->layer(l, n);
        p_->pos += n;
        p_->last_n = n;
        p_->poisoned = false;
        return true;
    } catch (const std::exception & e) { p_->poisoned = true; error = e.what(); return false; }
}

bool Engine::logits(int row, int count, std::vector<float> & out, std::string & error) {
    try {
        require(p_->ready && !p_->poisoned && row >= 0 && count > 0 && row + count <= p_->last_n, "invalid logits rows/state");
        Graph b(p_->backend, p_->cfg, p_->pos - p_->last_n + row);
        auto x = b.input(GGML_TYPE_F32, p_->g.embd, count);
        auto norm = b.tap(rms(b, x, p_->dense->get("output_norm.weight"), p_->g.rms_eps), "result_norm");
        auto output = b.tap(ggml_mul_mat(b.c, p_->dense->get("output.weight"), norm), "result_output");
        b.allocate({output});
        ggml_backend_tensor_set(x, p_->hidden.data() + size_t(row) * p_->g.embd, 0, ggml_nbytes(x));
        b.compute();
        out = Graph::read(output);
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

} // namespace strata::m31
