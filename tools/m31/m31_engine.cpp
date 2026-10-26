// MiniMax-M3.1 correctness path. Architecture source: llama.cpp PR #31192,
// src/models/minimax-m3.cpp at 6b95a9908d2c7b242817a447708aec38e937e2c0 (MIT).
// Append-only, one sequence, F16 caches. This deliberately starts with stock ggml ops.
#include "m31_engine.hpp"
#include "m31_arena.hpp"
#include "m31_gpu_experts.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <utility>

namespace strata::m31 {
namespace {

void require(bool ok, const std::string & reason) {
    if (!ok) throw std::runtime_error("M31: " + reason);
}
int64_t padded(int64_t n, int64_t align) { return (n + align - 1) / align * align; }
// PR's MSA position axis and main KV window both have a minimum 256-cell pad.
// Matching it also keeps FA's kernel/tiling choice identical for numerical parity.
int64_t kv_length(int64_t n, const Geometry & g) { return padded(n, std::max<int64_t>(256, g.block)); }

struct Weights {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    std::map<std::string, ggml_tensor *> tensors;
    std::unique_ptr<ExpertArena> arena;
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
    void load(const GgufModel & model, ggml_backend_t backend, bool dense_gpu,
              const Geometry * geometry = nullptr, double arena_gib = 0) {
        if (!dense_gpu && arena_gib > 0) {
            require(geometry != nullptr, "arena needs validated geometry");
            arena = std::make_unique<ExpertArena>();
            arena->init(model, *geometry, arena_gib);
        }
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
                if (!dense_gpu) {
                    const auto * packed = arena ? arena->find(ti.name) : nullptr;
                    if (packed) {
                        t->nb[2] = packed->expert_stride;
                        t->nb[3] = packed->expert_stride * size_t(t->ne[2]);
                    }
                    require(ggml_backend_tensor_alloc(packed ? arena->buffer() : buffer, t,
                                packed ? packed->data : const_cast<uint8_t *>(file.tensor_data(ti))) == GGML_STATUS_SUCCESS,
                            "cannot bind " + ti.name);
                }
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

struct Workspace {
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit Workspace(ggml_backend_t backend) {
        context = ggml_init({32 * 1024 * 1024, nullptr, true});
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!context || !allocator) {
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
            throw std::runtime_error("M31: workspace allocation failed");
        }
    }
    ~Workspace() {
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
    }
};

// Shared, immutable during a chunk: all layers have the same causal positions.
// Keep these outside transient graph scratch so the large mask is uploaded once.
struct AttentionInputs {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * positions = nullptr, * mask = nullptr, * bias = nullptr, * blocks = nullptr;
    int n = 0;
    int64_t nk = 0;
    ~AttentionInputs() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
    AttentionInputs(ggml_backend_t backend, const Geometry & g, int count, int64_t keys) : n(count), nk(keys) {
        context = ggml_init({64 * 1024, nullptr, true});
        require(context != nullptr, "shared attention input metadata allocation failed");
        positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, n);
        mask = ggml_new_tensor_3d(context, GGML_TYPE_F32, nk, 1, n);
        bias = ggml_new_tensor_3d(context, GGML_TYPE_F32, nk / g.block, 1, n);
        blocks = ggml_new_tensor_1d(context, GGML_TYPE_I32, nk);
        for (auto t : {positions, mask, bias, blocks}) ggml_set_input(t);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, ggml_backend_get_default_buffer_type(backend));
        if (!buffer) {
            ggml_free(context);
            throw std::runtime_error("M31: shared attention input allocation failed");
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
    bool borrowed = false;
    Graph(ggml_backend_t b, const Config & cfg, int64_t pos, Workspace * workspace = nullptr,
          size_t metadata_bytes = 32 * 1024 * 1024, size_t graph_nodes = 8192)
        : backend(b), config(cfg), pos0(pos), borrowed(workspace != nullptr) {
        if (workspace) {
            c = workspace->context;
            ggml_reset(c);
            allocator = workspace->allocator;
        } else {
            c = ggml_init({metadata_bytes, nullptr, true});
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        require(c != nullptr, "graph metadata allocation failed");
        graph = ggml_new_graph_custom(c, graph_nodes, false);
        require(allocator != nullptr, "graph allocator failed");
    }
    ~Graph() {
#ifdef M31_ENGINE_CUDA
        // The patched backend keys captured graphs by node addresses. Those addresses can be
        // reused by the next transient context: discard captures BEFORE freeing their buffers.
        ggml_backend_cuda_graph_cache_clear(backend);
#endif
        if (!borrowed) {
            if (allocator) ggml_gallocr_free(allocator);
            if (c) ggml_free(c);
        }
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
        // A reused allocator can see equal node counts with different aliasing topology.
        // Replan each transient graph while retaining already large scratch buffers.
        if (borrowed) require(ggml_gallocr_reserve(allocator, graph), "compute reservation failed");
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
    static void read_into(ggml_tensor * t, std::vector<float> & result) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "expected contiguous F32 readback");
        result.resize(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
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
    // This ggml CPU FA ignores F32 accumulation for F16 V and rounds its running
    // sum after every key. Preserve the F16 cache values but accumulate them in F32,
    // as requested below (CUDA already respects the accumulator precision).
    if (ggml_backend_is_cpu(b.backend)) v = ggml_cast(b.c, v, GGML_TYPE_F32);
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
    ggml_threadpool_t cpu_pool = nullptr;
    std::unique_ptr<Workspace> dense_workspace, cpu_workspace;
    std::unique_ptr<AttentionInputs> attention_inputs;
    struct RoutedGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * output = nullptr;
    };
    std::vector<std::unique_ptr<RoutedGraph>> cpu_decode_graphs;
    struct DenseGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * positions = nullptr, * mask = nullptr, * bias = nullptr;
        ggml_tensor * residual = nullptr, * norm = nullptr, * shared = nullptr;
        ggml_tensor * ids = nullptr, * weights = nullptr, * output = nullptr;
        int64_t nk = 0;
    };
    struct HeadGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * output = nullptr;
    };
    std::vector<std::unique_ptr<DenseGraph>> dense_decode_graphs;
    HeadGraph head_decode_graph;
    struct Timings {
        double dense = 0, routed_setup = 0, routed_compute = 0, routed_read = 0, residual = 0;
        double dense_build = 0, dense_allocate = 0, dense_upload = 0, dense_compute = 0, dense_read = 0, dense_destroy = 0;
    } timings;
    using Clock = std::chrono::steady_clock;
    static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
    Weights host, device;
    std::unique_ptr<GpuExperts> gpu_experts;
    const Weights * dense = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    struct Cache { ggml_tensor * k = nullptr, * v = nullptr, * index = nullptr; };
    std::vector<Cache> cache;
    int64_t capacity = 0, pos = 0;
    int last_n = 0;
    bool poisoned = false, ready = false;
    std::vector<float> hidden;
    // Opt-in host ownership only: every graph/backend operation stays unchanged.
    // Buffers are private to this serialized Engine and survive layer/reset boundaries.
    bool reuse_host_reads = false;
    struct HostReads {
        std::vector<float> input, residual, shared, weights, moe;
        std::vector<int32_t> ids;
        uint64_t reads = 0, reuses = 0, growths = 0, bytes = 0;
    } host_reads;
    void read_host(ggml_tensor * tensor, std::vector<float> & out) {
        const auto previous_capacity = out.capacity();
        Graph::read_into(tensor, out);
        ++host_reads.reads;
        host_reads.bytes += ggml_nbytes(tensor);
        if (out.capacity() > previous_capacity) ++host_reads.growths;
        else ++host_reads.reuses;
    }
    size_t host_retained_bytes() const {
        return (host_reads.input.capacity() + host_reads.residual.capacity() + host_reads.shared.capacity() +
                host_reads.weights.capacity() + host_reads.moe.capacity()) * sizeof(float) +
               host_reads.ids.capacity() * sizeof(int32_t);
    }

    ~Impl() {
        if (gpu_experts && cfg.profile) {
            const auto & s = gpu_experts->stats();
            std::fprintf(stderr, "M31 expert cache: hits=%llu misses=%llu evictions=%llu upload_bytes=%llu resident=%zu peak=%zu scratch=%zu pinned=%d\n",
                         (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.evictions,
                         (unsigned long long)s.upload_bytes, s.resident_bytes, s.peak_resident_bytes, s.scratch_bytes, s.pinned_staging);
        }
        gpu_experts.reset();
        cpu_decode_graphs.clear();
        dense_decode_graphs.clear();
        head_decode_graph.graph.reset();
        dense_workspace.reset();
        cpu_workspace.reset();
        attention_inputs.reset();
        close_weights();
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
        if (cpu_pool) {
            if (backend && ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_threadpool(backend, nullptr);
            if (cpu) ggml_backend_cpu_set_threadpool(cpu, nullptr);
        }
        if (own_backend && backend) ggml_backend_free(backend);
        if (cpu) ggml_backend_free(cpu);
        if (cpu_pool) ggml_threadpool_free(cpu_pool);
    }
    void close_weights() {
        for (Weights * w : {&device, &host}) {
            for (auto b : w->buffers) ggml_backend_buffer_free(b);
            w->buffers.clear();
            if (w->ctx) { ggml_free(w->ctx); w->ctx = nullptr; }
            w->tensors.clear();
            w->arena.reset();
        }
    }

    ggml_tensor * attention(Graph & b, int l, int64_t start, int n, ggml_tensor * x,
                             ggml_tensor * positions, ggml_tensor * pos_mask, ggml_tensor * bias,
                             bool indexed_kv_write = false) {
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
        if (indexed_kv_write) {
            // Keep pointers and graph topology stable across decode positions. Only
            // the positions tensor contents change. F32 -> F16 conversion is the
            // same storage operation as CPY; cache row order/precision are unchanged.
            for (const auto & item : {std::make_pair(kv.k, k), std::make_pair(kv.v, v)}) {
                auto dst = ggml_reshape_2d(c, item.first, g.head_dim * g.kv_heads, capacity);
                auto src = ggml_reshape_2d(c, item.second, g.head_dim * g.kv_heads, n);
                ggml_build_forward_expand(b.graph, ggml_set_rows(c, dst, src, positions));
            }
            ggml_build_forward_expand(b.graph, ggml_set_rows(c, kv.index,
                ggml_reshape_2d(c, ik, g.index_dim, n), positions));
        } else {
            auto kdst = ggml_view_3d(c, kv.k, g.head_dim, g.kv_heads, n, kv.k->nb[1], kv.k->nb[2], size_t(start) * kv.k->nb[2]);
            auto vdst = ggml_view_3d(c, kv.v, g.head_dim, g.kv_heads, n, kv.v->nb[1], kv.v->nb[2], size_t(start) * kv.v->nb[2]);
            auto idst = ggml_view_2d(c, kv.index, g.index_dim, n, kv.index->nb[1], size_t(start) * kv.index->nb[1]);
            ggml_build_forward_expand(b.graph, ggml_cpy(c, k, kdst));
            ggml_build_forward_expand(b.graph, ggml_cpy(c, v, vdst));
            ggml_build_forward_expand(b.graph, ggml_cpy(c, ggml_reshape_2d(c, ik, g.index_dim, n), idst));
        }

        const int64_t nk = kv_length(start + n, g), blocks = nk / g.block;
        const int64_t top = std::min(g.top_blocks, blocks);
        require(nk * g.index_heads * n < INT32_MAX, "MSA intermediate exceeds int32 kernel indexing; reduce chunk");
        auto ikv = ggml_view_2d(c, kv.index, g.index_dim, nk, kv.index->nb[1], 0);
        // PR's get_rows(pos->cell) converts the F16 cache to F32 before the dot.
        // An identity map still needs that conversion: a direct F16 mul_mat rounds IQ,
        // changing block selection near ties once more than top_blocks exist.
        ikv = ggml_cast(c, ikv, GGML_TYPE_F32);
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
            auto cell_block = attention_inputs ? attention_inputs->blocks : b.input(GGML_TYPE_I32, nk);
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
        if (gpu_experts && n >= cfg.gpu_experts_min_tokens) return gpu_experts->forward(l, n, x, ids, weights);
        std::vector<float> output;
        routed_cpu_into(l, n, x, ids, weights, output);
        return output;
    }

    void routed_cpu_into(int l, int n, const std::vector<float> & x,
                          const std::vector<int32_t> & ids, const std::vector<float> & weights,
                          std::vector<float> & output) {
        const auto setup_start = cfg.profile ? Clock::now() : Clock::time_point{};
        RoutedGraph temporary;
        RoutedGraph * cached = &temporary;
        if (cfg.cpu_decode_cache && n == 1 && !cfg.tap) {
            if (!cpu_decode_graphs[size_t(l)]) cpu_decode_graphs[size_t(l)] = std::make_unique<RoutedGraph>();
            cached = cpu_decode_graphs[size_t(l)].get();
        }
        if (!cached->graph) {
            // One token has fixed tensor shapes and only IDs/activation/route weights
            // vary. Keep the original ggml graph and its alias plan, not a new graph
            // with an assumed equivalent node count. Taps retain the uncached path.
            cached->graph = std::make_unique<Graph>(cpu, cfg, pos,
                cached == &temporary ? cpu_workspace.get() : nullptr,
                cached == &temporary ? 32 * 1024 * 1024 : 1024 * 1024,
                cached == &temporary ? 8192 : 512);
            build_routed(l, n, *cached);
        }
        auto & b = *cached->graph;
        ggml_backend_tensor_set(cached->input, x.data(), 0, ggml_nbytes(cached->input));
        ggml_backend_tensor_set(cached->ids, ids.data(), 0, ggml_nbytes(cached->ids));
        ggml_backend_tensor_set(cached->weights, weights.data(), 0, ggml_nbytes(cached->weights));
        if (cfg.profile) timings.routed_setup += elapsed(setup_start);
        const auto compute_start = cfg.profile ? Clock::now() : Clock::time_point{};
        b.compute();
        if (cfg.profile) timings.routed_compute += elapsed(compute_start);
        const auto read_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (reuse_host_reads) read_host(cached->output, output);
        else output = Graph::read(cached->output);
        if (cfg.profile) timings.routed_read += elapsed(read_start);
    }

    void build_routed(int l, int n, RoutedGraph & storage) {
        auto & b = *storage.graph;
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
        storage.input = inp;
        storage.ids = id;
        storage.weights = rw;
        storage.output = result;
    }

    void clear_dense_decode_cache() {
        // CUDA's captures reference tensor metadata and shared attention inputs.
        // Destroy them before either ownership group can be replaced/freed.
        for (auto & graph : dense_decode_graphs) graph.reset();
        head_decode_graph.graph.reset();
        head_decode_graph.input = head_decode_graph.output = nullptr;
    }

    void build_dense(int l, int n, DenseGraph & storage, bool stable_decode) {
        const auto build_start = cfg.profile ? Clock::now() : Clock::time_point{};
        auto & b = *storage.graph;
        auto c = b.c;
        storage.input = b.input(GGML_TYPE_F32, g.embd, n);
        storage.nk = kv_length(pos + n, g);
        const int64_t nb = storage.nk / g.block;
        storage.positions = attention_inputs ? attention_inputs->positions : b.input(GGML_TYPE_I32, n);
        storage.mask = attention_inputs ? attention_inputs->mask : b.input(GGML_TYPE_F32, storage.nk, 1, n);
        storage.bias = attention_inputs ? attention_inputs->bias : b.input(GGML_TYPE_F32, nb, 1, n);
        cell_blocks_input = nullptr;
        auto a = attention(b, l, pos, n, storage.input, storage.positions, storage.mask, storage.bias, stable_decode);
        auto ffn_inp = b.tap(ggml_add(c, a, storage.input), "ffn_inp", l);
        const std::string prefix = "blk." + std::to_string(l) + ".";
        auto norm = b.tap(rms(b, ffn_inp, dense->get(prefix + "ffn_norm.weight"), g.rms_eps), "ffn_norm", l);
        auto allocate = [&](std::initializer_list<ggml_tensor *> outputs) {
            if (cfg.profile) timings.dense_build += elapsed(build_start);
            const auto alloc_start = cfg.profile ? Clock::now() : Clock::time_point{};
            b.allocate(outputs);
            if (cfg.profile) timings.dense_allocate += elapsed(alloc_start);
        };
        if (!g.routed(l)) {
            auto ff = b.tap(ffn(b, *dense, norm, prefix, ""), "ffn_out", l);
            storage.output = b.tap(ggml_add(c, ff, ffn_inp), "l_out", l);
            allocate({storage.output});
            return;
        }
        storage.residual = ffn_inp;
        storage.norm = norm;
        storage.shared = b.tap(ffn(b, *dense, norm, prefix, "_shexp"), "ffn_shexp", l);
        auto probs = ggml_sigmoid(c, ggml_mul_mat(c, dense->get(prefix + "ffn_gate_inp.weight"), norm));
        storage.ids = b.tap(ggml_top_k(c, ggml_add(c, probs, dense->get(prefix + "exp_probs_b.bias")), int(g.used)), "ffn_moe_topk", l);
        auto weight = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, g.experts, n), storage.ids);
        weight = ggml_reshape_2d(c, weight, g.used, n);
        if (g.norm_routes) {
            auto sum = ggml_clamp(c, ggml_sum_rows(c, weight), 6.103515625e-5f, INFINITY);
            weight = ggml_div(c, weight, sum);
        }
        if (g.route_scale != 1) weight = ggml_scale(c, weight, g.route_scale);
        storage.weights = weight;
        // top_k is a strided view of the full expert argsort. A single device
        // pack replaces n synchronous tiny readbacks per layer during prefill.
        // Keep the default correctness graph untouched; the opt-in allocation
        // path replans its alias topology after this extra integer-only node.
        if (cfg.reuse_allocations && !ggml_is_contiguous(storage.ids))
            storage.ids = ggml_cont(c, storage.ids);
        allocate({storage.residual, storage.norm, storage.shared, storage.ids, storage.weights});
    }

    void layer(int l, int n) {
        const auto dense_start = cfg.profile ? Clock::now() : Clock::time_point{};
        std::vector<float> local_input, local_residual, local_shared, local_weights;
        std::vector<int32_t> local_ids;
        auto & ffn_input = reuse_host_reads ? host_reads.input : local_input;
        auto & residual = reuse_host_reads ? host_reads.residual : local_residual;
        auto & shared = reuse_host_reads ? host_reads.shared : local_shared;
        auto & route_weights = reuse_host_reads ? host_reads.weights : local_weights;
        auto & ids = reuse_host_reads ? host_reads.ids : local_ids;
        const auto ids_capacity = ids.capacity();
        ids.resize(size_t(g.used * n));
        if (reuse_host_reads) {
            if (ids.capacity() > ids_capacity) ++host_reads.growths;
            else ++host_reads.reuses;
        }
        DenseGraph temporary;
        DenseGraph * storage = &temporary;
        const bool stable_decode = cfg.dense_decode_cache && n == 1 && !cfg.tap;
        if (stable_decode) {
            if (!dense_decode_graphs[size_t(l)]) dense_decode_graphs[size_t(l)] = std::make_unique<DenseGraph>();
            storage = dense_decode_graphs[size_t(l)].get();
        }
        if (!storage->graph) {
            const auto create_start = cfg.profile ? Clock::now() : Clock::time_point{};
            storage->graph = std::make_unique<Graph>(backend, cfg, pos,
                stable_decode ? nullptr : dense_workspace.get(),
                stable_decode ? 4 * 1024 * 1024 : 32 * 1024 * 1024,
                stable_decode ? 2048 : 8192);
            if (cfg.profile) timings.dense_build += elapsed(create_start);
            build_dense(l, n, *storage, stable_decode);
        }
        require(storage->nk == kv_length(pos + n, g), "stale dense decode bucket");
        auto & b = *storage->graph;
        const auto upload_start = cfg.profile ? Clock::now() : Clock::time_point{};
        upload_attention(b, storage->input, storage->positions, storage->mask, storage->bias,
                         n, storage->nk, storage->nk / g.block);
        if (cfg.profile) timings.dense_upload += elapsed(upload_start);
        const auto compute_start = cfg.profile ? Clock::now() : Clock::time_point{};
        b.compute();
        if (cfg.profile) timings.dense_compute += elapsed(compute_start);
        const auto read_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (!g.routed(l)) {
            if (reuse_host_reads) read_host(storage->output, hidden);
            else hidden = Graph::read(storage->output);
        }
        else {
            if (reuse_host_reads) {
                read_host(storage->residual, residual);
                read_host(storage->norm, ffn_input);
                read_host(storage->shared, shared);
            } else {
                residual = Graph::read(storage->residual);
                ffn_input = Graph::read(storage->norm);
                shared = Graph::read(storage->shared);
            }
            if (ggml_is_contiguous(storage->ids)) {
                ggml_backend_tensor_get(storage->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            } else {
                for (int t = 0; t < n; ++t)
                    ggml_backend_tensor_get(storage->ids, ids.data() + t * g.used,
                        size_t(t) * storage->ids->nb[1], size_t(g.used) * sizeof(int32_t));
            }
            if (reuse_host_reads) {
                ++host_reads.reads;
                host_reads.bytes += ids.size() * sizeof(int32_t);
                read_host(storage->weights, route_weights);
            } else route_weights = Graph::read(storage->weights);
        }
        if (cfg.profile) timings.dense_read += elapsed(read_start);
        const auto destroy_start = cfg.profile ? Clock::now() : Clock::time_point{};
        temporary.graph.reset(); // cached graphs/captures retain their buffers and tensor addresses
        if (cfg.profile) {
            timings.dense_destroy += elapsed(destroy_start);
            timings.dense += elapsed(dense_start);
        }
        if (!g.routed(l)) return;
        if (cfg.route_observer) cfg.route_observer(l, n, ids, route_weights);
        std::vector<float> local_moe;
        const std::vector<float> * moe_result = nullptr;
        if (reuse_host_reads && !(gpu_experts && n >= cfg.gpu_experts_min_tokens)) {
            routed_cpu_into(l, n, ffn_input, ids, route_weights, host_reads.moe);
            moe_result = &host_reads.moe;
        } else {
            // GPUExperts retains its existing owned-vector API and lifetime.
            local_moe = routed(l, n, ffn_input, ids, route_weights);
            moe_result = &local_moe;
        }
        const auto & moe = *moe_result;
        const auto residual_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (cfg.cpu_residual && !cfg.tap) {
            hidden.resize(moe.size());
            for (size_t i = 0; i < moe.size(); ++i) {
                const float ff = moe[i] + shared[i];
                hidden[i] = ff + residual[i];
            }
            if (cfg.profile) timings.residual += elapsed(residual_start);
            return;
        }
        Graph residual_graph(backend, cfg, pos, dense_workspace.get());
        auto r = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto s = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto m = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto ff = residual_graph.tap(ggml_add(residual_graph.c, m, s), "ffn_out", l);
        auto out = residual_graph.tap(ggml_add(residual_graph.c, ff, r), "l_out", l);
        residual_graph.allocate({out});
        ggml_backend_tensor_set(r, residual.data(), 0, ggml_nbytes(r));
        ggml_backend_tensor_set(s, shared.data(), 0, ggml_nbytes(s));
        ggml_backend_tensor_set(m, moe.data(), 0, ggml_nbytes(m));
        residual_graph.compute();
        if (reuse_host_reads) read_host(out, hidden);
        else hidden = Graph::read(out);
        if (cfg.profile) timings.residual += elapsed(residual_start);
    }

    void upload_attention(Graph &, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask,
                          ggml_tensor * bias, int n, int64_t nk, int64_t nb) {
        ggml_backend_tensor_set(x, hidden.data(), 0, ggml_nbytes(x));
        if (attention_inputs) return; // uploaded once before the first layer
        fill_attention(positions, mask, bias, cell_blocks_input, n, nk, nb);
    }

    void fill_attention(ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * bias,
                        ggml_tensor * cell_blocks, int n, int64_t nk, int64_t nb) {
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
        if (cell_blocks) {
            std::vector<int32_t> blocks(size_t(nk), 0);
            for (int64_t i = 0; i < nk; ++i) blocks[size_t(i)] = int32_t(i / g.block);
            ggml_backend_tensor_set(cell_blocks, blocks.data(), 0, ggml_nbytes(cell_blocks));
        }
    }
};

Engine::Engine() : p_(new Impl) {}
Engine::~Engine() = default;

bool Engine::init(const std::string & path, const Config & cfg, std::string & error) {
    try {
        require(!p_->model, "init may only be called once");
        require(cfg.threads > 0 && cfg.context > 0 && cfg.max_chunk > 0, "invalid configuration");
        require(std::isfinite(cfg.arena_gib) && cfg.arena_gib >= 0, "invalid arena budget");
        require(std::isfinite(cfg.gpu_experts_gib) && cfg.gpu_experts_gib >= 0, "invalid GPU expert budget");
        require(cfg.gpu_experts_min_tokens > 0 && cfg.gpu_experts_min_tokens <= 2048, "invalid GPU expert token threshold");
        require(!cfg.gpu_experts_gib || cfg.arena_gib > 0, "GPU expert cache requires the complete RAM arena");
        require(!cfg.dense_decode_cache || (cfg.reuse_allocations && cfg.cpu_residual),
                "dense decode cache requires reuse-allocations and cpu-residual");
        p_->cfg = cfg;
        const char * host_read_mode = std::getenv("M31_REUSE_HOST_READS");
        const bool host_read_requested = host_read_mode && std::string(host_read_mode) == "1";
        p_->reuse_host_reads = host_read_requested && cfg.reuse_allocations && !cfg.tap;
        if (cfg.profile && host_read_requested)
            std::fprintf(stderr, "M31 host read reuse: enabled=%d (requires reuse-allocations and no taps)\n", int(p_->reuse_host_reads));
        p_->model = std::make_unique<GgufModel>(gguf_split_paths(path));
        p_->g = read_geometry(*p_->model);
        p_->cpu_decode_graphs.resize(size_t(p_->g.layers));
        p_->dense_decode_graphs.resize(size_t(p_->g.layers));
        require(cfg.context <= p_->g.context && cfg.context <= INT32_MAX, "context exceeds model maximum");
        p_->capacity = kv_length(cfg.context, p_->g);
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
        if (cfg.persistent_cpu_pool) {
            auto params = ggml_threadpool_params_default(cfg.threads);
            p_->cpu_pool = ggml_threadpool_new(&params);
            require(p_->cpu_pool != nullptr, "cannot create persistent CPU pool");
            ggml_backend_cpu_set_threadpool(p_->cpu, p_->cpu_pool);
            if (ggml_backend_is_cpu(p_->backend)) ggml_backend_cpu_set_threadpool(p_->backend, p_->cpu_pool);
        }
        if (cfg.reuse_allocations) {
            p_->dense_workspace = std::make_unique<Workspace>(p_->backend);
            p_->cpu_workspace = std::make_unique<Workspace>(p_->cpu);
        }
        p_->host.load(*p_->model, p_->cpu, false, &p_->g, cfg.arena_gib);
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
        if (cfg.gpu_experts_gib > 0) {
            require(p_->host.arena != nullptr, "GPU expert cache requires an arena");
            p_->gpu_experts = std::make_unique<GpuExperts>();
            p_->gpu_experts->init(*p_->model, p_->g, *p_->host.arena, p_->backend, cfg.gpu_experts_gib, cfg.max_chunk);
        }
        p_->ready = true;
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

const Geometry & Engine::geometry() const { return p_->g; }
int64_t Engine::position() const { return p_->pos; }
void Engine::reset() {
    p_->clear_dense_decode_cache();
    p_->pos = 0; p_->last_n = 0; p_->hidden.clear(); p_->poisoned = false;
}

bool Engine::forward(const int32_t * tokens, int n, std::string & error) {
    try {
        const auto prepare_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
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
        if (p_->cfg.reuse_allocations) {
            const auto nk = kv_length(p_->pos + n, g);
            if (!p_->attention_inputs || p_->attention_inputs->n != n || p_->attention_inputs->nk != nk) {
                p_->clear_dense_decode_cache(); // references the old shared positions/mask/bias
                p_->attention_inputs = std::make_unique<AttentionInputs>(p_->backend, g, n, nk);
            }
            auto & inputs = *p_->attention_inputs;
            p_->fill_attention(inputs.positions, inputs.mask, inputs.bias, inputs.blocks, n, nk, nk / g.block);
        }
        const auto started = std::chrono::steady_clock::now();
        const auto previous_host_reads = p_->host_reads.reads;
        const auto previous_host_reuses = p_->host_reads.reuses;
        const auto previous_host_growths = p_->host_reads.growths;
        const auto previous_host_bytes = p_->host_reads.bytes;
        const double prepare_seconds = p_->cfg.profile ? Impl::elapsed(prepare_started) : 0;
        p_->timings = {};
        for (int l = 0; l < g.layers; ++l) p_->layer(l, n);
        if (p_->cfg.profile) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::fprintf(stderr, "M31 forward pos=%lld n=%d seconds=%.6f tps=%.3f\n",
                         (long long)p_->pos, n, seconds, n / seconds);
            const auto & t = p_->timings;
            std::fprintf(stderr, "M31 stages pos=%lld n=%d prepare_ms=%.3f dense_ms=%.3f routed_setup_ms=%.3f routed_compute_ms=%.3f routed_read_ms=%.3f residual_ms=%.3f decode_cache=%d\n",
                         (long long)p_->pos, n, 1000*prepare_seconds, 1000*t.dense, 1000*t.routed_setup, 1000*t.routed_compute,
                         1000*t.routed_read, 1000*t.residual, int(p_->cfg.cpu_decode_cache));
            std::fprintf(stderr, "M31 dense-stages pos=%lld n=%d build_ms=%.3f allocate_ms=%.3f upload_ms=%.3f compute_ms=%.3f read_ms=%.3f destroy_ms=%.3f dense_cache=%d\n",
                         (long long)p_->pos, n, 1000*t.dense_build, 1000*t.dense_allocate, 1000*t.dense_upload,
                         1000*t.dense_compute, 1000*t.dense_read, 1000*t.dense_destroy, int(p_->cfg.dense_decode_cache));
            if (p_->reuse_host_reads) std::fprintf(stderr,
                "M31 host-reads pos=%lld n=%d reads=%llu reuses=%llu growths=%llu bytes=%llu retained_bytes=%zu\n",
                (long long)p_->pos, n,
                (unsigned long long)(p_->host_reads.reads-previous_host_reads),
                (unsigned long long)(p_->host_reads.reuses-previous_host_reuses),
                (unsigned long long)(p_->host_reads.growths-previous_host_growths),
                (unsigned long long)(p_->host_reads.bytes-previous_host_bytes), p_->host_retained_bytes());
        }
        p_->pos += n;
        p_->last_n = n;
        p_->poisoned = false;
        return true;
    } catch (const std::exception & e) { p_->poisoned = true; error = e.what(); return false; }
}

bool Engine::logits(int row, int count, std::vector<float> & out, std::string & error) {
    try {
        const auto head_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        require(p_->ready && !p_->poisoned && row >= 0 && count > 0 && row + count <= p_->last_n, "invalid logits rows/state");
        Impl::HeadGraph temporary;
        const bool stable_decode = p_->cfg.dense_decode_cache && count == 1 && p_->last_n == 1 && !p_->cfg.tap;
        auto & storage = stable_decode ? p_->head_decode_graph : temporary;
        if (!storage.graph) {
            storage.graph = std::make_unique<Graph>(p_->backend, p_->cfg, p_->pos - p_->last_n + row,
                stable_decode ? nullptr : p_->dense_workspace.get(),
                stable_decode ? 1024 * 1024 : 32 * 1024 * 1024,
                stable_decode ? 512 : 8192);
            auto & b = *storage.graph;
            storage.input = b.input(GGML_TYPE_F32, p_->g.embd, count);
            auto norm = b.tap(rms(b, storage.input, p_->dense->get("output_norm.weight"), p_->g.rms_eps), "result_norm");
            storage.output = b.tap(ggml_mul_mat(b.c, p_->dense->get("output.weight"), norm), "result_output");
            b.allocate({storage.output});
        }
        auto & b = *storage.graph;
        ggml_backend_tensor_set(storage.input, p_->hidden.data() + size_t(row) * p_->g.embd, 0, ggml_nbytes(storage.input));
        const double setup_seconds = p_->cfg.profile ? Impl::elapsed(head_started) : 0;
        const auto compute_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        b.compute();
        const double compute_seconds = p_->cfg.profile ? Impl::elapsed(compute_started) : 0;
        const auto read_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        out = Graph::read(storage.output);
        if (p_->cfg.profile) std::fprintf(stderr,
            "M31 head pos=%lld n=%d setup_ms=%.3f compute_ms=%.3f read_ms=%.3f\n",
            (long long)(p_->pos-p_->last_n+row), count, 1000*setup_seconds, 1000*compute_seconds, 1000*Impl::elapsed(read_started));
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

} // namespace strata::m31
