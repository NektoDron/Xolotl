#include "m31_gpu_experts.hpp"

#include "ggml-alloc.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::m31 {
namespace {
void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("M31 GPU experts: " + message);
}
void clear_captures(ggml_backend_t backend) {
#ifdef M31_ENGINE_CUDA
    ggml_backend_cuda_graph_cache_clear(backend);
#else
    (void) backend;
#endif
}
size_t aligned(size_t value, size_t alignment) {
    check(alignment && value <= std::numeric_limits<size_t>::max() - (alignment - 1), "alignment overflow");
    return (value + alignment - 1) / alignment * alignment;
}
size_t common_multiple(size_t a, size_t b) {
    check(a && b, "invalid stride multiple");
    const size_t divided = a / std::gcd(a, b);
    check(divided <= std::numeric_limits<size_t>::max() / b, "stride multiple overflow");
    return divided * b;
}
struct Context {
    ggml_context * value = nullptr;
    explicit Context(size_t bytes) : value(ggml_init({bytes, nullptr, true})) {
        check(value != nullptr, "metadata allocation failed");
    }
    ~Context() { if (value) ggml_free(value); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
};
} // namespace

struct GpuExperts::Impl {
    struct Layer {
        std::array<ggml_type, 3> types{};
        std::array<const ExpertArena::Slice *, 3> sources{};
        std::array<std::vector<float>, 3> scales;
        std::array<bool, 3> has_scale{};
        std::array<size_t, 3> offsets{}, raw_bytes{}, allocation_bytes{};
        size_t span = 0;
    };
    struct Entry {
        size_t slot = 0;
        uint64_t last_used = 0;
    };
    Geometry g;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t pool = nullptr, staging = nullptr;
    std::unique_ptr<Context> slot_metadata;
    ggml_tensor * pool_proxy = nullptr;
    std::vector<ggml_tensor *> slot_proxies;
    std::vector<size_t> free_slots;
    std::vector<uint8_t> pageable_staging;
    ggml_gallocr_t allocator = nullptr;
    std::vector<Layer> layers;
    std::map<uint64_t, Entry> entries;
    Stats stat;
    uint64_t clock = 0;
    int max_chunk = 0;
    bool ready = false;

    ~Impl() {
        if (backend) {
            ggml_backend_synchronize(backend);
            clear_captures(backend);
        }
        if (allocator) ggml_gallocr_free(allocator);
        if (pool) ggml_backend_buffer_free(pool);
        if (staging) ggml_backend_buffer_free(staging);
    }
    uint64_t key(int layer, int expert) const {
        return uint64_t(layer) * uint64_t(g.experts) + uint64_t(expert);
    }
    size_t load(int layer, int expert, const std::set<uint64_t> & protected_keys) {
        const uint64_t id = key(layer, expert);
        auto found = entries.find(id);
        if (found != entries.end()) {
            found->second.last_used = ++clock;
            ++stat.hits;
            return found->second.slot;
        }
        size_t slot = 0;
        if (!free_slots.empty()) {
            slot = free_slots.back();
            free_slots.pop_back();
        } else {
            auto victim = entries.end();
            for (auto it = entries.begin(); it != entries.end(); ++it)
                if (!protected_keys.count(it->first) &&
                    (victim == entries.end() || it->second.last_used < victim->second.last_used)) victim = it;
            check(victim != entries.end(), "all cache slots are protected; increase expert cache budget");
            slot = victim->second.slot;
            entries.erase(victim);
            ++stat.evictions;
        }
        const auto & source = layers[size_t(layer)];
        auto * stage = staging ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(staging)) : pageable_staging.data();
        // Includes each role's CUDA quant padding and slot-alignment padding. External
        // NVFP4 scales live in the graph, not in any role's byte stride.
        std::memset(stage, 0, stat.slot_stride);
        for (int r = 0; r < 3; ++r) {
            const auto * slice = source.sources[size_t(r)];
            std::memcpy(stage + source.offsets[size_t(r)],
                        slice->data + size_t(expert) * slice->expert_stride, source.raw_bytes[size_t(r)]);
        }
        // A single synchronous H2D copy, with no cudaMalloc/cudaFree on a cache miss.
        ggml_backend_tensor_set(slot_proxies[slot], stage, 0, stat.slot_stride);
        stat.upload_bytes += stat.slot_stride;
        ++stat.misses;
        entries.emplace(id, Entry{slot, ++clock});
        stat.occupied_slots = entries.size();
        return slot;
    }
    ggml_tensor * weight_view(ggml_context * c, int layer, int role) const {
        const auto & info = layers[size_t(layer)];
        const int64_t columns = role == 2 ? g.expert_ff : g.embd;
        const int64_t rows = role == 2 ? g.embd : g.expert_ff;
        auto tensor = ggml_new_tensor_3d(c, info.types[size_t(role)], columns, rows, int64_t(stat.slot_count));
        tensor->nb[2] = stat.slot_stride;
        tensor->nb[3] = stat.slot_stride * stat.slot_count;
        check(stat.slot_stride % ggml_type_size(tensor->type) == 0, "CUDA expert stride is not an integral number of quant blocks");
        const size_t offset = info.offsets[size_t(role)];
        check(offset + ggml_nbytes(tensor) <= stat.pool_bytes, "weight view exceeds slot pool");
        check(offset + ggml_backend_buft_get_alloc_size(buffer_type, tensor) <= stat.pool_bytes,
              "weight view padding exceeds slot pool");
        // Explicit borrowed reinterpret view over an I8 byte-storage tensor. Calling
        // tensor_alloc() on a naked quant tensor would zero its trailing quant padding:
        // the last slot may contain another layer with different role offsets/types.
        // CUDA's view initializer deliberately skips that write. All required padding
        // was already zeroed inside the selected expert's own slot before its upload.
        tensor->view_src = pool_proxy;
        tensor->view_offs = offset;
        check(ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS, "cannot bind borrowed weight view");
        return tensor;
    }
    std::vector<float> execute(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & mapped_ids,
                               const std::vector<float> & route_weights,
                               const std::array<std::vector<float>, 3> & scales) {
        Context context(8 * 1024 * 1024);
        auto c = context.value;
        auto graph = ggml_new_graph_custom(c, 8192, false);
        struct CaptureGuard {
            ggml_backend_t backend;
            ~CaptureGuard() { ggml_backend_synchronize(backend); clear_captures(backend); }
        } guard{backend};
        auto input = ggml_new_tensor_3d(c, GGML_TYPE_F32, g.embd, 1, n);
        auto ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, g.used, n);
        auto rw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, g.used, n);
        for (auto tensor : {input, ids, rw}) ggml_set_input(tensor);
        std::array<ggml_tensor *, 3> scale_inputs{};
        const auto & info = layers[size_t(layer)];
        auto mm = [&](int role, ggml_tensor * activation) {
            auto weight = weight_view(c, layer, role);
            auto out = ggml_mul_mat_id(c, weight, activation, ids);
            if (weight->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(out, GGML_PREC_BF16);
            if (stat.q8_activations) ggml_prec_set_src(out, GGML_PREC_Q8, 1);
            if (info.has_scale[size_t(role)]) {
                auto scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, int64_t(stat.slot_count));
                ggml_set_input(scale);
                scale_inputs[size_t(role)] = scale;
                auto s = ggml_repeat_4d(c, ggml_reshape_3d(c, scale, 1, int64_t(stat.slot_count), 1),
                                      1, int64_t(stat.slot_count), n, 1);
                out = ggml_mul(c, out, ggml_get_rows(c, s, ids));
            }
            return out;
        };
        // Preserve the original [used,n] shape: GPU dispatch/activation quantization
        // must not change merely because an expert received fewer than nine tokens.
        auto gate = mm(0, input);
        auto up = mm(1, input);
        auto activation = ggml_swiglu_oai(c, gate, up, 1.702f, 7.0f);
        auto down = mm(2, activation);
        auto weighted = ggml_mul(c, down, rw);
        ggml_tensor * result = nullptr;
        for (int64_t rank = 0; rank < g.used; ++rank) {
            auto term = ggml_view_2d(c, weighted, g.embd, n, weighted->nb[2], size_t(rank) * weighted->nb[1]);
            result = result ? ggml_add(c, result, term) : term;
        }
        result = ggml_cont(c, result);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            check(ggml_backend_dev_supports_op(device, node), std::string("backend does not support ") + ggml_op_name(node->op));
        }
        // Graph shapes/scale topology change across layers. Rebuild the alias plan,
        // retaining sufficiently large existing buffers rather than allocating afresh.
        check(ggml_gallocr_reserve(allocator, graph), "graph scratch reserve failed");
        check(ggml_gallocr_alloc_graph(allocator, graph), "graph scratch allocation failed");
        stat.scratch_bytes = std::max(stat.scratch_bytes, ggml_gallocr_get_buffer_size(allocator, 0));
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, mapped_ids.data(), 0, mapped_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(rw, route_weights.data(), 0, route_weights.size() * sizeof(float));
        for (int role = 0; role < 3; ++role)
            if (scale_inputs[size_t(role)])
                ggml_backend_tensor_set(scale_inputs[size_t(role)], scales[size_t(role)].data(), 0, stat.slot_count * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
        std::vector<float> output(size_t(g.embd) * size_t(n));
        check(ggml_is_contiguous(result) && ggml_nelements(result) == int64_t(output.size()), "invalid final expert result layout");
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        ++stat.graph_batches;
        return output;
    }
};

GpuExperts::GpuExperts() : p_(std::make_unique<Impl>()) {}
GpuExperts::~GpuExperts() = default;

void GpuExperts::init(const GgufModel & model, const Geometry & g, const ExpertArena & arena,
                      ggml_backend_t backend, double cache_gib, int max_chunk, bool q8_activations) {
    check(!p_->ready && p_->backend == nullptr, "repeated initialization");
    check(backend != nullptr && ggml_backend_get_device(backend) != nullptr, "backend/device is missing");
    check(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU,
          "this opt-in path requires a GPU backend");
    const char * session = std::getenv("M31_CARD_SESSION");
    check(session && std::string(session) == "1", "GPU requires the aictl/flock runner");
    check(std::isfinite(cache_gib) && cache_gib > 0 && cache_gib <= 512,
          "cache budget must be finite and positive (GiB)");
    check(max_chunk > 0 && max_chunk <= 2048 && g.embd > 0 && g.expert_ff > 0 && g.layers > g.dense_lead &&
          g.used > 0 && g.used <= g.experts && g.experts <= 4096 && g.layers <= 1024,
          "invalid geometry/chunk");
    check(arena.buffer() != nullptr && arena.layers().size() == size_t(g.layers), "complete RAM arena is required");
    p_->g = g;
    p_->backend = backend;
    p_->device = ggml_backend_get_device(backend);
    p_->buffer_type = ggml_backend_get_default_buffer_type(backend);
    p_->max_chunk = max_chunk;
    p_->stat.budget_bytes = size_t(cache_gib * 1073741824.0);
    p_->stat.q8_activations = q8_activations;
    p_->layers.resize(size_t(g.layers));
    const size_t alignment = ggml_backend_buft_get_alignment(p_->buffer_type);
    size_t stride_multiple = alignment, maximum_span = 0;
    const char * role_names[] = {"gate", "up", "down"};
    Context metadata(64 * 1024);
    for (int l = 0; l < g.layers; ++l) {
        if (!g.routed(l)) continue;
        auto & layer = p_->layers[size_t(l)];
        ggml_reset(metadata.value);
        for (int r = 0; r < 3; ++r) {
            const std::string name = layer_name(l, std::string("ffn_") + role_names[r] + "_exps.weight");
            size_t shard = 0;
            const auto * info = model.find(name, &shard);
            const int64_t n0 = r == 2 ? g.expert_ff : g.embd;
            const int64_t n1 = r == 2 ? g.embd : g.expert_ff;
            const std::vector<uint64_t> shape = {uint64_t(n0), uint64_t(n1), uint64_t(g.experts)};
            check(info && info->shape == shape && info->type < GGML_TYPE_COUNT && model.in_bounds(*info, shard), "invalid weight " + name);
            const auto type = ggml_type(info->type);
            layer.types[size_t(r)] = type;
            layer.sources[size_t(r)] = arena.find(name);
            check(layer.sources[size_t(r)] != nullptr, "missing arena slice " + name);
            auto tensor = ggml_new_tensor_2d(metadata.value, type, n0, n1);
            layer.raw_bytes[size_t(r)] = ggml_nbytes(tensor);
            layer.allocation_bytes[size_t(r)] = ggml_backend_buft_get_alloc_size(p_->buffer_type, tensor);
            layer.offsets[size_t(r)] = aligned(layer.span, alignment);
            layer.span = layer.offsets[size_t(r)] + aligned(layer.allocation_bytes[size_t(r)], alignment);
            check(layer.raw_bytes[size_t(r)] <= layer.sources[size_t(r)]->expert_stride, "invalid arena stride");
            // CUDA casts nb[2]/type_size to a quant-block stride: rounding to 128
            // bytes alone silently truncates the expert offset for IQ3/IQ4/NVFP4.
            stride_multiple = common_multiple(stride_multiple, ggml_type_size(type));
            layer.scales[size_t(r)].assign(size_t(g.experts), 1.0f);
            const std::string scale_name = name.substr(0, name.size() - 6) + "scale";
            const auto * scales = model.find(scale_name, &shard);
            layer.has_scale[size_t(r)] = scales != nullptr;
            if (scales) {
                check(scales->type == GGML_TYPE_F32 && scales->shape == std::vector<uint64_t>{uint64_t(g.experts)} &&
                      model.in_bounds(*scales, shard), "invalid external scales " + scale_name);
                std::memcpy(layer.scales[size_t(r)].data(), model.shard(shard).tensor_data(*scales), size_t(g.experts) * sizeof(float));
            } else check(type != GGML_TYPE_NVFP4, "missing external NVFP4 scale");
            for (float scale : layer.scales[size_t(r)]) check(std::isfinite(scale), "non-finite external scale");
        }
        maximum_span = std::max(maximum_span, layer.span);
    }
    p_->stat.slot_stride = aligned(maximum_span, stride_multiple);
    p_->stat.minimum_budget_bytes = p_->stat.slot_stride;
    check(p_->stat.budget_bytes >= p_->stat.slot_stride,
          "cache budget must fit one complete expert slot: minimum bytes=" + std::to_string(p_->stat.slot_stride));
    const size_t total_experts = size_t(g.experts) * size_t(g.layers - g.dense_lead);
    p_->stat.slot_count = std::min(p_->stat.budget_bytes / p_->stat.slot_stride, total_experts);
    p_->stat.pool_bytes = p_->stat.slot_count * p_->stat.slot_stride;
    check(p_->stat.slot_count && p_->stat.slot_count <= size_t(std::numeric_limits<int32_t>::max()), "invalid slot count");
    p_->pool = ggml_backend_buft_alloc_buffer(p_->buffer_type, p_->stat.pool_bytes);
    check(p_->pool != nullptr, "fixed VRAM slot allocation failed (budget excludes dense/KV/graph scratch)");
    check(ggml_backend_buffer_get_size(p_->pool) <= p_->stat.budget_bytes, "backend allocated more than the requested cache budget");
    ggml_backend_buffer_set_usage(p_->pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    p_->stat.resident_bytes = ggml_backend_buffer_get_size(p_->pool);
    p_->stat.peak_resident_bytes = p_->stat.resident_bytes;
    p_->slot_metadata = std::make_unique<Context>((p_->stat.slot_count + 1) * ggml_tensor_overhead() + 4096);
    p_->pool_proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.pool_bytes));
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(p_->pool));
    check(ggml_backend_tensor_alloc(p_->pool, p_->pool_proxy, base) == GGML_STATUS_SUCCESS, "cannot bind pool byte tensor");
    for (size_t slot = 0; slot < p_->stat.slot_count; ++slot) {
        auto proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.slot_stride));
        check(ggml_backend_tensor_alloc(p_->pool, proxy, base + slot * p_->stat.slot_stride) == GGML_STATUS_SUCCESS,
              "cannot bind slot byte tensor");
        p_->slot_proxies.push_back(proxy);
        p_->free_slots.push_back(p_->stat.slot_count - slot - 1);
    }
    p_->allocator = ggml_gallocr_new(p_->buffer_type);
    check(p_->allocator != nullptr, "graph allocator initialization failed");
    const auto host_type = ggml_backend_dev_host_buffer_type(p_->device);
    if (host_type) p_->staging = ggml_backend_buft_alloc_buffer(host_type, p_->stat.slot_stride);
    p_->stat.pinned_staging = p_->staging != nullptr;
    if (!p_->staging) p_->pageable_staging.resize(p_->stat.slot_stride);
    std::fprintf(stderr, "M31 GPU experts: fixed_slots=%zu slot_stride=%zu pool_bytes=%zu budget=%zu pinned=%d source_q8=%d (layer mul_mat_id; graph scratch separate)\n",
                 p_->stat.slot_count, p_->stat.slot_stride, p_->stat.pool_bytes, p_->stat.budget_bytes,
                 int(p_->stat.pinned_staging), int(p_->stat.q8_activations));
    p_->ready = true;
}

std::vector<float> GpuExperts::forward(int layer, int n, const std::vector<float> & x,
                                      const std::vector<int32_t> & ids, const std::vector<float> & weights) {
    check(p_->ready && p_->g.routed(layer), "uninitialized backend or invalid routed layer");
    check(n > 0 && n <= p_->max_chunk, "invalid token count");
    const auto & g = p_->g;
    check(x.size() == size_t(g.embd) * size_t(n) && ids.size() == size_t(g.used) * size_t(n) && weights.size() == ids.size(),
          "input dimensions differ");
    std::set<int> selected;
    std::set<uint64_t> protected_keys;
    for (int t = 0; t < n; ++t) for (int rank = 0; rank < g.used; ++rank) {
        const size_t route = size_t(t) * size_t(g.used) + size_t(rank);
        const int expert = ids[route];
        check(expert >= 0 && expert < g.experts && std::isfinite(weights[route]), "invalid route ID/weight");
        for (int j = 0; j < rank; ++j)
            check(ids[size_t(t) * size_t(g.used) + size_t(j)] != expert, "duplicate top-k expert");
        selected.insert(expert);
        protected_keys.insert(p_->key(layer, expert));
    }
    check(selected.size() <= p_->stat.slot_count,
          "layer needs " + std::to_string(selected.size()) + " distinct expert slots; cache has " + std::to_string(p_->stat.slot_count) +
          "; minimum batch cache bytes=" + std::to_string(selected.size() * p_->stat.slot_stride));
    std::vector<int32_t> expert_to_slot(size_t(g.experts), -1);
    std::array<std::vector<float>, 3> scales;
    for (auto & values : scales) values.assign(p_->stat.slot_count, 1.0f);
    for (int expert : selected) {
        const size_t slot = p_->load(layer, expert, protected_keys);
        expert_to_slot[size_t(expert)] = int32_t(slot);
        for (int role = 0; role < 3; ++role)
            scales[size_t(role)][slot] = p_->layers[size_t(layer)].scales[size_t(role)][size_t(expert)];
    }
    std::vector<int32_t> mapped_ids(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) mapped_ids[i] = expert_to_slot[size_t(ids[i])];
    auto output = p_->execute(layer, n, x, mapped_ids, weights, scales);
    ++p_->stat.forwards;
    return output;
}

const GpuExperts::Stats & GpuExperts::stats() const { return p_->stat; }
void GpuExperts::reset_stats() {
    const auto old = p_->stat;
    p_->stat = {};
    p_->stat.resident_bytes = old.resident_bytes;
    p_->stat.peak_resident_bytes = old.resident_bytes;
    p_->stat.budget_bytes = old.budget_bytes;
    p_->stat.minimum_budget_bytes = old.minimum_budget_bytes;
    p_->stat.scratch_bytes = old.scratch_bytes;
    p_->stat.pinned_staging = old.pinned_staging;
    p_->stat.q8_activations = old.q8_activations;
    p_->stat.slot_stride = old.slot_stride;
    p_->stat.slot_count = old.slot_count;
    p_->stat.pool_bytes = old.pool_bytes;
    p_->stat.occupied_slots = old.occupied_slots;
}
} // namespace strata::m31
