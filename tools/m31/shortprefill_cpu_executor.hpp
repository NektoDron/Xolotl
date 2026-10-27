#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::m31 {

// Serialized CPU lane for a complete set of assignments of selected experts.
// Uses the Engine's persistent CPU backend/pool and original, immutable arena
// tensors. Each expert retains its complete assignment histogram and original
// token/rank order, including ggml's scalar/tiled boundary at eight assignments.
// The result deliberately has no rank reduction: GPU and CPU lanes must combine
// their weighted terms in the original rank order after both lanes finish.
class ShortPrefillCpuExecutor {
public:
    struct Stats {
        uint64_t forwards = 0, assignments = 0, graph_builds = 0, graph_hits = 0;
        double setup_seconds = 0, graph_build_seconds = 0, pack_seconds = 0, compute_seconds = 0, read_seconds = 0;
        size_t retained_graphs = 0, retained_scratch_bytes = 0;
    };

private:
    using Clock = std::chrono::steady_clock;
    struct Graph {
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * graph = nullptr;
        std::array<ggml_tensor *, 3> weight{};
        std::array<ggml_tensor *, 3> scale{};
        ggml_tensor * input = nullptr, * ids = nullptr, * routes = nullptr, * output = nullptr;
        std::vector<int64_t> key;
        ~Graph() {
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }
    };
    ggml_backend_t backend_;
    size_t graph_limit_;
    std::list<std::unique_ptr<Graph>> graphs_;
    std::vector<size_t> order_;
    std::vector<int32_t> compact_ids_;
    std::vector<float> packed_input_, packed_routes_, packed_scale_, packed_output_;
    Stats stats_;

    static void require(bool value, const char * message) {
        if (!value) throw std::invalid_argument(message);
    }
    static double seconds(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
    static size_t product(size_t a, size_t b) {
        require(!b || a <= std::numeric_limits<size_t>::max() / b,
                "M31 short CPU: size overflow");
        return a * b;
    }
    static ggml_tensor * input(ggml_context * ctx, ggml_type type, int64_t a, int64_t b, int64_t c) {
        auto * result = ggml_new_tensor_3d(ctx, type, a, b, c);
        ggml_set_input(result);
        return result;
    }
    static std::vector<int64_t> key_for(const std::array<const ggml_tensor *, 3> & weights,
                                      const std::array<const float *, 3> & scales, size_t count) {
        std::vector<int64_t> key{int64_t(count)};
        for (size_t role = 0; role < weights.size(); ++role) {
            key.push_back(int64_t(weights[role]->type));
            key.push_back(scales[role] != nullptr);
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                key.push_back(weights[role]->ne[dim]);
                key.push_back(int64_t(weights[role]->nb[dim]));
            }
        }
        return key;
    }
    Graph & graph_for(const std::array<const ggml_tensor *, 3> & weights,
                      const std::array<const float *, 3> & scales, size_t count) {
        const auto key = key_for(weights, scales, count);
        auto found = std::find_if(graphs_.begin(), graphs_.end(), [&](const auto & graph) {
            return graph->key == key;
        });
        if (found != graphs_.end()) {
            graphs_.splice(graphs_.begin(), graphs_, found);
            ++stats_.graph_hits;
        } else {
            const auto graph_start = Clock::now();
            // Evict before allocating another scratch arena, bounding live memory.
            if (graphs_.size() == graph_limit_) graphs_.pop_back();
            auto graph = std::make_unique<Graph>();
            graph->key = key;
            graph->context = ggml_init({128 * 1024, nullptr, true});
            if (!graph->context) throw std::runtime_error("M31 short CPU: graph context allocation failed");
            auto * ctx = graph->context;
            const int64_t embd = weights[0]->ne[0];
            graph->input = input(ctx, GGML_TYPE_F32, embd, 1, int64_t(count));
            graph->ids = input(ctx, GGML_TYPE_I32, 1, int64_t(count), 1);
            graph->routes = input(ctx, GGML_TYPE_F32, 1, 1, int64_t(count));
            for (size_t role = 0; role < 3; ++role) {
                auto * alias = ggml_dup_tensor(ctx, weights[role]);
                for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) alias->nb[dim] = weights[role]->nb[dim];
                alias->buffer = weights[role]->buffer;
                alias->data = weights[role]->data;
                graph->weight[role] = alias;
                if (scales[role]) graph->scale[role] = input(ctx, GGML_TYPE_F32, 1, 1, int64_t(count));
            }
            auto mm = [&](size_t role, ggml_tensor * activation) {
                auto * result = ggml_mul_mat_id(ctx, graph->weight[role], activation, graph->ids);
                if (weights[role]->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(result, GGML_PREC_BF16);
                if (graph->scale[role]) result = ggml_mul(ctx, result, graph->scale[role]);
                return result;
            };
            auto * gate = mm(0, graph->input);
            auto * up = mm(1, graph->input);
            auto * hidden = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.f);
            graph->output = ggml_mul(ctx, mm(2, hidden), graph->routes);
            ggml_set_output(graph->output);
            graph->graph = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(graph->graph, graph->output);
            graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (!graph->allocator || !ggml_gallocr_alloc_graph(graph->allocator, graph->graph))
                throw std::runtime_error("M31 short CPU: graph scratch allocation failed");
            graphs_.push_front(std::move(graph));
            ++stats_.graph_builds;
            stats_.graph_build_seconds += seconds(graph_start);
        }
        auto & graph = *graphs_.front();
        for (size_t role = 0; role < 3; ++role) {
            graph.weight[role]->buffer = weights[role]->buffer;
            graph.weight[role]->data = weights[role]->data;
        }
        stats_.retained_graphs = graphs_.size();
        stats_.retained_scratch_bytes = 0;
        for (const auto & retained : graphs_)
            stats_.retained_scratch_bytes += ggml_gallocr_get_buffer_size(retained->allocator, 0);
        return graph;
    }

public:
    explicit ShortPrefillCpuExecutor(ggml_backend_t cpu, size_t retained_graph_limit = 4)
        : backend_(cpu), graph_limit_(retained_graph_limit) {
        require(cpu && ggml_backend_is_cpu(cpu) && graph_limit_ > 0 && graph_limit_ <= 16,
                "M31 short CPU: requires CPU backend and graph cache limit 1..16");
    }
    ShortPrefillCpuExecutor(const ShortPrefillCpuExecutor &) = delete;
    ShortPrefillCpuExecutor & operator=(const ShortPrefillCpuExecutor &) = delete;

    // x=[embd,n], ids/weights=[used,n]. Each route is token*used+rank.
    // Every assignment of an included expert must be present. Output is
    // [embd,route_indices.size()] in the caller's original route_indices order.
    void forward(const ggml_tensor * wg, const ggml_tensor * wu, const ggml_tensor * wd,
                 const float * gate_scale, const float * up_scale, const float * down_scale,
                 const std::vector<float> & x, const std::vector<int32_t> & ids,
                 const std::vector<float> & weights, const std::vector<size_t> & route_indices,
                 size_t used, std::vector<float> & output) {
        require(wg && wu && wd && wg->data && wu->data && wd->data && used > 0,
                "M31 short CPU: missing weights/used");
        const auto setup_start = Clock::now();
        const size_t embd = size_t(wg->ne[0]), experts = size_t(wg->ne[2]);
        require(embd > 0 && experts > 0 && used <= experts && wg->ne[1] > 0 &&
                wg->ne[3] == 1 && wu->ne[3] == 1 && wd->ne[3] == 1 &&
                wu->ne[0] == wg->ne[0] && wu->ne[1] == wg->ne[1] &&
                wd->ne[0] == wg->ne[1] && wd->ne[1] == wg->ne[0] &&
                wu->ne[2] == wg->ne[2] && wd->ne[2] == wg->ne[2] &&
                !x.empty() && x.size() % embd == 0,
                "M31 short CPU: invalid expert/input shape");
        const size_t n = x.size() / embd;
        require(ids.size() == product(used, n) && weights.size() == ids.size(),
                "M31 short CPU: invalid route geometry");
        require(&output != &x && &output != &weights, "M31 short CPU: output aliases input");
        std::vector<size_t> full_counts(experts, 0), selected_counts(experts, 0);
        for (size_t route = 0; route < ids.size(); ++route) {
            require(ids[route] >= 0 && size_t(ids[route]) < experts && std::isfinite(weights[route]),
                    "M31 short CPU: invalid route ID/weight");
            for (size_t rank = route - route % used; rank < route; ++rank)
                require(ids[rank] != ids[route], "M31 short CPU: duplicate expert within token");
            ++full_counts[size_t(ids[route])];
        }
        order_.resize(route_indices.size());
        for (size_t i = 0; i < order_.size(); ++i) order_[i] = i;
        std::sort(order_.begin(), order_.end(), [&](size_t a, size_t b) { return route_indices[a] < route_indices[b]; });
        size_t previous = ids.size();
        for (size_t index : order_) {
            const size_t route = route_indices[index];
            require(route < ids.size() && route != previous, "M31 short CPU: invalid/duplicate selected route");
            previous = route;
            ++selected_counts[size_t(ids[route])];
        }
        for (size_t expert = 0; expert < experts; ++expert)
            require(!selected_counts[expert] || selected_counts[expert] == full_counts[expert],
                    "M31 short CPU: partial expert would change grouped arithmetic");
        if (route_indices.empty()) { output.clear(); return; }
        const std::array<const ggml_tensor *, 3> tensors{wg, wu, wd};
        const std::array<const float *, 3> scales{gate_scale, up_scale, down_scale};
        auto & graph = graph_for(tensors, scales, route_indices.size());
        stats_.setup_seconds += seconds(setup_start);
        const auto pack_start = Clock::now();
        packed_input_.resize(product(embd, order_.size()));
        compact_ids_.resize(order_.size());
        packed_routes_.resize(order_.size());
        for (size_t compact = 0; compact < order_.size(); ++compact) {
            const size_t route = route_indices[order_[compact]], token = route / used;
            std::memcpy(packed_input_.data() + compact * embd, x.data() + token * embd, embd * sizeof(float));
            compact_ids_[compact] = ids[route];
            packed_routes_[compact] = weights[route];
        }
        ggml_backend_tensor_set(graph.input, packed_input_.data(), 0, ggml_nbytes(graph.input));
        ggml_backend_tensor_set(graph.ids, compact_ids_.data(), 0, ggml_nbytes(graph.ids));
        ggml_backend_tensor_set(graph.routes, packed_routes_.data(), 0, ggml_nbytes(graph.routes));
        for (size_t role = 0; role < 3; ++role) if (scales[role]) {
            packed_scale_.resize(order_.size());
            for (size_t compact = 0; compact < order_.size(); ++compact)
                packed_scale_[compact] = scales[role][size_t(compact_ids_[compact])];
            ggml_backend_tensor_set(graph.scale[role], packed_scale_.data(), 0, ggml_nbytes(graph.scale[role]));
        }
        stats_.pack_seconds += seconds(pack_start);
        const auto compute_start = Clock::now();
        if (ggml_backend_graph_compute(backend_, graph.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("M31 short CPU: graph compute failed");
        stats_.compute_seconds += seconds(compute_start);
        const auto read_start = Clock::now();
        packed_output_.resize(product(embd, order_.size()));
        output.resize(packed_output_.size());
        ggml_backend_tensor_get(graph.output, packed_output_.data(), 0, ggml_nbytes(graph.output));
        for (size_t compact = 0; compact < order_.size(); ++compact)
            std::memcpy(output.data() + order_[compact] * embd, packed_output_.data() + compact * embd, embd * sizeof(float));
        stats_.read_seconds += seconds(read_start);
        ++stats_.forwards;
        stats_.assignments += order_.size();
    }
    const Stats & stats() const { return stats_; }
};
} // namespace strata::m31
