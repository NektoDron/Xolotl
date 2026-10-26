// Read-only calibration: observe the unmodified router, never change selected IDs.
#include "m31_engine.hpp"
#include "m31_run.hpp"
#include "ggml-cuda.h"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;
using namespace strata::m31;

int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const auto args = run::parse(argc, argv);
        if (args.arena_gib || args.gpu_experts_gib || args.oracle_op_offload || args.taps != "none")
            throw std::runtime_error("calibration uses original mapped CPU experts without taps/offload");
        const auto inputs = run::cases(args);
        if (run::fs::exists(args.out)) throw std::runtime_error("use a fresh output path");
        std::ofstream output(args.out);
        if (!output) throw std::runtime_error("cannot create output");
        if (args.backend == "cuda") {
            backend = ggml_backend_cuda_init(0); // run::parse enforces M31_CARD_SESSION.
            if (!backend) throw std::runtime_error("cannot initialize GPU0");
        }
        {
            Config cfg;
            cfg.backend = backend;
            cfg.threads = args.threads;
            cfg.context = args.context;
            cfg.max_chunk = args.chunk;
            cfg.reuse_allocations = true;
            cfg.persistent_cpu_pool = true;
            cfg.cpu_residual = true;
            cfg.cpu_decode_cache = true;
            std::string name;
            int64_t position = 0;
            int used = 0, experts = 0;
            cfg.route_observer = [&](int layer, int n, const std::vector<int32_t> & ids,
                                     const std::vector<float> & weights) {
                if (ids.size() != size_t(n * used) || weights.size() != ids.size())
                    throw std::runtime_error("invalid route observation dimensions");
                for (size_t i = 0; i < ids.size(); ++i)
                    if (ids[i] < 0 || ids[i] >= experts || !std::isfinite(weights[i]) || weights[i] < 0)
                        throw std::runtime_error("invalid route observation");
                output << json{{"kind", "routes"}, {"case", name}, {"position", position},
                    {"layer", layer}, {"n", n}, {"ids", ids}, {"weights", weights}}.dump() << '\n';
                if (!output) throw std::runtime_error("route output write failed");
            };
            Engine engine;
            std::string error;
            if (!engine.init(args.model, cfg, error)) throw std::runtime_error(error);
            const auto & g = engine.geometry();
            used = int(g.used); experts = int(g.experts);
            output << json{{"kind", "header"}, {"format", "m31-routing-trace-v1"},
                {"model", run::fs::absolute(args.model).string()}, {"backend", args.backend},
                {"cpu_experts", true}, {"layers", g.layers}, {"dense_lead", g.dense_lead},
                {"experts", g.experts}, {"used", g.used}, {"embd", g.embd},
                {"expert_ff", g.expert_ff}, {"heads", g.heads}, {"kv_heads", g.kv_heads},
                {"head_dim", g.head_dim}, {"index_dim", g.index_dim}, {"index_heads", g.index_heads},
                {"block", g.block}, {"top_blocks", g.top_blocks},
                {"gpu_weight_bytes", g.other_bytes - g.embedding_bytes - g.expert_scale_bytes},
                {"expert_bytes", g.expert_bytes}, {"chunk", args.chunk}, {"threads", args.threads},
                {"cuda_fusion_disabled", std::getenv("GGML_CUDA_DISABLE_FUSION") &&
                    std::string(std::getenv("GGML_CUDA_DISABLE_FUSION")) == "1"}}.dump() << '\n';
            for (const auto & input : inputs) {
                const auto ids = run::read_ids(input);
                engine.reset();
                name = input.stem().string();
                output << json{{"kind", "case"}, {"case", name}, {"tokens", ids.size()},
                               {"input", run::fs::absolute(input).string()}}.dump() << '\n';
                for (position = 0; position < int64_t(ids.size());) {
                    const int n = run::chunk_size(args, int(ids.size()), int(position));
                    if (!engine.forward(ids.data() + position, n, error)) throw std::runtime_error(error);
                    position += n;
                }
                output.flush();
                std::cerr << "calibrated " << name << " tokens=" << ids.size() << '\n';
            }
        }
        if (backend) { ggml_backend_free(backend); backend = nullptr; }
        return 0;
    } catch (const std::exception & e) {
        if (backend) ggml_backend_free(backend);
        std::cerr << "cache_calibrate: " << e.what() << '\n';
        return 1;
    }
}
