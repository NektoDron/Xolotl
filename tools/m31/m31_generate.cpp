#include "m31_engine.hpp"
#include "m31_run.hpp"
#include "m31_mixed.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <chrono>

int main(int argc, char ** argv) {
    ggml_backend_t gpu = nullptr;
    try {
        using namespace strata::m31;
        const auto args = run::parse(argc, argv);
        if (args.oracle_op_offload) throw std::runtime_error("--oracle-op-offload is oracle-only");
        const auto cases = run::cases(args);
        if (!run::fs::create_directory(args.out)) throw std::runtime_error("output directory exists; use a new path");
        if (args.backend == "cuda") {
#ifdef M31_HAVE_CUDA
            gpu = ggml_backend_cuda_init(0);
            if (!gpu) throw std::runtime_error("cannot initialize GPU 0");
#else
            throw std::runtime_error("built without CUDA");
#endif
        }
        {
            Engine engine;
            Config cfg;
            cfg.backend = gpu;
            cfg.threads = args.threads;
            cfg.context = args.context;
            cfg.max_chunk = args.chunk;
            cfg.arena_gib = args.arena_gib;
            cfg.gpu_experts_gib = args.gpu_experts_gib;
            cfg.gpu_experts_min_tokens = args.gpu_experts_min_tokens;
            cfg.reuse_allocations = args.reuse_allocations;
            cfg.persistent_cpu_pool = args.persistent_cpu_pool;
            cfg.cpu_residual = args.cpu_residual;
            cfg.cpu_decode_cache = args.cpu_decode_cache;
            cfg.dense_decode_cache = args.dense_decode_cache;
            cfg.profile = args.profile;
            run::Dump * active_dump = nullptr;
            if (args.taps != "none") cfg.tap = [&](const auto & name, int64_t pos, const auto * tensor) {
                if (active_dump) active_dump->tensor(name, pos, tensor);
            };
            std::string error;
            if (!engine.init(args.model, cfg, error)) throw std::runtime_error(error);
            for (const auto & path : cases) {
                MixedInput mixed;
                if (!args.mixed_manifest.empty()) mixed = load_mixed_manifest(path, int(engine.geometry().vocab), args.context);
                auto ids = args.mixed_manifest.empty() ? run::read_ids(path) : mixed.tokens;
                if (ids.size() > size_t(args.context)) throw std::runtime_error("case exceeds configured context");
                engine.reset();
                run::Dump dump(run::fs::path(args.out) / path.stem(), args, ids, int(engine.geometry().vocab));
                active_dump = &dump;
                const auto t0 = std::chrono::steady_clock::now();
                for (int pos = 0; pos < int(ids.size()); ) {
                    const int requested = run::chunk_size(args, int(ids.size()), pos);
                    const int n = engine.prefill_chunk(requested, error);
                    if (n <= 0) throw std::runtime_error(error);
                    dump.record_chunk(pos, n);
                    const bool ok = args.mixed_manifest.empty()
                        ? engine.forward(ids.data() + pos, n, error)
                        : engine.forward_mixed(ids.data() + pos, mixed.visual_rows.data() + pos,
                                               mixed.embeddings.data(), mixed.visual_count, n, error);
                    if (!ok) throw std::runtime_error(error);
                    // Match the oracle's batched head (MMQ vs single-row MMV has
                    // different activation rounding). Chunk bounds vocab scratch.
                    const int head_count = args.head_tail ? std::min(n, args.head_tail) : n;
                    const int head_first = n - head_count;
                    std::vector<float> logits;
                    if (!engine.logits(head_first, head_count, logits, error)) throw std::runtime_error(error);
                    for (int r = head_first; r < n; ++r) if (dump.wants(pos + r))
                        dump.row(pos + r, logits.data() + size_t(r - head_first) * engine.geometry().vocab);
                    pos += n;
                }
                dump.finish("xolotl-m31-correctness", int(ids.size()));
                active_dump = nullptr;
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "M31 case=%s tokens=%zu rows=%zu seconds=%.3f (correctness run, not M2 benchmark)\n",
                             path.stem().string().c_str(), ids.size(), dump.positions.size(), s);
            }
        }
        if (gpu) ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "m31_generate: %s\n", e.what());
        if (gpu) ggml_backend_free(gpu);
        return 1;
    }
}
