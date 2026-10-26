#include "m31_engine.hpp"
#include "m31_run.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <chrono>

int main(int argc, char ** argv) {
    ggml_backend_t gpu = nullptr;
    try {
        using namespace strata::m31;
        const auto args = run::parse(argc, argv);
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
            run::Dump * active_dump = nullptr;
            if (args.taps != "none") cfg.tap = [&](const auto & name, int64_t pos, const auto * tensor) {
                if (active_dump) active_dump->tensor(name, pos, tensor);
            };
            std::string error;
            if (!engine.init(args.model, cfg, error)) throw std::runtime_error(error);
            for (const auto & path : cases) {
                auto ids = run::read_ids(path);
                if (ids.size() > size_t(args.context)) throw std::runtime_error("case exceeds configured context");
                engine.reset();
                run::Dump dump(run::fs::path(args.out) / path.stem(), args, ids, int(engine.geometry().vocab));
                active_dump = &dump;
                const auto t0 = std::chrono::steady_clock::now();
                for (int pos = 0; pos < int(ids.size()); ) {
                    const int n = run::chunk_size(args, int(ids.size()), pos);
                    if (!engine.forward(ids.data() + pos, n, error)) throw std::runtime_error(error);
                    // Match the oracle's batched head (MMQ vs single-row MMV has
                    // different activation rounding). Chunk bounds vocab scratch.
                    std::vector<float> logits;
                    if (!engine.logits(0, n, logits, error)) throw std::runtime_error(error);
                    for (int r = 0; r < n; ++r) if (dump.wants(pos + r))
                        dump.row(pos + r, logits.data() + size_t(r) * engine.geometry().vocab);
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
