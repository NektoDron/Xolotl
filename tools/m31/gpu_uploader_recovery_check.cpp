// Small original-byte transport check: no attention graph, so tiny CUDA FA is not required.
#include "m31_gpu_experts.hpp"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace strata::m31;
void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        check(argc == 3, "usage: gpu_uploader_recovery_check TINY.gguf REPORT.json");
        check(std::getenv("M31_CARD_SESSION") && std::string(std::getenv("M31_CARD_SESSION")) == "1", "aictl runner required");
        unsetenv("M31_GPU_CPU_QUANTS");
        strata::GgufModel model(strata::gguf_split_paths(argv[1]));
        const auto g = read_geometry(model);
        check(g.embd <= 64 && g.experts == 4 && g.used == 2, "small fixture expected");
        backend = ggml_backend_cuda_init(0);
        check(backend, "CUDA init failed");
        {
            ExpertArena arena; arena.init(model, g, .01);
            unsetenv("M31_GPU_PREFILL_ASYNC"); unsetenv("M31_GPU_PREFILL_TEST_FAIL_AFTER");
            GpuExperts baseline; baseline.init(model, g, arena, backend, .00005, 7);
            setenv("M31_GPU_PREFILL_ASYNC", "1", 1); setenv("M31_GPU_PREFILL_TEST_FAIL_AFTER", "2", 1);
            GpuExperts candidate; candidate.init(model, g, arena, backend, .00005, 7);
            std::vector<float> x(size_t(g.embd)*7), weights(size_t(g.used)*7);
            std::vector<int32_t> ids(weights.size());
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i)*.12f);
            for (int t = 0; t < 7; ++t) for (int r = 0; r < g.used; ++r) {
                const auto i = size_t(t)*size_t(g.used)+size_t(r);
                ids[i] = (t+r)%4; weights[i] = r ? .7f : 1.3f;
            }
            bool failed = false;
            try { candidate.forward(int(g.dense_lead), 7, x, ids, weights); }
            catch (const std::exception & e) { failed = std::string(e.what()) == "M31 prefill injected upload failure"; }
            check(failed, "queued upload failure was not injected");
            check(candidate.stats().occupied_slots == 0, "failed batch left stale cache entries");
            size_t batches = 0, elements = 0;
            for (int repeat = 0; repeat < 3; ++repeat) for (int layer = int(g.dense_lead); layer < g.layers; ++layer) {
                const auto reference = baseline.forward(layer, 7, x, ids, weights);
                const auto actual = candidate.forward(layer, 7, x, ids, weights);
                check(reference.size() == actual.size() && std::memcmp(reference.data(), actual.data(), actual.size()*sizeof(float)) == 0,
                      "recovered pipeline differs from synchronous baseline");
                ++batches; elements += actual.size();
            }
            check(candidate.stats().evictions > 0, "slot reuse/eviction was not exercised");
            std::ofstream report(argv[2]);
            report << "{\"pass\":true,\"injected_failure\":true,\"stale_entries_after_failure\":0,\"batches\":" << batches
                   << ",\"bitwise_elements\":" << elements << ",\"evictions\":" << candidate.stats().evictions << "}\n";
            check(bool(report), "report write failed");
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "uploader recovery check: %s\n", e.what());
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
