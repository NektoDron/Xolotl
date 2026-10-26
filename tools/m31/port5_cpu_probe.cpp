#include "cpu_direct_decode.hpp"
#include "cpu_fused_decode.hpp"
#include "cpu_threadpool_config.hpp"
#include "m31_geometry.hpp"
#include "ggml-alloc.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

using namespace strata::m31;
void ensure(bool ok, const char * error) { if (!ok) throw std::runtime_error(error); }
int main(int argc, char ** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        ensure(argc >= 2, "usage: port5_cpu_probe MODEL [threads] [repetitions]");
        const int threads = argc > 2 ? std::stoi(argv[2]) : 16;
        const int repeats = argc > 3 ? std::stoi(argv[3]) : 0;
        ensure(threads >= 1 && threads <= 64 && repeats >= 0 && repeats <= 100, "invalid probe limits");
        strata::GgufModel model(strata::gguf_split_paths(argv[1]));
        const auto geom = read_geometry(model);
        auto backend = ggml_backend_cpu_init(); ensure(backend, "backend");
        ggml_backend_cpu_set_n_threads(backend, threads);
        auto params = ggml_threadpool_params_default(threads);
        apply_cpu_pool_environment(params);
        auto pool = ggml_threadpool_new(&params); ensure(pool, "pool");
        ggml_backend_cpu_set_threadpool(backend, pool);
        CpuDirectDecode direct{unsigned(threads)};
        auto fused=std::make_unique<CpuFusedDecode>(backend,pool,unsigned(threads));
        const std::array<int, 5> layers{3, 10, 52, 58, 59};
        for (const int layer : layers) {
            auto context = ggml_init({8*1024*1024, nullptr, true}); ensure(context, "context");
            std::array<ggml_tensor *, 3> wt;
            std::array<std::vector<float>, 3> scales;
            std::array<bool, 3> has_scale{};
            const char * roles[] = {"gate", "up", "down"};
            for (int r = 0; r < 3; ++r) {
                size_t shard = 0;
                std::string name = layer_name(layer, std::string("ffn_") + roles[r] + "_exps.weight");
                auto info = model.find(name, &shard); ensure(info, "missing weight");
                wt[r] = ggml_new_tensor_3d(context, ggml_type(info->type), info->shape[0], info->shape[1], 4);
                scales[r].resize(4, 1.f);
                const std::string scale_name = name.substr(0, name.size()-6) + "scale";
                size_t ss = 0;
                if (auto scale = model.find(scale_name, &ss)) {
                    has_scale[r] = true;
                    std::memcpy(scales[r].data(), model.shard(ss).tensor_data(*scale), 4*sizeof(float));
                }
            }
            auto weights_buffer = ggml_backend_alloc_ctx_tensors(context, backend); ensure(weights_buffer, "weights buffer");
            for (int r = 0; r < 3; ++r) {
                size_t shard = 0;
                auto info = model.find(layer_name(layer, std::string("ffn_")+roles[r]+"_exps.weight"), &shard);
                ggml_backend_tensor_set(wt[r], model.shard(shard).tensor_data(*info), 0, ggml_nbytes(wt[r]));
            }
            auto graph = ggml_new_graph_custom(context, 256, false);
            auto inp = ggml_new_tensor_3d(context, GGML_TYPE_F32, geom.embd, 1, 1);
            auto ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, 4, 1);
            auto rw = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 4, 1);
            ggml_set_input(inp); ggml_set_input(ids); ggml_set_input(rw);
            auto mm = [&](int role, ggml_tensor * x) {
                auto result = ggml_mul_mat_id(context, wt[role], x, ids);
                if (wt[role]->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(result, GGML_PREC_BF16);
                if (has_scale[role]) {
                    auto scale = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 4, 1);
                    ggml_set_input(scale);
                    // Defer upload until graph allocator has assigned its storage.
                    ggml_set_name(scale, (std::string("scale") + std::to_string(role)).c_str());
                    result = ggml_mul(context, result, ggml_get_rows(context, scale, ids));
                }
                return result;
            };
            auto gate = mm(0, inp), up = mm(1, inp);
            auto hidden = ggml_swiglu_oai(context, gate, up, 1.702f, 7.f);
            auto down = mm(2, hidden);
            auto weighted = ggml_mul(context, down, rw);
            ggml_tensor * result = nullptr;
            for (int k=0; k<4; ++k) {
                auto view = ggml_view_2d(context, weighted, geom.embd, 1, weighted->nb[2], size_t(k)*weighted->nb[1]);
                result = result ? ggml_add(context, result, view) : view;
            }
            result = ggml_cont(context, result); ggml_set_output(result); ggml_build_forward_expand(graph, result);
            auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ensure(ggml_gallocr_alloc_graph(alloc, graph), "graph allocate");
            std::vector<float> input(size_t(geom.embd)), weights{.1f,.2f,.3f,.4f}, actual, expected(size_t(geom.embd));
            std::vector<int32_t> experts{3, 0, 2, 1};
            for (size_t i=0; i<input.size(); ++i) input[i] = std::sin(float(i)*.017f);
            auto upload = [&] {
                ggml_backend_tensor_set(inp, input.data(), 0, input.size()*sizeof(float));
                ggml_backend_tensor_set(ids, experts.data(), 0, experts.size()*sizeof(int32_t));
                ggml_backend_tensor_set(rw, weights.data(), 0, weights.size()*sizeof(float));
                for (int r=0; r<3; ++r) if(has_scale[r]) {
                    auto scale = ggml_get_tensor(context, (std::string("scale")+std::to_string(r)).c_str());
                    ggml_backend_tensor_set(scale, scales[r].data(), 0, 4*sizeof(float));
                }
            };
            upload();
            ensure(ggml_backend_graph_compute(backend, graph)==GGML_STATUS_SUCCESS, "graph compute");
            ggml_backend_tensor_get(result, expected.data(), 0, expected.size()*sizeof(float));
            auto run_direct = [&] {
                direct.forward(wt[0], wt[1], wt[2], has_scale[0]?scales[0].data():nullptr,
                    has_scale[1]?scales[1].data():nullptr, has_scale[2]?scales[2].data():nullptr,
                    input, experts, weights, actual);
            };
            run_direct();
            size_t diffs=0; float maxabs=0;
            for(size_t i=0; i<expected.size(); ++i) {
                if(std::memcmp(&expected[i], &actual[i], sizeof(float))) ++diffs;
                maxabs=std::max(maxabs,std::abs(expected[i]-actual[i]));
            }
            std::printf("layer=%d threads=%d diff=%zu maxabs=%.9g\n", layer, threads, diffs, maxabs);
            ensure(diffs==0, "direct CPU differs from CPU graph");
            auto run_fused = [&] {
                fused->forward(wt[0], wt[1], wt[2], has_scale[0]?scales[0].data():nullptr,
                    has_scale[1]?scales[1].data():nullptr, has_scale[2]?scales[2].data():nullptr,
                    input, experts, weights, actual);
            };
            run_fused();
            ensure(!std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)),"fused CPU differs from CPU graph");
            std::printf("fused layer=%d threads=%d diff=0\n",layer,threads);
            if (repeats) {
                auto start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i) { upload(); ensure(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat graph"); }
                double graph_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
                start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i) run_fused();
                double fused_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
                ggml_threadpool_pause(pool);
                start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i) run_direct();
                double direct_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
                std::printf("timing layer=%d graph_ms=%.6f direct_ms=%.6f fused_ms=%.6f repeats=%d\n",layer,graph_ms,direct_ms,fused_ms,repeats);
            }
            ggml_gallocr_free(alloc); ggml_backend_buffer_free(weights_buffer); ggml_free(context);
        }
        fused.reset();
        ggml_backend_cpu_set_threadpool(backend,nullptr); ggml_threadpool_free(pool); ggml_backend_free(backend);
    } catch (const std::exception & error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
