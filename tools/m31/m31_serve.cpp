// Engine subprocess for the opt-in M31 HTTP adapter. One request at a time.
#include "m31_engine.hpp"
#include "json.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace strata::m31;
static double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
static void send(const json & value) { std::cout << value.dump() << '\n' << std::flush; }
#include "server_sampling.hpp"

int main(int argc, char ** argv) {
    ggml_backend_t gpu = nullptr;
    try {
        std::string model, backend = "cpu";
        Config cfg;
        cfg.context = 4096;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--model" || key == "-m") model = value;
            else if (key == "--backend") backend = value;
            else if (key == "--ctx") cfg.context = std::stoll(value);
            else if (key == "--chunk") cfg.max_chunk = std::stoi(value);
            else if (key == "--threads") cfg.threads = std::stoi(value);
            else if (key == "--arena-gib") cfg.arena_gib = std::stod(value);
            else if (key == "--gpu-experts-gib") cfg.gpu_experts_gib = std::stod(value);
            else if (key == "--gpu-experts-min-tokens") cfg.gpu_experts_min_tokens = std::stoi(value);
            else if (key == "--reuse-allocations") cfg.reuse_allocations = std::stoi(value) != 0;
            else if (key == "--persistent-cpu-pool") cfg.persistent_cpu_pool = std::stoi(value) != 0;
            else if (key == "--cpu-decode-cache") cfg.cpu_decode_cache = std::stoi(value) != 0;
            else if (key == "--dense-decode-cache") cfg.dense_decode_cache = std::stoi(value) != 0;
            else if (key == "--cpu-residual") cfg.cpu_residual = std::stoi(value) != 0;
            else if (key == "--profile") cfg.profile = std::stoi(value) != 0;
            else throw std::runtime_error("unknown option " + key);
        }
        if (model.empty() || cfg.context < 1 || cfg.context > 1048576 || cfg.max_chunk < 1 || cfg.max_chunk > 2048 || cfg.threads < 1)
            throw std::runtime_error("invalid model/context/chunk/threads");
        if (backend == "cuda") {
            const char * session = std::getenv("M31_CARD_SESSION");
            if (!session || std::string(session) != "1") throw std::runtime_error("CUDA requires the aictl/flock runner");
#ifdef M31_HAVE_CUDA
            gpu = ggml_backend_cuda_init(0);
            if (!gpu) throw std::runtime_error("GPU 0 initialization failed");
#else
            throw std::runtime_error("built without CUDA");
#endif
        } else if (backend != "cpu") throw std::runtime_error("invalid backend");
        cfg.backend = gpu;
        {
            Engine engine;
            std::string error;
            if (!engine.init(model, cfg, error)) throw std::runtime_error(error);
            send({{"event", "ready"}, {"vocab", engine.geometry().vocab}, {"context", cfg.context}});
            std::string line;
            while (std::getline(std::cin, line)) {
                try {
                    const auto req = json::parse(line);
                    auto ids = req.at("tokens").get<std::vector<int32_t>>();
                    const int maximum = req.value("max_tokens", 256);
                    const double temperature = req.value("temperature", 1.0), top_p = req.value("top_p", 0.95), min_p = req.value("min_p", 0.05);
                    const int top_k = req.value("top_k", 40);
                    if (ids.empty() || maximum < 0 || !std::isfinite(temperature) || temperature < 0 ||
                        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || !std::isfinite(min_p) || min_p < 0 || min_p > 1 || top_k < 0)
                        throw std::runtime_error("invalid request/sampling parameters");
                    if (int64_t(ids.size()) > cfg.context || int64_t(ids.size()) + maximum > cfg.context)
                        throw std::runtime_error("prompt plus maximum generation exceeds context");
                    for (int token : ids) if (token < 0 || token >= engine.geometry().vocab) throw std::runtime_error("token out of vocabulary");
                    const auto stops = req.value("stop_ids", std::vector<int>{});
                    const std::set<int> stop_ids(stops.begin(), stops.end());
                    const bool ignore_eos = req.value("ignore_eos", false);
                    const auto suppressed = req.value("suppress_ids", std::vector<int>{});
                    std::set<int> excluded(suppressed.begin(), suppressed.end());
                    if (ignore_eos) excluded.insert(stop_ids.begin(), stop_ids.end());
                    const uint64_t seed = req.value("seed", uint64_t(std::random_device{}()));
                    std::mt19937_64 rng(seed);
                    size_t grammar_checks = 0, grammar_rejections = 0, sampling_calls = 0, interactive_waits = 0;
                    double sampling_ms = 0, grammar_wait_ms = 0, interactive_wait_ms = 0;
                    std::vector<float> logits_for_sampling;
                    auto draw = [&](const std::set<int> * allowed = nullptr) {
                        const auto started = Clock::now();
                        const int token = sample(logits_for_sampling, temperature, top_p, top_k, min_p, excluded, rng, allowed);
                        sampling_ms += ms(started);
                        ++sampling_calls;
                        return token;
                    };
                    auto choose = [&]() {
                        int token = draw();
                        if (!req.value("grammar_check", false)) return token;
                        for (int attempt = 0; attempt < 2; ++attempt) {
                            const auto grammar_started = Clock::now();
                            ++grammar_checks;
                            send({{"event", "sample"}, {"token", token}});
                            std::string response;
                            if (!std::getline(std::cin, response)) throw std::runtime_error("grammar controller disconnected");
                            grammar_wait_ms += ms(grammar_started);
                            const auto control = json::parse(response);
                            if (control.value("accepted", false)) return token;
                            ++grammar_rejections;
                            const auto ids = control.at("allowed").get<std::vector<int>>();
                            const std::set<int> allowed(ids.begin(), ids.end());
                            if (allowed.empty()) throw std::runtime_error("empty grammar sampling support");
                            token = draw(&allowed);
                        }
                        throw std::runtime_error("grammar rejected filtered sample");
                    };
                    engine.reset();
                    const auto start = Clock::now();
                    int last_n = 0;
                    for (size_t p = 0; p < ids.size();) {
                        last_n = int(std::min(size_t(cfg.max_chunk), ids.size() - p));
                        if (!engine.forward(ids.data() + p, last_n, error)) throw std::runtime_error(error);
                        p += size_t(last_n);
                    }
                    if (maximum && !engine.logits(last_n - 1, 1, logits_for_sampling, error)) throw std::runtime_error(error);
                    // The reference counts the first sample in prefill and measures n_gen-1
                    // generation steps, ending at the final sample rather than token transport.
                    int sampled = maximum ? choose() : -1;
                    const auto decode_start = Clock::now();
                    auto decode_end = decode_start;
                    const double prompt_ms = std::chrono::duration<double, std::milli>(decode_start - start).count();
                    send({{"event", "prefill"}, {"prompt_n", ids.size()}, {"prompt_ms", prompt_ms}});
                    int predicted = 0;
                    std::string finish = "length";
                    for (int i = 0; i < maximum; ++i) {
                        if (i > 0) sampled = choose();
                        const int token = sampled;
                        decode_end = i > 0 ? Clock::now() : decode_start;
                        ++predicted;
                        if (!ignore_eos && stop_ids.count(token)) { finish = "stop"; break; }
                        send({{"event", "token"}, {"token", token}, {"index", i}});
                        if (req.value("interactive", false)) {
                            const auto interactive_started = Clock::now();
                            std::string control;
                            if (!std::getline(std::cin, control)) throw std::runtime_error("request controller disconnected");
                            interactive_wait_ms += ms(interactive_started);
                            ++interactive_waits;
                            if (!json::parse(control).value("continue", false)) { finish = "stop"; break; }
                        }
                        if (i + 1 < maximum) {
                            const int32_t next = token;
                            if (!engine.forward(&next, 1, error) || !engine.logits(0, 1, logits_for_sampling, error)) throw std::runtime_error(error);
                        }
                    }
                    const double predicted_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
                    send({{"event", "done"}, {"finish_reason", finish}, {"timings", {
                        {"grammar_checks", grammar_checks}, {"grammar_rejections", grammar_rejections},
                        {"sampling_ms", sampling_ms}, {"sampling_calls", sampling_calls},
                        {"grammar_wait_ms", grammar_wait_ms}, {"interactive_wait_ms", interactive_wait_ms}, {"interactive_waits", interactive_waits},
                        {"prompt_n", ids.size()}, {"prompt_ms", prompt_ms}, {"prompt_per_second", prompt_ms > 0 ? 1000.0 * ids.size() / prompt_ms : 0},
                        {"predicted_n", predicted}, {"predicted_ms", predicted_ms}, {"predicted_per_second", predicted_ms > 0 ? 1000.0 * std::max(0, predicted - 1) / predicted_ms : 0}}}});
                } catch (const std::exception & e) { engine.reset(); send({{"event", "error"}, {"error", e.what()}}); }
            }
        }
        if (gpu) ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "m31_serve: %s\n", e.what());
        if (gpu) ggml_backend_free(gpu);
        return 1;
    }
}
