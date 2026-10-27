// Engine subprocess for the opt-in M31 HTTP adapter. One request at a time.
#include "m31_engine.hpp"
#include "m31_mixed.hpp"
#include "m31_prompt_cache.hpp"
#include "server_prefix_cache.hpp"
#include "server_dspark_policy.hpp"
#include "m31_dspark_session.hpp"
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
        bool prefix_cache_enabled = false;
        std::string dspark_model;
        int dspark_depth = 3;
        float dspark_p_min = 0;
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
            else if (key == "--dspark-model") dspark_model = value;
            else if (key == "--dspark-depth") dspark_depth = std::stoi(value);
            else if (key == "--dspark-p-min") dspark_p_min = std::stof(value);
            else if (key == "--prefix-cache") prefix_cache_enabled = std::stoi(value) != 0;
            else if (key == "--profile") cfg.profile = std::stoi(value) != 0;
            else throw std::runtime_error("unknown option " + key);
        }
        if (model.empty() || cfg.context < 1 || cfg.context > 1048576 || cfg.max_chunk < 1 || cfg.max_chunk > 32768 || cfg.threads < 1)
            throw std::runtime_error("invalid model/context/chunk/threads");
        if (dspark_depth < 1 || dspark_depth > 7 || !std::isfinite(dspark_p_min) || dspark_p_min < 0 || dspark_p_min > 1)
            throw std::runtime_error("invalid DSpark depth/p_min");
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
            prompt_cache::Cache prefix_cache(prompt_cache::environment_budget());
            std::string cache_identity;
            if (prefix_cache.budget()) {
                std::vector<std::string> arguments(argv, argv + argc);
                cache_identity = json({{"arguments", arguments}, {"environment", prompt_cache::engine_environment_identity()},
                                       {"context", cfg.context}, {"max_chunk", cfg.max_chunk}}).dump();
                std::fprintf(stderr, "M31 exact whole-prompt cache: host_budget=%zu bytes; no partial-prefix reuse\n", prefix_cache.budget());
            }
            json ready = {{"event", "ready"}, {"vocab", engine.geometry().vocab}, {"context", cfg.context}};
            if (!dspark_model.empty()) ready["dspark"] = {{"configured", true}, {"mode", "greedy_only"},
                {"sampling_fallback", true}, {"depth", dspark_depth}, {"p_min", dspark_p_min}};
            send(ready);
            ExactPromptCache text_prefix_cache;
            std::string line;
            while (std::getline(std::cin, line)) {
                prompt_cache::Digest active_cache_key{};
                bool active_cache_key_known = false;
                try {
                    const auto request_start = Clock::now();
                    const auto req = json::parse(line);
                    const bool trace = req.value("ttft_trace", false);
                    json trace_ms = json::object();
                    json trace_chunks = json::array();
                    const auto parsed_at = Clock::now();
                    if (trace) trace_ms["json_parse_ms"] = std::chrono::duration<double, std::milli>(parsed_at-request_start).count();
                    auto ids = req.at("tokens").get<std::vector<int32_t>>();
                    const std::string mixed_manifest = req.value("mixed_manifest", "");
                    MixedInput mixed;
                    const auto mixed_start = Clock::now();
                    if (!mixed_manifest.empty()) {
                        mixed = load_mixed_manifest(mixed_manifest, int(engine.geometry().vocab), cfg.context);
                        ids = mixed.tokens;
                    }
                    if (trace) trace_ms["mixed_manifest_read_validate_ms"] = ms(mixed_start);
                    const auto setup_start = Clock::now();
                    const int maximum = req.value("max_tokens", 256);
                    const double temperature = req.value("temperature", 1.0), top_p = req.value("top_p", 0.95), min_p = req.value("min_p", 0.05);
                    const int top_k = req.value("top_k", 40);
                    if (ids.empty() || maximum < 0 || !std::isfinite(temperature) || temperature < 0 ||
                        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || !std::isfinite(min_p) || min_p < 0 || min_p > 1 || top_k < 0)
                        throw std::runtime_error("invalid request/sampling parameters");
                    if (int64_t(ids.size()) > cfg.context || int64_t(ids.size()) + maximum > cfg.context)
                        throw std::runtime_error("prompt plus maximum generation exceeds context");
                    if (mixed_manifest.empty())
                        for (int token : ids) if (token < 0 || token >= engine.geometry().vocab) throw std::runtime_error("token out of vocabulary");
                    const auto stops = req.value("stop_ids", std::vector<int>{});
                    const std::set<int> stop_ids(stops.begin(), stops.end());
                    const bool ignore_eos = req.value("ignore_eos", false);
                    const auto suppressed = req.value("suppress_ids", std::vector<int>{});
                    std::set<int> excluded(suppressed.begin(), suppressed.end());
                    if (ignore_eos) excluded.insert(stop_ids.begin(), stop_ids.end());
                    auto dspark_policy = dspark_request_policy(!dspark_model.empty(), backend == "cuda",
                        cfg.persistent_cpu_pool, cfg.gpu_experts_min_tokens, int(engine.geometry().layers),
                        temperature, req.value("grammar_check", false), maximum);
                    // Mixed target capture/draft seeding has not been validated.
                    if (!mixed_manifest.empty() && dspark_policy.enabled)
                        dspark_policy = {false, "mixed_prompt"};
                    std::unique_ptr<DSparkGreedySession> dspark_session;
                    if (dspark_policy.enabled) {
                        DSparkSessionConfig session_cfg;
                        session_cfg.draft.backend = cfg.backend;
                        session_cfg.draft.sidecar = dspark_model;
                        session_cfg.draft.target = model;
                        session_cfg.depth = dspark_depth;
                        session_cfg.p_min = dspark_p_min;
                        session_cfg.context = cfg.context;
                        dspark_session = std::make_unique<DSparkGreedySession>(session_cfg);
                        if (!dspark_session->begin_prefill(engine, error)) throw std::runtime_error(error);
                    }
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
                    if (trace) trace_ms["request_setup_ms"] = ms(setup_start);
                    const auto prefill_pipeline_start = Clock::now();
                    if ((prefix_cache.budget() || prefix_cache_enabled) && req.contains("cache_prompt") && !req.at("cache_prompt").is_boolean())
                        throw std::runtime_error("cache_prompt must be boolean");
                    const bool cache_requested = prefix_cache.budget() && maximum > 0 && req.value("cache_prompt", true);
                    // The live-KV cache is text-only: visual placeholder IDs do not
                    // identify embeddings. The snapshot cache below hashes all media.
                    const bool use_cache = prefix_cache_enabled && mixed_manifest.empty() && maximum > 0 &&
                        !dspark_policy.enabled && cfg.draft_feature_layers.empty() && req.value("cache_prompt", true);
                    if (!use_cache) text_prefix_cache.clear();
                    // Draft feature rings are not part of a PrefixCheckpoint.
                    bool cache_usable = cache_requested && dspark_model.empty() && cfg.draft_feature_layers.empty();
                    bool cache_hit = false;
                    std::vector<PromptChunk> actual_chunks;
                    const auto padded_keys = [&](int64_t position, int rows) {
                        const int64_t block = std::max<int64_t>(256, engine.geometry().block);
                        return ((position + rows + block - 1) / block) * block;
                    };
                    json cache_info = {{"enabled", cache_requested}, {"hit", false}, {"budget_bytes", prefix_cache.budget()}};
                    if (prefix_cache.budget() && req.contains("cache_prompt") && !req.at("cache_prompt").get<bool>())
                        cache_info["skip"] = "cache_prompt:false";
                    if (cache_requested && !cache_usable) cache_info["skip"] = "draft feature capture is not checkpointed";
                    json cached_recipe = json::array();
                    prompt_cache::Recipe fresh_recipe;
                    bool cache_reset = false;
                    double cache_reset_ms = 0;
                    if (use_cache && !text_prefix_cache.empty()) {
                        const int requested = int(std::min(size_t(cfg.max_chunk), ids.size()));
                        const int planned = engine.prefill_chunk_at(0, requested, error);
                        if (text_prefix_cache.matches(ids, planned, engine.position(), engine.geometry().vocab)) {
                            if (!engine.rewind(int64_t(ids.size()), error)) throw std::runtime_error(error);
                            const auto & chunk = text_prefix_cache.chunk();
                            actual_chunks.push_back(chunk);
                            cached_recipe.push_back({{"position", chunk.position}, {"rows", chunk.tokens}});
                            logits_for_sampling = text_prefix_cache.raw_logits();
                            cache_hit = true;
                            cache_info["skip"] = "live text prefix cache hit";
                        }
                    }
                    auto reset_engine = [&]() {
                        // A reset or snapshot restore invalidates this live-KV descriptor.
                        text_prefix_cache.clear();
                        engine.reset();
                        if (!dspark_model.empty()) {
                            if (!engine.set_external_resident_bytes(0, error) ||
                                !engine.set_draft_feature_layers(dspark_policy.enabled
                                    ? std::vector<int>{4,18,32,46,60} : std::vector<int>{}, 4096, error))
                                throw std::runtime_error(error);
                            const char * capture_reuse = std::getenv("M31_DSPARK_CAPTURE_REUSE");
                            if (dspark_policy.enabled && capture_reuse && std::string(capture_reuse) == "1") {
                                const auto begin = std::max<int64_t>(0, int64_t(ids.size()) - 4096);
                                const char * capture_view = std::getenv("M31_DSPARK_CAPTURE_VIEW");
                                const bool window_ok = capture_view && std::string(capture_view) == "1"
                                    ? engine.set_draft_feature_view_window(begin, int64_t(ids.size()), error)
                                    : engine.set_draft_feature_window(begin, error);
                                if (!window_ok) throw std::runtime_error(error);
                            }
                        }
                    };
                    if (cache_usable && !cache_hit) {
                        const auto cache_reset_start = Clock::now();
                        reset_engine();
                        cache_reset_ms = ms(cache_reset_start);
                        cache_reset = true;
                        const auto geometry_start = Clock::now();
                        const int current_plan = ids.size() <= size_t(cfg.max_chunk)
                            ? engine.prefill_chunk(int(ids.size()), error) : 0;
                        cache_info["geometry_probe_ms"] = ms(geometry_start);
                        cache_info["planned_rows"] = current_plan;
                        if (current_plan != int(ids.size())) {
                            cache_usable = false;
                            cache_info["skip"] = "only a currently admissible single complete prefill is cached";
                        }
                        const auto cache_key_start = Clock::now();
                        try {
                            if (cache_usable) {
                                active_cache_key = prompt_cache::input_key(cache_identity, ids, mixed.visual_rows, mixed.embeddings, mixed.positions);
                                active_cache_key_known = true;
                                cache_info["input_sha256"] = prompt_cache::hex(active_cache_key);
                            }
                        } catch (const std::exception & e) {
                            cache_usable = false;
                            cache_info["error"] = std::string("hash: ") + e.what();
                        }
                        cache_info["key_ms"] = ms(cache_key_start);
                        const auto lookup_start = Clock::now();
                        const auto * entry = cache_usable ? prefix_cache.find(active_cache_key) : nullptr;
                        cache_info["lookup_ms"] = ms(lookup_start);
                        if (entry) {
                            const auto restore_start = Clock::now();
                            if (engine.restore_prefix(entry->checkpoint, error)) {
                                logits_for_sampling = entry->logits;
                                for (auto [position, rows] : entry->recipe) {
                                    cached_recipe.push_back({{"position", position}, {"rows", rows}});
                                    actual_chunks.push_back({position, rows, padded_keys(position, rows)});
                                }
                                cache_info["entry_sha256"] = prompt_cache::hex(entry->key);
                                cache_hit = true;
                            } else {
                                cache_info["error"] = std::string("restore: ") + error;
                                prefix_cache.erase(active_cache_key);
                            }
                            cache_info["restore_ms"] = ms(restore_start);
                        }
                        cache_info["hit"] = cache_hit;
                    }
                    const auto reset_start = Clock::now();
                    if (!cache_hit && !cache_reset) reset_engine();
                    if (trace) trace_ms["reset_ms"] = cache_reset ? cache_reset_ms : ms(reset_start);
                    double plan_ms = 0, forward_ms = 0, first_decode_ms = 0, first_decode_head_ms = 0;
                    const auto start = (cache_requested || cache_hit) ? prefill_pipeline_start : Clock::now();
                    int last_n = 0;
                    for (size_t p = 0; !cache_hit && p < ids.size();) {
                        const int requested = int(std::min(size_t(cfg.max_chunk), ids.size() - p));
                        const auto plan_start = Clock::now();
                        last_n = engine.prefill_chunk(requested, error);
                        if (trace) plan_ms += ms(plan_start);
                        if (last_n <= 0) throw std::runtime_error(error);
                        const auto forward_start = Clock::now();
                        const bool ok = mixed_manifest.empty()
                            ? engine.forward(ids.data() + p, last_n, error)
                            : engine.forward_mixed(ids.data() + p, mixed.visual_rows.data() + p,
                                                   mixed.embeddings.data(), mixed.visual_count, last_n, error);
                        if (!ok) throw std::runtime_error(error);
                        if (dspark_session && !dspark_session->observe_prefill(engine, error)) throw std::runtime_error(error);
                        if (prefix_cache_enabled) actual_chunks.push_back({int64_t(p), last_n, padded_keys(int64_t(p), last_n)});
                        if (trace) {
                            const double elapsed = ms(forward_start);
                            forward_ms += elapsed;
                            trace_chunks.push_back({{"position", p}, {"rows", last_n}, {"tokens", last_n}, {"padded_keys", padded_keys(int64_t(p), last_n)}, {"forward_ms", elapsed}});
                        }
                        if (cache_usable) fresh_recipe.emplace_back(int64_t(p), last_n);
                        p += size_t(last_n);
                    }
                    if (trace) { trace_ms["prefill_plan_ms"] = plan_ms; trace_ms["prefill_forward_ms"] = forward_ms; }
                    const auto head_start = Clock::now();
                    if (maximum && !cache_hit && !engine.logits(last_n - 1, 1, logits_for_sampling, error)) throw std::runtime_error(error);
                    if (trace) trace_ms["prefill_head_ms"] = cache_hit ? 0.0 : ms(head_start);
                    if (use_cache && text_prefix_cache.empty())
                        text_prefix_cache.store(ids, actual_chunks, logits_for_sampling, int(engine.geometry().vocab));
                    if (cache_usable && !cache_hit && fresh_recipe.size() == 1 && fresh_recipe[0].first == 0 && fresh_recipe[0].second == int(ids.size())) {
                        const auto snapshot_start = Clock::now();
                        try {
                            const size_t checkpoint_bytes = engine.prefix_checkpoint_bytes();
                            const size_t total = prompt_cache::Entry::estimated_bytes(checkpoint_bytes, logits_for_sampling.size(), fresh_recipe.size());
                            if (prefix_cache.reserve(total)) {
                                Engine::PrefixCheckpointPtr checkpoint;
                                if (engine.checkpoint_prefix(checkpoint_bytes, checkpoint, error)) {
                                    prompt_cache::Entry entry;
                                    entry.input = active_cache_key;
                                    entry.checkpoint = std::move(checkpoint);
                                    entry.logits = logits_for_sampling;
                                    entry.recipe = fresh_recipe;
                                    cache_info["stored"] = prefix_cache.put(std::move(entry));
                                    cache_info["checkpoint_bytes"] = checkpoint_bytes;
                                } else cache_info["error"] = std::string("snapshot: ") + error;
                            } else cache_info["skip"] = "entry exceeds host cache budget";
                        } catch (const std::exception & e) { cache_info["error"] = std::string("snapshot: ") + e.what(); }
                        cache_info["snapshot_store_ms"] = ms(snapshot_start);
                    }
                    if (cache_requested) {
                        cache_info["resident_bytes"] = prefix_cache.bytes();
                        cache_info["entries"] = prefix_cache.entries();
                        if (cache_info.contains("error")) std::fprintf(stderr, "M31 prompt cache fallback: %s\n", cache_info["error"].get<std::string>().c_str());
                    }
                    // The reference counts the first sample in prefill and measures n_gen-1
                    // generation steps, ending at the final sample rather than token transport.
                    const auto sample_start = Clock::now();
                    int sampled = maximum ? choose() : -1;
                    if (trace) trace_ms["first_sampling_including_grammar_ms"] = ms(sample_start);
                    const auto decode_start = Clock::now();
                    auto decode_end = decode_start;
                    const double prompt_ms = std::chrono::duration<double, std::milli>(decode_start - start).count();
                    json prefill = {{"event", "prefill"}, {"prompt_n", ids.size()}, {"prompt_ms", prompt_ms}};
                    if (prefix_cache_enabled) {
                        prefill["cached_n"] = cache_hit ? ids.size() : 0;
                        prefill["prompt_eval_n"] = cache_hit ? 0 : ids.size();
                        prefill["cache_hit"] = cache_hit;
                        prefill["actual_chunks"] = json::array();
                        for (const auto & chunk : actual_chunks) prefill["actual_chunks"].push_back({
                            {"position", chunk.position}, {"tokens", chunk.tokens}, {"padded_keys", chunk.padded_keys}});
                        if (cache_hit) prefill["cached_prefill_recipe"] = cached_recipe;
                    }
                    if (trace) {
                        trace_ms["prefill_ready_from_request_ms"] = ms(request_start);
                        prefill["ttft_stages_ms"] = trace_ms;
                        prefill["actual_chunks"] = trace_chunks;
                    }
                    if (prefix_cache.budget()) {
                        prefill["prompt_cache"] = cache_info;
                        if (cache_hit) prefill["cached_prefill_recipe"] = cached_recipe;
                    }
                    send(prefill);
                    int predicted = 0;
                    std::string finish = "length";
                    for (int i = 0; i < maximum; ++i) {
                        if (i > 0) sampled = choose();
                        const int token = sampled;
                        decode_end = i > 0 ? Clock::now() : decode_start;
                        ++predicted;
                        if (!ignore_eos && stop_ids.count(token)) { finish = "stop"; break; }
                        if (trace && i == 0) trace_ms["first_token_sent_from_request_ms"] = ms(request_start);
                        send({{"event", "token"}, {"token", token}, {"index", i}});
                        if (req.value("interactive", false)) {
                            const auto interactive_started = Clock::now();
                            std::string control;
                            if (!std::getline(std::cin, control)) throw std::runtime_error("request controller disconnected");
                            interactive_wait_ms += ms(interactive_started);
                            ++interactive_waits;
                            if (!json::parse(control).value("continue", false)) { text_prefix_cache.clear(); finish = "stop"; break; }
                        }
                        if (i + 1 < maximum) {
                            const int32_t next = token;
                            const auto decode_forward_start = Clock::now();
                            if (dspark_session) {
                                // Draft preparation remains after token0 and its ACK.
                                if (!dspark_session->advance(engine, next, maximum - i - 1, logits_for_sampling, error))
                                    throw std::runtime_error(error);
                                if (trace && i == 0) first_decode_ms = ms(decode_forward_start);
                            } else {
                                if (!engine.forward(&next, 1, error)) throw std::runtime_error(error);
                                if (trace && i == 0) first_decode_ms = ms(decode_forward_start);
                                const auto decode_head_start = Clock::now();
                                if (!engine.logits(0, 1, logits_for_sampling, error)) throw std::runtime_error(error);
                                if (trace && i == 0) first_decode_head_ms = ms(decode_head_start);
                            }
                        }
                    }
                    const double predicted_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
                    double dspark_finish_ms = 0;
                    if (dspark_session) {
                        const auto finish_started = Clock::now();
                        if (!dspark_session->finish(engine, error)) throw std::runtime_error(error);
                        dspark_finish_ms = ms(finish_started);
                        const int64_t expected_fed = int64_t(ids.size()) + std::max(0, predicted - 1);
                        if (engine.position() != expected_fed) throw std::runtime_error("DSpark final KV position differs from emitted samples");
                    }
                    json done = {{"event", "done"}, {"finish_reason", finish}, {"timings", {
                        {"grammar_checks", grammar_checks}, {"grammar_rejections", grammar_rejections},
                        {"sampling_ms", sampling_ms}, {"sampling_calls", sampling_calls},
                        {"grammar_wait_ms", grammar_wait_ms}, {"interactive_wait_ms", interactive_wait_ms}, {"interactive_waits", interactive_waits},
                        {"prompt_n", ids.size()}, {"prompt_ms", prompt_ms}, {"prompt_per_second", prompt_ms > 0 ? 1000.0 * ids.size() / prompt_ms : 0},
                        {"predicted_n", predicted}, {"predicted_ms", predicted_ms}, {"predicted_per_second", predicted_ms > 0 ? 1000.0 * std::max(0, predicted - 1) / predicted_ms : 0}}}};
                    if (trace) {
                        trace_ms["first_decode_forward_ms"] = first_decode_ms;
                        trace_ms["first_decode_head_ms"] = first_decode_head_ms;
                        trace_ms["request_total_ms"] = ms(request_start);
                        done["timings"]["ttft_stages_ms"] = trace_ms;
                        done["timings"]["actual_chunks"] = trace_chunks;
                    }
                    if (prefix_cache.budget()) {
                        done["timings"]["prompt_cache"] = cache_info;
                        if (cache_hit) done["timings"]["cached_prefill_recipe"] = cached_recipe;
                    }
                    if (!dspark_model.empty()) {
                        auto & timings = done["timings"];
                        timings["dspark_mode"] = dspark_policy.enabled ? "greedy" : "target_fallback";
                        timings["dspark_fallback_reason"] = dspark_policy.reason;
                        timings["dspark_finish_ms"] = dspark_finish_ms;
                        if (dspark_session) {
                            const auto & stats = dspark_session->stats();
                            timings["dspark_rounds"] = stats.rounds;
                            timings["dspark_proposed"] = stats.proposed;
                            timings["dspark_accepted"] = stats.accepted;
                            timings["dspark_verified_rows"] = stats.verified_rows;
                            timings["dspark_cached_rows"] = stats.cached_rows;
                            timings["dspark_prepare_ms"] = stats.prepare_ms;
                            timings["dspark_draft_ms"] = stats.draft_ms;
                            timings["dspark_verify_ms"] = stats.verify_ms;
                            timings["dspark_append_ms"] = stats.append_ms;
                        }
                    }
                    if (prefix_cache_enabled || !dspark_model.empty()) {
                        done["kv_position"] = engine.position();
                        done["fed_generated_n"] = engine.position() - int64_t(ids.size());
                    }
                    send(done);
                } catch (const std::exception & e) {
                    text_prefix_cache.clear();
                    if (active_cache_key_known) prefix_cache.erase(active_cache_key);
                    engine.reset(); send({{"event", "error"}, {"error", e.what()}});
                }
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
