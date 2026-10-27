// Isolated mtmd encoder subprocess (CPU by default, CUDA opt-in). It never creates a llama_context or runs the LLM.
// Link only to the audited PR archives; do not mix its ggml ABI with Xolotl.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "json.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
template<class T> static void save(const fs::path & p, const T * data, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), n * sizeof(T));
    if (!f) throw std::runtime_error("cannot write " + p.string());
}
struct Media {
    mtmd_helper_bitmap_wrapper value{};
    ~Media() {
        if (value.bitmap) mtmd_bitmap_free(value.bitmap);
        if (value.video_ctx) mtmd_helper_video_free(value.video_ctx);
    }
};
int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "usage: vision_adapter SPEC.json NEW_OUTPUT_DIR | --worker CONFIG.json\n"; return 2; }
    const bool worker = std::string(argv[1]) == "--worker";
    try {
        std::ifstream input(worker ? argv[2] : argv[1]);
        json config; input >> config;
        const std::string backend = config.value("backend", std::string("cpu"));
        if (backend != "cpu" && backend != "cuda") throw std::runtime_error("backend must be cpu|cuda");
        if (backend == "cpu") setenv("CUDA_VISIBLE_DEVICES", "", 1);
        else {
            const char * session = std::getenv("M31_CARD_SESSION");
            const char * devices = std::getenv("CUDA_VISIBLE_DEVICES");
            if (!session || std::string(session) != "1" || !devices || std::string(devices) != "0")
                throw std::runtime_error("CUDA encoder requires aictl card-0 session (M31_CARD_SESSION=1, CUDA_VISIBLE_DEVICES=0)");
        }
        json spec = config;
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.vocab_only = true; mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(spec.at("model").get<std::string>().c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("vocabulary loading failed");
        auto params = mtmd_context_params_default();
        params.use_gpu = backend == "cuda"; params.device = nullptr; params.warmup = false;
        std::string backend_device = "CPU";
        if (params.use_gpu) {
            params.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (!params.device) throw std::runtime_error("requested CUDA encoder device is unavailable");
            backend_device = ggml_backend_dev_name(params.device);
            if (backend_device.rfind("CUDA", 0) != 0) throw std::runtime_error("requested CUDA encoder selected a non-CUDA device");
            // An explicit device makes clip initialization throw if allocation or
            // initialization fails, instead of silently falling back to CPU.
        }
        params.print_timings = false; params.n_threads = spec.value("threads", 8);
        params.batch_max_tokens = spec.value("encoder_batch_tokens", 4096);
        if (params.n_threads < 1 || params.batch_max_tokens < 1) throw std::runtime_error("invalid thread/batch budget");
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> ctx(
            mtmd_init_from_file(spec.at("mmproj").get<std::string>().c_str(), model.get(), params), mtmd_free);
        if (!ctx || !mtmd_support_vision(ctx.get()) || mtmd_decode_use_mrope(ctx.get()))
            throw std::runtime_error("requires ordinary-position MiniMax vision encoder");
        const int cpu_workers = backend == "cpu" ? config.value("cpu_workers", 1) : 1;
        if (cpu_workers < 1 || cpu_workers > 16) throw std::runtime_error("cpu_workers must be 1..16");
        std::vector<std::unique_ptr<mtmd_context, decltype(&mtmd_free)>> contexts;
        const size_t cache_limit = size_t(config.value("embedding_cache_mb", 256)) * 1024 * 1024;
        size_t cache_bytes = 0;
        std::unordered_map<std::string, std::vector<float>> embedding_cache;
        std::deque<std::string> cache_order;
        auto process = [&](const json & spec, const fs::path & out) {
        if (!fs::create_directory(out)) throw std::runtime_error("output exists; use a new path");
        std::string prompt;
        if (spec.contains("prompt")) prompt = spec.at("prompt").get<std::string>();
        else {
            auto templates = common_chat_templates_init(model.get(), "");
            common_chat_templates_inputs ci;
            ci.messages = common_chat_msgs_parse_oaicompat(common_json::parse(spec.at("messages").dump()));
            ci.enable_thinking = spec.value("enable_thinking", true);
            ci.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            ci.add_generation_prompt = true;
            if (spec.contains("chat_template_kwargs"))
                for (auto it=spec["chat_template_kwargs"].begin(); it!=spec["chat_template_kwargs"].end(); ++it)
                    ci.chat_template_kwargs[it.key()] = it.value().dump();
            prompt = common_chat_templates_apply(templates.get(), ci).prompt.str();
        }
        std::ofstream(out/"prompt.txt") << prompt;
        auto opt = mtmd_helper_init_opt_default();
        opt.video_params.fps_target = spec.value("video_fps", 1.0f);
        opt.video_params.timestamp_interval_ms = spec.value("timestamp_interval_ms", int64_t(5000));
        std::vector<std::unique_ptr<Media>> owners;
        std::vector<const mtmd_bitmap *> bitmaps;
        json media_meta = json::array();
        for (const auto & path : spec.at("media")) {
            auto media = std::make_unique<Media>();
            const auto name = path.get<std::string>();
            media->value = mtmd_helper_bitmap_init_from_file(ctx.get(), name.c_str(), false, opt);
            if (!media->value.bitmap) throw std::runtime_error("media load failed: " + name);
            json item = {{"path", name}};
            if (media->value.video_ctx) {
                auto info = mtmd_helper_video_get_info(media->value.video_ctx);
                item["video"] = {{"width", info.width}, {"height", info.height}, {"fps", info.fps}, {"estimated_frames", info.n_frames}};
            } else item["image"] = {{"width", mtmd_bitmap_get_nx(media->value.bitmap)}, {"height", mtmd_bitmap_get_ny(media->value.bitmap)}};
            media_meta.push_back(item);
            bitmaps.push_back(media->value.bitmap);
            owners.push_back(std::move(media));
        }
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        mtmd_input_text text{prompt.data(), prompt.size(), spec.value("add_special", false), true};
        if (mtmd_tokenize(ctx.get(), chunks.get(), &text, bitmaps.data(), bitmaps.size()))
            throw std::runtime_error("mtmd_tokenize failed");
        // vocab_only deliberately skips model hyperparameters; n_embd_inp() is
        // zero in that mode. Read the audited GGUF scalar without loading weights.
        char embd_meta[64]{};
        if (llama_model_meta_val_str(model.get(), "minimax-m3.embedding_length", embd_meta, sizeof(embd_meta)) <= 0)
            throw std::runtime_error("missing embedding length metadata");
        const int embd = std::stoi(embd_meta);
        if (embd != 6144) throw std::runtime_error("unexpected MiniMax embedding dimension");
        const bool encode = spec.value("encode", false);
        const std::string encoder_mode = spec.value("encoder_mode", std::string("batch"));
        if (encoder_mode != "batch" && encoder_mode != "single") throw std::runtime_error("encoder_mode must be batch|single");
        json manifest = {{"format", "m31-vision-ipc-v1"}, {"source_pr", "6b95a9908d2c7b242817a447708aec38e937e2c0"},
            {"backend", backend}, {"backend_device", backend_device}, {"llm_inference", false}, {"encoded", encode}, {"n_embd", embd},
            {"encoder_mode", encoder_mode}, {"encoder_threads", params.n_threads}, {"cpu_workers", cpu_workers},
            {"media", media_meta}, {"video_fps", opt.video_params.fps_target}, {"timestamp_interval_ms", opt.video_params.timestamp_interval_ms},
            {"chunks", json::array()}, {"encoder_batches", json::array()}};
        size_t row = 0, visual_row = 0;
        std::vector<const mtmd_input_chunk *> images;
        std::vector<size_t> image_indices;
        for (size_t i=0; i<mtmd_input_chunks_size(chunks.get()); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
            const auto n = mtmd_input_chunk_get_n_tokens(chunk);
            const auto npos = mtmd_input_chunk_get_n_pos(chunk);
            if (size_t(npos) != n) throw std::runtime_error("unexpected non-linear decoder positions");
            json part = {{"index", i}, {"row", row}, {"rows", n}, {"position", row}, {"n_pos", npos}};
            const std::string stem = "chunk-" + std::to_string(i);
            size_t serialized_size = 0;
            if (mtmd_input_chunk_save(chunk, nullptr, 0, &serialized_size)) throw std::runtime_error("cannot size serialized chunk");
            std::vector<char> serialized(serialized_size);
            if (mtmd_input_chunk_save(chunk, serialized.data(), serialized.size(), &serialized_size)) throw std::runtime_error("cannot serialize chunk");
            save(out/(stem+".mtmd"), serialized.data(), serialized.size());
            part["metadata"] = stem+".mtmd";
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t count = 0;
                const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &count);
                part["type"] = "text"; part["tokens"] = std::vector<llama_token>(ids, ids+count);
            } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                part["type"] = "image"; part["visual_row"] = visual_row;
                const auto * id = mtmd_input_chunk_get_id(chunk);
                part["id"] = id ? id : "";
                if (!id || !*id) throw std::runtime_error("missing media content ID for encoder cache");
                const auto * image = mtmd_input_chunk_get_tokens_image(chunk);
                part["decoder_positions"] = json::array();
                for (size_t t=0; t<n; ++t) {
                    const auto p = mtmd_image_tokens_get_decoder_pos(image, row, t);
                    if (p.t != row+t || p.x != row+t || p.y != row+t || p.z != row+t)
                        throw std::runtime_error("unexpected decoder position layout");
                }
                part["decoder_positions"] = {{"start", row}, {"step", 1}, {"axes", 1}};
                if (encode) part["embeddings"] = stem+".f32";
                images.push_back(chunk); image_indices.push_back(i);
                visual_row += n;
            } else throw std::runtime_error("audio is not supported by this adapter");
            manifest["chunks"].push_back(part);
            row += n;
        }
        size_t cached_images = 0;
        if (encode && cache_limit) {
            std::vector<const mtmd_input_chunk *> pending;
            std::vector<size_t> pending_indices;
            for (size_t j=0; j<images.size(); ++j) {
                const auto & part = static_cast<const json &>(manifest)["chunks"][image_indices[j]];
                const auto key = part["id"].get<std::string>() + ":" + std::to_string(part["rows"].get<size_t>()) + ":" + std::to_string(opt.video_params.fps_target);
                auto found = embedding_cache.find(key);
                if (found != embedding_cache.end()) {
                    save(out/part["embeddings"].get<std::string>(), found->second.data(), found->second.size());
                    ++cached_images;
                } else { pending.push_back(images[j]); pending_indices.push_back(image_indices[j]); }
            }
            images.swap(pending); image_indices.swap(pending_indices);
        }
        std::vector<std::vector<float>> cache_updates(images.size());
        const auto init_start = std::chrono::steady_clock::now();
        const int active_workers = encode ? std::min(cpu_workers, int(images.size())) : 0;
        for (int i=int(contexts.size())+1; i<active_workers; ++i) {
            std::unique_ptr<mtmd_context, decltype(&mtmd_free)> extra(
                mtmd_init_from_file(config.at("mmproj").get<std::string>().c_str(), model.get(), params), mtmd_free);
            if (!extra) throw std::runtime_error("parallel encoder initialization failed");
            contexts.push_back(std::move(extra));
        }
        manifest["worker_init_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-init_start).count();
        manifest["active_workers"] = active_workers;
        const auto encode_start = std::chrono::steady_clock::now();
        auto save_embedding = [&](size_t j, const float * values) {
            const auto count = mtmd_input_chunk_get_n_tokens(images[j]) * size_t(embd);
            if (!values) throw std::runtime_error("missing encoder output");
            for (size_t k=0; k<count; ++k) if (!std::isfinite(values[k])) throw std::runtime_error("nonfinite embedding");
            const auto name = static_cast<const json &>(manifest)["chunks"][image_indices[j]]["embeddings"].get<std::string>();
            save(out/name, values, count);
            if (count*sizeof(float) <= cache_limit) cache_updates[j].assign(values, values+count);
        };
        if (encode && cpu_workers > 1 && images.size() > 1) {
            // Each worker owns an independent mtmd context/scheduler. The source
            // chunks are immutable; output files have disjoint names. Preserve
            // original row order in the manifest regardless of completion order.
            std::atomic<size_t> next{0};
            std::exception_ptr failure;
            std::mutex failure_mutex;
            std::vector<std::thread> threads;
            for (int k=0; k<active_workers; ++k) threads.emplace_back([&, k] {
                try {
                    auto * local = k == 0 ? ctx.get() : contexts[k-1].get();
                    for (;;) {
                        const size_t j = next.fetch_add(1);
                        if (j >= images.size()) break;
                        if (mtmd_encode_chunk(local, images[j])) throw std::runtime_error("parallel encoder computation failed");
                        save_embedding(j, mtmd_get_output_embd(local));
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!failure) failure = std::current_exception();
                    next.store(images.size());
                }
            });
            for (auto & thread : threads) thread.join();
            if (failure) std::rethrow_exception(failure);
            for (auto i : image_indices) manifest["encoder_batches"].push_back(json::array({i}));
        } else if (encode && encoder_mode == "single") {
            for (size_t j=0; j<images.size(); ++j) {
                if (mtmd_encode_chunk(ctx.get(), images[j])) throw std::runtime_error("single-chunk encoder computation failed");
                save_embedding(j, mtmd_get_output_embd(ctx.get()));
                manifest["encoder_batches"].push_back(json::array({image_indices[j]}));
            }
        } else if (encode) {
            for (size_t first=0; first<images.size();) {
                std::unique_ptr<mtmd_batch, decltype(&mtmd_batch_free)> batch(mtmd_batch_init(ctx.get()), mtmd_batch_free);
                size_t end = first;
                while (end<images.size()) {
                    const int rc = mtmd_batch_add_chunk(batch.get(), images[end]);
                    if (rc == 0) { ++end; continue; }
                    if (end == first || rc == 1) throw std::runtime_error("cannot batch image");
                    break;
                }
                if (mtmd_batch_encode(batch.get())) throw std::runtime_error("encoder computation failed");
                json group = json::array();
                for (size_t j=first; j<end; ++j) {
                    save_embedding(j, mtmd_batch_get_output_embd(batch.get(), images[j]));
                    group.push_back(image_indices[j]);
                }
                manifest["encoder_batches"].push_back(group); first=end;
            }
        }
        for (size_t j=0; j<cache_updates.size(); ++j) {
            auto & value = cache_updates[j];
            if (value.empty()) continue;
            const auto & part = static_cast<const json &>(manifest)["chunks"][image_indices[j]];
            const auto key = part["id"].get<std::string>() + ":" + std::to_string(part["rows"].get<size_t>()) + ":" + std::to_string(opt.video_params.fps_target);
            if (embedding_cache.count(key)) continue;
            while (cache_bytes + value.size()*sizeof(float) > cache_limit && !cache_order.empty()) {
                const auto old = cache_order.front(); cache_order.pop_front();
                cache_bytes -= embedding_cache.at(old).size()*sizeof(float); embedding_cache.erase(old);
            }
            cache_bytes += value.size()*sizeof(float); cache_order.push_back(key);
            embedding_cache.emplace(key, std::move(value));
        }
        manifest["cached_image_chunks"] = cached_images;
        manifest["encode_seconds"] = encode ? std::chrono::duration<double>(std::chrono::steady_clock::now()-encode_start).count() : 0;
        manifest["rows"] = row; manifest["visual_rows"] = visual_row;
        std::ofstream result(out/"manifest.json"); result << manifest.dump(2) << '\n';
        if (!result) throw std::runtime_error("cannot save manifest");
        return json({{"ok", true}, {"manifest", (out/"manifest.json").string()}, {"rows", row}, {"visual_rows", visual_row}, {"image_chunks", images.size()+cached_images}, {"encoded", encode}});
        };
        if (!worker) {
            std::cout << process(config, fs::absolute(argv[2])).dump() << '\n';
        } else {
            std::cout << json({{"ready", true}, {"backend", backend}, {"cpu_workers", cpu_workers}}).dump() << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                try {
                    const auto request = json::parse(line);
                    json merged = config;
                    // Model/backend/context settings are fixed for the worker lifetime.
                    for (const char * key : {"prompt", "messages", "media", "video_fps", "timestamp_interval_ms", "encode", "add_special", "enable_thinking", "chat_template_kwargs", "encoder_mode"})
                        if (request.contains(key)) merged[key] = request[key];
                    std::cout << process(merged, fs::absolute(request.at("out").get<std::string>())).dump() << std::endl;
                } catch (const std::exception & e) {
                    std::cout << json({{"ok", false}, {"error", e.what()}}).dump() << std::endl;
                }
            }
        }
        return 0;
    } catch (const std::exception & e) { std::cerr << "vision_adapter: " << e.what() << '\n'; return 1; }
}
