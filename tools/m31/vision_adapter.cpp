// CPU-only mtmd encoder subprocess. It never creates a llama_context or runs the LLM.
// Link only to the audited PR archives; do not mix its ggml ABI with Xolotl.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "json.hpp"
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
    if (argc != 3) { std::cerr << "usage: vision_adapter SPEC.json NEW_OUTPUT_DIR\n"; return 2; }
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    try {
        std::ifstream input(argv[1]);
        json spec; input >> spec;
        const fs::path out = fs::absolute(argv[2]);
        if (!fs::create_directory(out)) throw std::runtime_error("output exists; use a new path");
        auto mp = llama_model_default_params();
        mp.vocab_only = true; mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(spec.at("model").get<std::string>().c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("vocabulary loading failed");
        auto params = mtmd_context_params_default();
        params.use_gpu = false; params.device = nullptr; params.warmup = false;
        params.print_timings = false; params.n_threads = spec.value("threads", 8);
        params.batch_max_tokens = spec.value("encoder_batch_tokens", 4096);
        if (params.n_threads < 1 || params.batch_max_tokens < 1) throw std::runtime_error("invalid thread/batch budget");
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> ctx(
            mtmd_init_from_file(spec.at("mmproj").get<std::string>().c_str(), model.get(), params), mtmd_free);
        if (!ctx || !mtmd_support_vision(ctx.get()) || mtmd_decode_use_mrope(ctx.get()))
            throw std::runtime_error("requires ordinary-position MiniMax vision encoder");
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
            {"backend", "cpu"}, {"llm_inference", false}, {"encoded", encode}, {"n_embd", embd},
            {"encoder_mode", encoder_mode}, {"encoder_threads", params.n_threads},
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
        const auto encode_start = std::chrono::steady_clock::now();
        auto save_embedding = [&](size_t j, const float * values) {
            const auto count = mtmd_input_chunk_get_n_tokens(images[j]) * size_t(embd);
            if (!values) throw std::runtime_error("missing encoder output");
            for (size_t k=0; k<count; ++k) if (!std::isfinite(values[k])) throw std::runtime_error("nonfinite embedding");
            const auto name = manifest["chunks"][image_indices[j]]["embeddings"].get<std::string>();
            save(out/name, values, count);
        };
        if (encode && encoder_mode == "single") {
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
        manifest["encode_seconds"] = encode ? std::chrono::duration<double>(std::chrono::steady_clock::now()-encode_start).count() : 0;
        manifest["rows"] = row; manifest["visual_rows"] = visual_row;
        std::ofstream result(out/"manifest.json"); result << manifest.dump(2) << '\n';
        if (!result) throw std::runtime_error("cannot save manifest");
        std::cout << json({{"rows", row}, {"visual_rows", visual_row}, {"image_chunks", images.size()}, {"encoded", encode}}).dump() << '\n';
        return 0;
    } catch (const std::exception & e) { std::cerr << "vision_adapter: " << e.what() << '\n'; return 1; }
}
