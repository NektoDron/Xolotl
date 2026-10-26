// Shared dump contract for the native M31 tool and the separately linked llama.cpp oracle.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31::run {
namespace fs = std::filesystem;

struct Args {
    std::string model, ids, cases, out, backend = "cpu", rows = "all", taps = "none";
    int chunk = 128, context = 4096, threads = 32, decode_tail = 0;
    double arena_gib = 0;
    double gpu_experts_gib = 0;
    int gpu_experts_min_tokens = 32;
    bool reuse_allocations = false, persistent_cpu_pool = false, cpu_residual = false, profile = false;
    bool cpu_decode_cache = false;
    bool dense_decode_cache = false;
    bool oracle_op_offload = false; // separate placement diagnostic; accepted M1 remains false
};
inline Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
        const std::string v = argv[++i];
        auto integer = [&]() {
            size_t end = 0;
            int n = std::stoi(v, &end);
            if (end != v.size() || n <= 0) throw std::runtime_error("invalid positive integer for " + k);
            return n;
        };
        if (k == "-m" || k == "--model") a.model = v;
        else if (k == "--ids-file") a.ids = v;
        else if (k == "--cases") a.cases = v;
        else if (k == "--out") a.out = v;
        else if (k == "--backend") a.backend = v;
        else if (k == "--rows") a.rows = v;
        else if (k == "--taps") a.taps = v;
        else if (k == "--chunk") a.chunk = integer();
        else if (k == "--ctx") a.context = integer();
        else if (k == "--threads") a.threads = integer();
        else if (k == "--decode-tail") a.decode_tail = integer();
        else if (k == "--gpu-experts-min-tokens") a.gpu_experts_min_tokens = integer();
        else if (k == "--oracle-op-offload") {
            if (v != "0" && v != "1") throw std::runtime_error("expected 0|1 for " + k);
            a.oracle_op_offload = v == "1";
        }
        else if (k == "--reuse-allocations" || k == "--persistent-cpu-pool" || k == "--cpu-residual" || k == "--cpu-decode-cache" || k == "--dense-decode-cache" || k == "--profile") {
            if (v != "0" && v != "1") throw std::runtime_error("expected 0|1 for " + k);
            if (k == "--reuse-allocations") a.reuse_allocations = v == "1";
            if (k == "--persistent-cpu-pool") a.persistent_cpu_pool = v == "1";
            if (k == "--cpu-residual") a.cpu_residual = v == "1";
            if (k == "--cpu-decode-cache") a.cpu_decode_cache = v == "1";
            if (k == "--dense-decode-cache") a.dense_decode_cache = v == "1";
            if (k == "--profile") a.profile = v == "1";
        }
        else if (k == "--arena-gib") {
            size_t end = 0;
            a.arena_gib = std::stod(v, &end);
            if (end != v.size() || !std::isfinite(a.arena_gib) || a.arena_gib <= 0 || a.arena_gib > 512)
                throw std::runtime_error("invalid arena budget");
        }
        else if (k == "--gpu-experts-gib") {
            size_t end = 0;
            a.gpu_experts_gib = std::stod(v, &end);
            if (end != v.size() || !std::isfinite(a.gpu_experts_gib) || a.gpu_experts_gib <= 0 || a.gpu_experts_gib > 32)
                throw std::runtime_error("invalid GPU expert budget");
        }
        else throw std::runtime_error("unknown option " + k);
    }
    if (a.model.empty() || a.out.empty() || (a.ids.empty() == a.cases.empty()))
        throw std::runtime_error("usage: -m model.gguf (--ids-file x.i32 | --cases DIR) --out NEW-DIR "
                                 "[--backend cpu|cuda --chunk 128 --ctx 4096 --threads 32 "
                                 "--rows all|last|stride:N --taps none|layers|all]");
    if (a.backend != "cpu" && a.backend != "cuda") throw std::runtime_error("backend must be cpu or cuda");
    if (a.taps != "none" && a.taps != "layers" && a.taps != "all") throw std::runtime_error("invalid taps mode");
    if (a.chunk > 2048 || a.context > 1048576) throw std::runtime_error("chunk/context above correctness-tool cap");
    if (a.backend == "cuda") {
        const char * session = std::getenv("M31_CARD_SESSION");
        if (!session || std::string(session) != "1")
            throw std::runtime_error("CUDA requires the aictl/flock runner (M31_CARD_SESSION=1)");
    }
    return a;
}
inline int chunk_size(const Args & a, int total, int pos) {
    const int prefix = std::max(0, total - a.decode_tail);
    if (a.decode_tail && pos >= prefix) return 1;
    return std::min(a.chunk, (a.decode_tail ? prefix : total) - pos);
}
inline std::string quote(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) {
            char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b;
        } else out += char(c);
    }
    return out + "\"";
}
inline std::vector<fs::path> cases(const Args & a) {
    std::vector<fs::path> result;
    if (!a.ids.empty()) result.emplace_back(a.ids);
    else {
        for (const auto & e : fs::directory_iterator(a.cases))
            if (e.is_regular_file() && e.path().extension() == ".i32") result.push_back(e.path());
        std::sort(result.begin(), result.end());
    }
    if (result.empty()) throw std::runtime_error("no .i32 cases");
    for (auto & path : result) {
        const std::string name = path.stem().string();
        if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
            throw std::runtime_error("case stem must contain only letters, digits, '-' and '_': " + path.string());
    }
    return result;
}
inline std::vector<int32_t> read_ids(const fs::path & path) {
    const auto bytes = fs::file_size(path);
    if (bytes == 0 || bytes % sizeof(int32_t) || bytes > 4 * 1048576ull)
        throw std::runtime_error("invalid .i32 size: " + path.string());
    std::vector<int32_t> result(size_t(bytes / sizeof(int32_t)), 0);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(result.data()), std::streamsize(bytes));
    if (!f) throw std::runtime_error("short .i32 read");
    return result;
}
inline std::vector<int> selected_rows(const std::string & spec, int n) {
    if (spec == "last") return {n - 1};
    int stride = 1;
    if (spec != "all") {
        if (spec.rfind("stride:", 0) != 0) throw std::runtime_error("invalid rows spec");
        size_t end = 0;
        const std::string value = spec.substr(7);
        stride = std::stoi(value, &end);
        if (end != value.size() || stride <= 0) throw std::runtime_error("invalid row stride");
    }
    std::vector<int> result;
    for (int p = 0; p < n; p += stride) result.push_back(p);
    // Keep both sides of block/sparsity boundaries even in a sampled run.
    for (int p : {63, 127, 128, 129, 2047, 2048, 2049, 2999, n - 1}) if (p < n) result.push_back(p);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct Dump {
    fs::path dir;
    const Args & args;
    int vocab;
    std::vector<int> positions;
    std::vector<int> written;
    std::ofstream logits;
    std::ofstream taps;
    Dump(const fs::path & directory, const Args & a, const std::vector<int32_t> & ids, int n_vocab)
        : dir(directory), args(a), vocab(n_vocab), positions(selected_rows(a.rows, int(ids.size()))) {
        if (!fs::create_directory(dir)) throw std::runtime_error("case directory already exists: " + dir.string());
        std::ofstream f(dir / "tokens.i32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(ids.data()), std::streamsize(ids.size() * sizeof(int32_t)));
        if (!f) throw std::runtime_error("cannot write token IDs");
        logits.open(dir / "logits.f32", std::ios::binary);
        if (!logits) throw std::runtime_error("cannot create logits file");
        if (args.taps != "none") {
            fs::create_directory(dir / "taps");
            taps.open(dir / "taps.jsonl");
            if (!taps) throw std::runtime_error("cannot create taps index");
        }
    }
    bool wants(int pos) const { return std::binary_search(positions.begin(), positions.end(), pos); }
    void row(int pos, const float * values) {
        if (!wants(pos)) return;
        logits.write(reinterpret_cast<const char *>(values), std::streamsize(vocab * sizeof(float)));
        if (!logits) throw std::runtime_error("cannot write logits row");
        written.push_back(pos);
    }
    bool wants_tap(const std::string & name) const {
        if (args.taps == "none") return false;
        if (args.taps == "all") return true;
        for (const char * base : {"attn_norm-", "ffn_inp-", "ffn_norm-", "ffn_shexp-", "ffn_moe_out-", "l_out-", "msa_topk-"})
            if (name.rfind(base, 0) == 0) return true;
        return name == "result_norm" || name == "result_output";
    }
    void tensor(const std::string & name, int64_t pos0, const ggml_tensor * t) {
        if (!wants_tap(name)) return;
        if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
            throw std::runtime_error("unsafe tap name");
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32)
            throw std::runtime_error("unsupported tap dtype");
        const std::string file = "taps/" + std::to_string(pos0) + "_" + name + ".bin";
        if (fs::exists(dir / file)) throw std::runtime_error("duplicate tap: " + file);
        std::ofstream f(dir / file, std::ios::binary);
        const size_t row_bytes = size_t(t->ne[0]) * ggml_type_size(t->type);
        std::vector<char> row(row_bytes);
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    ggml_backend_tensor_get(t, row.data(), size_t(i3) * t->nb[3] + size_t(i2) * t->nb[2] + size_t(i1) * t->nb[1], row_bytes);
                    f.write(row.data(), std::streamsize(row_bytes));
                }
        if (!f) throw std::runtime_error("cannot write tap");
        taps << "{\"name\":" << quote(name) << ",\"pos0\":" << pos0 << ",\"type\":" << int(t->type)
             << ",\"shape\":[" << t->ne[0] << ',' << t->ne[1] << ',' << t->ne[2] << ',' << t->ne[3]
             << "],\"file\":" << quote(file) << "}\n";
        taps.flush();
        if (!taps) throw std::runtime_error("cannot write tap index");
    }
    void finish(const std::string & engine, int token_count) {
        if (written != positions) throw std::runtime_error("missing/reordered logits rows");
        logits.flush();
        if (!logits) throw std::runtime_error("cannot flush logits");
        std::ofstream f(dir / "manifest.json");
        f << "{\"format\":\"m31-logits-v1\",\"engine\":" << quote(engine)
          << ",\"model\":" << quote(fs::canonical(args.model).string())
          << ",\"tokens\":" << token_count << ",\"vocab\":" << vocab
          << ",\"chunk\":" << args.chunk << ",\"context\":" << args.context
          << ",\"decode_tail\":" << args.decode_tail << ",\"threads\":" << args.threads
          << ",\"arena_gib\":" << args.arena_gib
          << ",\"gpu_experts_gib\":" << args.gpu_experts_gib
          << ",\"gpu_experts_min_tokens\":" << args.gpu_experts_min_tokens
          << ",\"reuse_allocations\":" << (args.reuse_allocations ? "true" : "false")
          << ",\"persistent_cpu_pool\":" << (args.persistent_cpu_pool ? "true" : "false")
          << ",\"cpu_residual\":" << (args.cpu_residual ? "true" : "false")
          << ",\"cpu_decode_cache\":" << (args.cpu_decode_cache ? "true" : "false")
          << ",\"dense_decode_cache\":" << (args.dense_decode_cache ? "true" : "false")
          << ",\"cuda_mmq_precision\":" << quote(std::getenv("GGML_CUDA_MMQ_PREC") ? std::getenv("GGML_CUDA_MMQ_PREC") : "auto")
          << ",\"gpu_cpu_quants_requested\":" <<
             ((std::getenv("M31_GPU_CPU_QUANTS") && std::string(std::getenv("M31_GPU_CPU_QUANTS")) == "1") ? "true" : "false")
          << ",\"gpu_prefill_async_requested\":" <<
             ((std::getenv("M31_GPU_PREFILL_ASYNC") && std::string(std::getenv("M31_GPU_PREFILL_ASYNC")) == "1") ? "true" : "false")
          << ",\"oracle_op_offload\":" << (args.oracle_op_offload ? "true" : "false")
          << ",\"cuda_fusion_disabled\":" <<
             ((std::getenv("GGML_CUDA_DISABLE_FUSION") && std::string(std::getenv("GGML_CUDA_DISABLE_FUSION")) == "1") ? "true" : "false")
          << ",\"backend\":" << quote(args.backend) << ",\"kv\":\"f16\",\"positions\":[";
        for (size_t i = 0; i < positions.size(); ++i) f << (i ? "," : "") << positions[i];
        f << "]}\n";
        if (!f) throw std::runtime_error("cannot write manifest");
    }
};

} // namespace strata::m31::run
