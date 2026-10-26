#include "m31_run.hpp"
#include "llama.h"
#include "ggml-cpu.h"

#include <climits>
#include <iterator>
#include <memory>
#include <set>

namespace {
using namespace strata::m31;

struct Capture {
    run::Dump * dump = nullptr;
    int64_t pos0 = 0;
    std::string error;
};
bool capture_name(const std::string & name) {
    const auto dash = name.rfind('-');
    const std::string base = dash == std::string::npos ? name : name.substr(0, dash);
    static const std::set<std::string> names = {
        "attn_norm", "Qcur_normed", "Kcur_normed", "Qcur", "Kcur", "Vcur",
        "msa_bs", "msa_topk", "msa_block_mask", "kqv_out", "ffn_inp", "ffn_norm",
        "ffn_shexp", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_down_scaled",
        "ffn_moe_topk", "ffn_moe_out", "ffn_out", "l_out", "result_norm", "result_output"
    };
    return names.count(base) != 0;
}
bool callback(ggml_tensor * t, bool ask, void * data) {
    auto & c = *static_cast<Capture *>(data);
    const bool wanted = c.dump && capture_name(t->name) && c.dump->wants_tap(t->name);
    if (ask) return wanted;
    if (!wanted) return true;
    try { c.dump->tensor(t->name, c.pos0, t); return true; }
    catch (const std::exception & e) { c.error = e.what(); return false; }
}

struct ModelDeleter { void operator()(llama_model * m) const { llama_model_free(m); } };
struct ContextDeleter { void operator()(llama_context * c) const { llama_free(c); } };
using Model = std::unique_ptr<llama_model, ModelDeleter>;
using Context = std::unique_ptr<llama_context, ContextDeleter>;
struct Batch {
    llama_batch b;
    explicit Batch(int n) : b(llama_batch_init(n, 0, 1)) {}
    ~Batch() { llama_batch_free(b); }
};
struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

int tokenize(int argc, char ** argv) {
    if (argc != 5) throw std::runtime_error("usage: m31_oracle tokenize MODEL.gguf TEXT-DIR NEW-ID-DIR");
    // Vocab-only operation must not probe a visible GPU.
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    ggml_backend_load_all();
    Backend backend;
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    Model model(llama_model_load_from_file(argv[2], mp));
    if (!model) throw std::runtime_error("cannot load vocabulary");
    const auto * vocab = llama_model_get_vocab(model.get());
    if (!run::fs::create_directory(argv[4])) throw std::runtime_error("ID output directory already exists");
    std::vector<run::fs::path> texts;
    for (const auto & e : run::fs::directory_iterator(argv[3]))
        if (e.is_regular_file() && e.path().extension() == ".txt") texts.push_back(e.path());
    std::sort(texts.begin(), texts.end());
    if (texts.empty()) throw std::runtime_error("no .txt prompts");
    for (const auto & path : texts) {
        std::ifstream f(path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (text.empty() || text.size() > INT_MAX) throw std::runtime_error("invalid prompt size");
        int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, true);
        if (n >= 0) throw std::runtime_error("expected a negative required token count");
        std::vector<int32_t> ids(size_t(-n), 0);
        n = llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), int(ids.size()), true, true);
        if (n <= 0) throw std::runtime_error("tokenization failed");
        ids.resize(size_t(n));
        std::ifstream length(path.parent_path() / (path.stem().string() + ".length"));
        int wanted = 0;
        if (length >> wanted) {
            if (wanted <= 0 || wanted > n) throw std::runtime_error("prompt too short for requested .length");
            ids.resize(size_t(wanted));
        }
        std::ofstream out(run::fs::path(argv[4]) / (path.stem().string() + ".i32"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(ids.data()), std::streamsize(ids.size() * sizeof(int32_t)));
        if (!out) throw std::runtime_error("cannot write token IDs");
        std::fprintf(stderr, "tokenized %s: %zu IDs\n", path.stem().string().c_str(), ids.size());
    }
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "tokenize") return tokenize(argc, argv);
        const auto args = run::parse(argc, argv);
        const auto cases = run::cases(args);
        if (args.backend == "cpu") setenv("CUDA_VISIBLE_DEVICES", "", 1);
        if (!run::fs::create_directory(args.out)) throw std::runtime_error("output directory exists; use a new path");
        ggml_backend_load_all();
        Backend backend;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = args.backend == "cuda" ? 999 : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps\\..*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
        };
        if (args.backend == "cuda") mp.tensor_buft_overrides = overrides;
        Model model(llama_model_load_from_file(args.model.c_str(), mp));
        if (!model) throw std::runtime_error("cannot load PR model");
        Capture capture;
        auto cp = llama_context_default_params();
        cp.n_ctx = args.context;
        cp.n_batch = args.chunk;
        cp.n_ubatch = args.chunk;
        cp.n_seq_max = 1;
        cp.n_threads = args.threads;
        cp.n_threads_batch = args.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = args.backend == "cuda";
        cp.op_offload = args.backend == "cuda";
        cp.kv_unified = false;
        cp.no_perf = true;
        if (args.taps != "none") { cp.cb_eval = callback; cp.cb_eval_user_data = &capture; }
        Context ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) throw std::runtime_error("cannot initialize PR context");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        Batch batch(args.chunk);
        for (const auto & path : cases) {
            const auto ids = run::read_ids(path);
            if (ids.size() > size_t(args.context)) throw std::runtime_error("case exceeds context");
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            run::Dump dump(run::fs::path(args.out) / path.stem(), args, ids, vocab);
            capture.dump = &dump;
            for (int pos = 0; pos < int(ids.size()); pos += args.chunk) {
                const int n = std::min(args.chunk, int(ids.size()) - pos);
                batch.b.n_tokens = n;
                for (int r = 0; r < n; ++r) {
                    batch.b.token[r] = ids[size_t(pos + r)];
                    batch.b.pos[r] = pos + r;
                    batch.b.n_seq_id[r] = 1;
                    batch.b.seq_id[r][0] = 0;
                    // All outputs keep the last-layer taps aligned with every input row.
                    batch.b.logits[r] = true;
                }
                capture.pos0 = pos;
                if (llama_decode(ctx.get(), batch.b) != 0) throw std::runtime_error("PR llama_decode failed");
                if (!capture.error.empty()) throw std::runtime_error(capture.error);
                for (int r = 0; r < n; ++r) if (dump.wants(pos + r)) {
                    const float * logits = llama_get_logits_ith(ctx.get(), r);
                    if (!logits) throw std::runtime_error("missing oracle logits");
                    dump.row(pos + r, logits);
                }
            }
            dump.finish("llama.cpp-PR31192", int(ids.size()));
            capture.dump = nullptr;
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "m31_oracle: %s\n", e.what());
        return 1;
    }
}
