// Standalone M2 diagnostic: ORIGINAL single-expert bytes, no 153-GiB arena/cache.
// Compares dense and indirect CPU/GPU primitives, then a complete OAI FFN.
// CUDA launches are allowed only by the aictl/flock job controller.
#include "m31_geometry.hpp"
#include "m31_run.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "shortprefill_cuda_math_cpu.hpp"
#include "shortprefill_cuda_nvfp4_cpu.hpp"
#ifdef M31_CPU_QUANTS_CUDA
#include "gpu_cpu_quants.hpp"
#include "cpu_gpu_host_ops.hpp"
#endif
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <array>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace strata::m31;
namespace fs = std::filesystem;

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("m31_expert_check: " + message);
}
struct Args {
    std::string model, out, backend = "cuda", source_precision = "both", accumulator = "auto";
    std::vector<int> layers{3, 10, 52}, tokens{1, 7, 128};
    int expert = 0, threads = 64;
    uint64_t seed = 20261026;
    float input_scale = 1;
    bool raw = false;
    bool short_cpu_check = false;
    std::string short_cpu_math = "both";
};
Args parse(int argc, char ** argv) {
    Args a;
    bool tokens_given=false,source_given=false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        check(i + 1 < argc, "missing value for " + key);
        const std::string value = argv[++i];
        auto integer = [&](int minimum) {
            size_t end = 0;
            const int number = std::stoi(value, &end);
            check(end == value.size() && number >= minimum, "invalid integer for " + key);
            return number;
        };
        if (key == "-m" || key == "--model") a.model = value;
        else if (key == "--out") a.out = value;
        else if (key == "--backend") a.backend = value;
        else if (key == "--layer") a.layers = {integer(0)};
        else if (key == "--tokens") { a.tokens = {integer(1)};tokens_given=true; }
        else if (key == "--expert") a.expert = integer(0);
        else if (key == "--threads") a.threads = integer(1);
        else if (key == "--seed") {
            size_t end = 0;
            a.seed = std::stoull(value, &end);
            check(end == value.size() && a.seed != 0, "seed must be a positive integer");
        } else if (key == "--raw") {
            check(value == "0" || value == "1", "--raw expects 0|1");
            a.raw = value == "1";
        } else if (key == "--src-precision") { a.source_precision = value;source_given=true; }
        else if (key == "--acc-precision") a.accumulator = value;
        else if (key == "--short-cpu-check") {
            check(value=="0"||value=="1","--short-cpu-check expects 0|1");a.short_cpu_check=value=="1";
        } else if (key == "--short-cpu-math") a.short_cpu_math=value;
        else if (key == "--input-scale") {
            size_t end = 0;
            a.input_scale = std::stof(value, &end);
            check(end == value.size() && std::isfinite(a.input_scale) && a.input_scale > 0,
                  "input scale must be finite and positive");
        } else throw std::runtime_error("unknown option " + key);
    }
    check(!a.model.empty() && !a.out.empty(),
          "usage: -m MODEL --out NEW-DIR [--backend cuda|cpu --layer 3 --expert 0 --tokens 128 "
          "--threads 64 --raw 0|1 --src-precision both|auto|q8 --acc-precision auto|f32 --input-scale 1 "
          "--short-cpu-check 0|1 --short-cpu-math both|compat|ggml]");
    check(a.short_cpu_math=="both"||a.short_cpu_math=="compat"||a.short_cpu_math=="ggml","invalid short CPU math mode");
    if(a.short_cpu_check){
        check(a.backend=="cuda","--short-cpu-check requires native CUDA reference");
        if(!tokens_given)a.tokens={1,7,8,16,128};
        if(!source_given)a.source_precision="auto";
    }
    check(a.backend == "cpu" || a.backend == "cuda", "backend must be cpu|cuda");
    check(a.source_precision == "auto" || a.source_precision == "q8" || a.source_precision == "both",
          "source precision must be auto|q8|both");
    check(a.accumulator == "auto" || a.accumulator == "f32", "accumulator must be auto|f32");
    check(a.tokens.back() <= 2048, "tokens above diagnostic cap");
    if (a.backend == "cuda") {
        const char * session = std::getenv("M31_CARD_SESSION");
        check(session && std::string(session) == "1", "CUDA requires aictl/flock runner (M31_CARD_SESSION=1)");
    }
    return a;
}
void clear_captures(ggml_backend_t backend) {
#ifdef M31_HAVE_CUDA
    ggml_backend_cuda_graph_cache_clear(backend);
#else
    (void) backend;
#endif
}
struct Backend {
    ggml_backend_t value = nullptr;
    ~Backend() { if (value) ggml_backend_free(value); }
};
struct Context {
    ggml_context * value = nullptr;
    explicit Context(size_t bytes) : value(ggml_init({bytes, nullptr, true})) {
        check(value != nullptr, "context allocation failed");
    }
    ~Context() { if (value) ggml_free(value); }
    Context(const Context &) = delete;
};
struct WeightSet {
    Context context{64 * 1024};
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, 3> tensors{};
    std::array<float, 3> scales{1, 1, 1};
    std::array<size_t, 3> source_bytes{};
    std::array<std::string, 3> names;
    ~WeightSet() { if (buffer) ggml_backend_buffer_free(buffer); }
    WeightSet(const strata::GgufModel & model, const Geometry & g, int layer, int expert, ggml_backend_t backend) {
        std::array<const uint8_t *, 3> source{};
        const char * roles[] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            const int64_t n0 = r == 2 ? g.expert_ff : g.embd;
            const int64_t n1 = r == 2 ? g.embd : g.expert_ff;
            names[size_t(r)] = layer_name(layer, std::string("ffn_") + roles[r] + "_exps.weight");
            size_t shard = 0;
            const auto * info = model.find(names[size_t(r)], &shard);
            check(info && info->shape == std::vector<uint64_t>{uint64_t(n0), uint64_t(n1), uint64_t(g.experts)} &&
                  info->type < GGML_TYPE_COUNT && model.in_bounds(*info, shard), "invalid weight " + names[size_t(r)]);
            const auto type = ggml_type(info->type);
            const size_t bytes = ggml_row_size(type, n0) * size_t(n1);
            source_bytes[size_t(r)] = bytes;
            source[size_t(r)] = model.shard(shard).tensor_data(*info) + size_t(expert) * bytes;
            // A one-expert 3D tensor works for both mul_mat and mul_mat_id.
            tensors[size_t(r)] = ggml_new_tensor_3d(context.value, type, n0, n1, 1);
            const std::string scale_name = names[size_t(r)].substr(0, names[size_t(r)].size() - 6) + "scale";
            const auto * scale = model.find(scale_name, &shard);
            if (scale) {
                check(scale->type == GGML_TYPE_F32 && scale->shape == std::vector<uint64_t>{uint64_t(g.experts)} &&
                      model.in_bounds(*scale, shard), "invalid external scales");
                std::memcpy(&scales[size_t(r)], model.shard(shard).tensor_data(*scale) + size_t(expert) * sizeof(float), sizeof(float));
            } else check(type != GGML_TYPE_NVFP4, "missing NVFP4 scale");
            check(std::isfinite(scales[size_t(r)]), "non-finite external scale");
        }
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(context.value, ggml_backend_get_default_buffer_type(backend));
        check(buffer != nullptr, "single-expert weight allocation failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int r = 0; r < 3; ++r) {
            check(ggml_nbytes(tensors[size_t(r)]) == source_bytes[size_t(r)], "source byte count differs");
            ggml_backend_tensor_set(tensors[size_t(r)], source[size_t(r)], 0, source_bytes[size_t(r)]);
        }
    }
};
std::vector<float> random_input(size_t count, uint64_t seed, float scale) {
    std::vector<float> x(count);
    for (auto & v : x) {
        // Fixed bit-based generator; avoids std::normal_distribution differences across libraries.
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        v = (float((seed >> 40) & 0xffffffu) * (1.0f / 8388608.0f) - 1.0f) * 1.7320508075688772f * scale;
    }
    return x;
}
using Outputs = std::map<std::string, std::vector<float>>;
struct Mode {
    std::string name;
    ggml_backend_t backend;
    const WeightSet * weights;
    bool indirect = false, q8 = false, cpu_quants = false;
    bool short_cpu_compat = false;
};
Outputs evaluate(const Mode & mode, int role, int n, const std::vector<float> & input, bool acc_f32) {
    if(mode.short_cpu_compat){
        check(!acc_f32,"compatible CPU MMQ diagnostic models native auto accumulation only");
        Outputs output;const auto &w=*mode.weights;
        auto mm=[&](int r,const std::vector<float>&activation,const std::string &name){
            const auto *weight=w.tensors[size_t(r)];const size_t columns=size_t(weight->ne[0]),rows=size_t(weight->ne[1]);
            check(activation.size()==columns*size_t(n),"compatible CPU input geometry differs");
            std::vector<float> raw;
            if(weight->type==GGML_TYPE_NVFP4){
                raw.resize(rows*size_t(n));
                for(int token=0;token<n;++token){
                    cuda_nvfp4_cpu::Activation q;cuda_nvfp4_cpu::quantize(activation.data()+size_t(token)*columns,columns,q);
                    for(size_t row=0;row<rows;++row)raw[size_t(token)*rows+row]=cuda_nvfp4_cpu::dot(static_cast<const uint8_t*>(weight->data)+row*weight->nb[1],q,columns);
                }
            }else{
                check(cuda_math_cpu::supported(weight->type),"unsupported compatible CPU quant format");
                cuda_math_cpu::mul_mat_rows(weight->type,weight->data,weight->nb[1],rows,columns,activation.data(),size_t(n),raw);
            }
            output[name+"_raw"]=raw;for(auto &value:raw)value*=w.scales[size_t(r)];output[name+"_scaled"]=raw;return raw;
        };
        if(role>=0)mm(role,input,"mm");
        else{
            const auto gate=mm(0,input,"gate"),up=mm(1,input,"up");std::vector<float> activation(gate.size());
            for(size_t i=0;i<gate.size();++i){const float a=std::min(gate[i],7.f),b=std::clamp(up[i],-7.f,7.f);activation[i]=(a/(1.f+expf(1.702f*(-a))))*(b+1.f);}
            output["oai"]=activation;mm(2,activation,"down");
        }
        return output;
    }
#ifdef M31_CPU_QUANTS_CUDA
    if (mode.cpu_quants) {
        GpuCpuQuants gpu;
        Outputs output;
        const auto & w = *mode.weights;
        std::vector<int32_t> ids(size_t(n), 0);
        auto mm = [&](int r, const std::vector<float> & activation, const std::string & name) {
            const auto * weight = w.tensors[size_t(r)];
            auto raw = gpu.multiply(weight->type, weight->data, weight->nb[2], int(weight->ne[0]), int(weight->ne[1]), 1, n, activation, ids);
            output[name + "_raw"] = raw;
            for (auto & value : raw) value *= w.scales[size_t(r)];
            output[name + "_scaled"] = raw;
            return raw;
        };
        if (role >= 0) mm(role, input, "mm");
        else {
            const auto gate = mm(0, input, "gate");
            const auto up = mm(1, input, "up");
            std::vector<float> activation(gate.size());
            gpu_host_oai_rows(gpu.host_executor(), size_t(w.tensors[0]->ne[1]), gate, up, activation);
            output["oai"] = activation;
            mm(2, activation, "down");
        }
        return output;
    }
#endif
    Context context(8 * 1024 * 1024);
    auto c = context.value;
    auto graph = ggml_new_graph_custom(c, 2048, false);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mode.backend));
    check(allocator != nullptr, "graph allocator failed");
    struct GraphGuard {
        ggml_backend_t backend;
        ggml_gallocr_t allocator;
        ~GraphGuard() {
            ggml_backend_synchronize(backend);
            clear_captures(backend);
            ggml_gallocr_free(allocator);
        }
    } guard{mode.backend, allocator};
    const auto & w = *mode.weights;
    const int input_role = role < 0 ? 0 : role;
    auto x = mode.indirect ?
        ggml_new_tensor_3d(c, GGML_TYPE_F32, w.tensors[size_t(input_role)]->ne[0], 1, n) :
        ggml_new_tensor_2d(c, GGML_TYPE_F32, w.tensors[size_t(input_role)]->ne[0], n);
    check(ggml_nelements(x) == int64_t(input.size()), "input dimensions differ");
    ggml_set_input(x);
    ggml_tensor * ids = nullptr;
    if (mode.indirect) {
        ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, n);
        ggml_set_input(ids);
    }
    std::map<std::string, ggml_tensor *> capture;
    auto save = [&](const std::string & name, ggml_tensor * tensor) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
        capture.emplace(name, tensor);
        return tensor;
    };
    auto mm = [&](int r, ggml_tensor * activation, const std::string & name) {
        auto weight = w.tensors[size_t(r)];
        auto result = mode.indirect ? ggml_mul_mat_id(c, weight, activation, ids) : ggml_mul_mat(c, weight, activation);
        if (acc_f32) ggml_prec_set_acc(result, GGML_PREC_F32);
        else if (weight->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(result, GGML_PREC_BF16);
        if (mode.q8) ggml_prec_set_src(result, GGML_PREC_Q8, 1);
        save(name + "_raw", result);
        // Apply even scale=1 here to compare primitive and scale semantics explicitly.
        return save(name + "_scaled", ggml_scale(c, result, w.scales[size_t(r)]));
    };
    if (role >= 0) mm(role, x, "mm");
    else {
        auto gate = mm(0, x, "gate");
        auto up = mm(1, x, "up");
        auto activation = save("oai", ggml_swiglu_oai(c, gate, up, 1.702f, 7.0f));
        mm(2, activation, "down");
    }
    const auto device = ggml_backend_get_device(mode.backend);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        check(ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i)), "backend does not support diagnostic graph");
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (ids) {
        std::vector<int32_t> zero_ids(size_t(n), 0);
        ggml_backend_tensor_set(ids, zero_ids.data(), 0, zero_ids.size() * sizeof(int32_t));
    }
    check(ggml_backend_graph_compute(mode.backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    ggml_backend_synchronize(mode.backend);
    Outputs output;
    for (const auto & pair : capture) {
        check(pair.second->type == GGML_TYPE_F32 && ggml_is_contiguous(pair.second), "expected contiguous F32 result");
        auto & values = output[pair.first];
        values.resize(size_t(ggml_nelements(pair.second)));
        ggml_backend_tensor_get(pair.second, values.data(), 0, values.size() * sizeof(float));
    }
    return output;
}
void raw_file(const fs::path & path, const std::vector<float> & values) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    check(bool(f), "cannot write raw file");
}
struct Metrics {
    size_t count = 0, nonfinite_reference = 0, nonfinite_candidate = 0;
    double max_abs = 0, max_reference_abs = 0, rms_reference = 0, rms_candidate = 0;
    double rmse = 0, relative_l2 = 0, cosine = 0;
    bool bitwise = false;
};
Metrics metrics(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size() && !a.empty(), "comparison sizes differ");
    Metrics m;
    m.count = a.size();
    m.bitwise = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    double aa = 0, bb = 0, ab = 0, dd = 0;
    size_t finite = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) ++m.nonfinite_reference;
        if (!std::isfinite(b[i])) ++m.nonfinite_candidate;
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) continue;
        const double x = a[i], y = b[i], d = x - y;
        aa += x * x; bb += y * y; ab += x * y; dd += d * d;
        m.max_abs = std::max(m.max_abs, std::abs(d));
        m.max_reference_abs = std::max(m.max_reference_abs, std::abs(x));
        ++finite;
    }
    if (finite) {
        m.rms_reference = std::sqrt(aa / finite);
        m.rms_candidate = std::sqrt(bb / finite);
        m.rmse = std::sqrt(dd / finite);
        m.relative_l2 = std::sqrt(dd / std::max(aa, 1e-300));
        m.cosine = aa && bb ? ab / std::sqrt(aa * bb) : aa == bb ? 1 : 0;
    }
    return m;
}
void metric_json(std::ostream & out, const Metrics & m) {
    out << "{\"count\":" << m.count << ",\"bitwise\":" << (m.bitwise ? "true" : "false")
        << ",\"nonfinite_reference\":" << m.nonfinite_reference << ",\"nonfinite_candidate\":" << m.nonfinite_candidate
        << ",\"max_abs\":" << m.max_abs << ",\"max_reference_abs\":" << m.max_reference_abs
        << ",\"rms_reference\":" << m.rms_reference << ",\"rms_candidate\":" << m.rms_candidate
        << ",\"rmse\":" << m.rmse << ",\"relative_l2\":" << m.relative_l2 << ",\"cosine\":" << m.cosine << '}';
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = parse(argc, argv);
        check(fs::create_directory(args.out), "output directory exists; use a new path");
        Backend cpu, gpu;
        cpu.value = ggml_backend_cpu_init();
        check(cpu.value != nullptr, "cannot initialize CPU backend");
        ggml_backend_cpu_set_n_threads(cpu.value, args.threads);
        if (args.backend == "cuda") {
#ifdef M31_HAVE_CUDA
            gpu.value = ggml_backend_cuda_init(0);
            check(gpu.value != nullptr, "cannot initialize CUDA0");
#else
            throw std::runtime_error("m31_expert_check was built without CUDA");
#endif
        }
        strata::GgufModel model(strata::gguf_split_paths(args.model));
        const auto g = read_geometry(model);
        check(args.expert < g.experts, "expert ID out of range");
        std::ostringstream report;
        report << std::setprecision(12) << "{\"format\":\"m31-expert-check-v1\",\"model\":" << run::quote(fs::canonical(args.model).string())
            << ",\"seed\":" << args.seed << ",\"input_scale\":" << args.input_scale << ",\"expert\":" << args.expert
            << ",\"threads\":" << args.threads << ",\"src_precision\":" << run::quote(args.source_precision)
            << ",\"acc_precision\":" << run::quote(args.accumulator) << ",\"raw\":" << (args.raw ? "true" : "false")
            << ",\"fusion_disabled\":" << ((std::getenv("GGML_CUDA_DISABLE_FUSION") && std::string(std::getenv("GGML_CUDA_DISABLE_FUSION")) == "1") ? "true" : "false")
            << ",\"mmq_precision_env\":" << run::quote(std::getenv("GGML_CUDA_MMQ_PREC") ? std::getenv("GGML_CUDA_MMQ_PREC") : "")
            << ",\"short_cpu_check\":" << (args.short_cpu_check?"true":"false") << ",\"short_cpu_math\":" <<run::quote(args.short_cpu_math)
            << ",\"note\":" << run::quote(args.short_cpu_check
                ?"Native CUDA mul_mat_id reference on identical original single-expert bytes. CPU compatible MMQ uses CUDA activation quantizers; small-n native MMV may differ. Full FFN additionally includes CPU expf OAI. No accuracy or speed acceptance is implied."
                :"Each result is compared with CPU mul_mat on identical bytes and inputs. auto uses BF16 accumulator for NVFP4. q8 forces source precision, unless overridden by GGML_CUDA_MMQ_PREC. No pass/fail tolerance is implied.") << ",\"cases\":[";
        bool first_case = true;
        for (int layer : args.layers) {
            check(g.routed(layer), "requested layer is not a routed layer");
            WeightSet cpu_weights(model, g, layer, args.expert, cpu.value);
            std::unique_ptr<WeightSet> gpu_weights;
            if (gpu.value) gpu_weights = std::make_unique<WeightSet>(model, g, layer, args.expert, gpu.value);
            std::vector<Mode> modes{{"cpu_mm", cpu.value, &cpu_weights, false, false}, {"cpu_id", cpu.value, &cpu_weights, true, false}};
            if(args.short_cpu_check&&args.short_cpu_math=="compat")modes.clear();
            if (gpu.value) {
                if (args.source_precision != "q8") {
                    modes.push_back({"cuda_mm_auto", gpu.value, gpu_weights.get(), false, false});
                    modes.push_back({"cuda_id_auto", gpu.value, gpu_weights.get(), true, false});
                }
                if (args.source_precision != "auto") {
                    modes.push_back({"cuda_mm_q8", gpu.value, gpu_weights.get(), false, true});
                    modes.push_back({"cuda_id_q8", gpu.value, gpu_weights.get(), true, true});
                }
            }
#ifdef M31_CPU_QUANTS_CUDA
            if (gpu.value) modes.push_back({"cuda_cpu_quants", gpu.value, gpu_weights.get(), true, false, true});
#endif
            if(args.short_cpu_check&&args.short_cpu_math!="ggml")modes.push_back({"cpu_cuda_mmq",cpu.value,&cpu_weights,false,false,false,true});
            if(args.short_cpu_check){
                const std::string name=args.source_precision=="q8"?"cuda_id_q8":"cuda_id_auto";
                const auto it=std::find_if(modes.begin(),modes.end(),[&](const Mode&m){return m.name==name;});
                check(it!=modes.end(),"native reference mode unavailable");std::iter_swap(modes.begin(),it);
            }
            for (int n : args.tokens) for (int role : {0, 1, 2, -1}) {
                const std::string role_name = role < 0 ? "full" : role == 0 ? "gate" : role == 1 ? "up" : "down";
                const std::string stem = "l" + std::to_string(layer) + "_n" + std::to_string(n) + "_" + role_name;
                const int64_t input_size = role == 2 ? g.expert_ff : g.embd;
                const auto input = random_input(size_t(input_size) * size_t(n), args.seed + uint64_t(layer) * 1024 + uint64_t(role + 1), args.input_scale);
                if (args.raw) raw_file(fs::path(args.out) / (stem + "_input.f32"), input);
                const auto reference_start=std::chrono::steady_clock::now();
                const auto reference = evaluate(modes.front(), role, n, input, args.accumulator == "f32");
                const double reference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-reference_start).count();
                if (!first_case) report << ',';
                first_case = false;
                report << "{\"layer\":" << layer << ",\"tokens\":" << n << ",\"operation\":" << run::quote(role_name)
                    << ",\"types\":[" << int(cpu_weights.tensors[0]->type) << ',' << int(cpu_weights.tensors[1]->type) << ',' << int(cpu_weights.tensors[2]->type)
                    << "],\"type_names\":[" << run::quote(ggml_type_name(cpu_weights.tensors[0]->type)) << ','
                    << run::quote(ggml_type_name(cpu_weights.tensors[1]->type)) << ',' << run::quote(ggml_type_name(cpu_weights.tensors[2]->type))
                    << "],\"scales\":[" << cpu_weights.scales[0] << ',' << cpu_weights.scales[1] << ',' << cpu_weights.scales[2]
                    << "],\"reference_mode\":" << run::quote(modes.front().name) << ",\"results\":[";
                bool first_result = true;
                for (const auto & mode : modes) {
                    const auto evaluated_start=std::chrono::steady_clock::now();
                    const auto candidate = mode.name == modes.front().name ? reference : evaluate(mode, role, n, input, args.accumulator == "f32");
                    const double evaluation_ms=mode.name==modes.front().name?reference_ms:std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-evaluated_start).count();
                    for (const auto & stage : candidate) {
                        if (!first_result) report << ',';
                        first_result = false;
                        report << "{\"mode\":" << run::quote(mode.name) << ",\"stage\":" << run::quote(stage.first) << ",\"metrics\":";
                        metric_json(report, metrics(reference.at(stage.first), stage.second));
                        if(args.short_cpu_check)report << ",\"mode_evaluate_ms\":" << evaluation_ms;
                        if (args.raw) {
                            const auto filename = stem + "_" + mode.name + "_" + stage.first + ".f32";
                            raw_file(fs::path(args.out) / filename, stage.second);
                            report << ",\"file\":" << run::quote(filename);
                        }
                        report << '}';
                    }
                }
                report << "]}";
                std::fprintf(stderr, "expert diagnostic layer=%d expert=%d tokens=%d operation=%s complete\n", layer, args.expert, n, role_name.c_str());
            }
        }
        report << "]}\n";
        std::ofstream output(fs::path(args.out) / "report.json");
        output << report.str();
        check(bool(output), "cannot write final report");
        std::fprintf(stderr, "expert diagnostic complete: %s/report.json\n", args.out.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
