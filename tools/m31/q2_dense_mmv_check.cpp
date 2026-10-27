#include "m31_geometry.hpp"
#include "q2_dense_mmv_probe.hpp"
#include "ggml-cuda.h"
#include "json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace strata;
using Json = nlohmann::json;
struct Projection {
    std::string role;
    std::vector<std::string> parts;
    int type = 0;
    int64_t columns = 0, rows = 0;
    uint64_t bytes = 0;
};
int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        if (argc != 3 || !std::getenv("M31_CARD_SESSION") || std::strcmp(std::getenv("M31_CARD_SESSION"), "1") ||
                !std::getenv("CUDA_VISIBLE_DEVICES") || std::strcmp(std::getenv("CUDA_VISIBLE_DEVICES"), "0"))
            throw std::runtime_error("usage: aictl/flock GPU0 probe MODEL OUT.json");
        std::ifstream existing(argv[2]);
        if (existing.good()) throw std::runtime_error("output already exists");
        const auto model = GgufModel::open(argv[1]);
        const auto geometry = m31::read_geometry(model);
        std::vector<Projection> selected;
        std::set<std::tuple<std::string,int,int64_t,int64_t>> signatures;
        auto add = [&](const std::string & role, const std::vector<std::string> & names) {
            Projection projection; projection.role = role; projection.parts = names;
            for (const auto & name : names) {
                size_t shard = 0;
                auto info = model.find(name, &shard);
                if (!info || info->shape.size() != 2 || !model.in_bounds(*info, shard)) return;
                if (projection.rows && (projection.type != int(info->type) || projection.columns != int64_t(info->shape[0]))) return;
                projection.type = int(info->type); projection.columns = int64_t(info->shape[0]);
                projection.rows += int64_t(info->shape[1]); projection.bytes += tensor_payload_bytes(*info);
            }
            if (signatures.emplace(role,projection.type,projection.columns,projection.rows).second)
                selected.push_back(std::move(projection));
        };
        for (int layer = 0; layer < geometry.layers; ++layer) {
            const std::string p = "blk."+std::to_string(layer)+".";
            for (const auto & role : {"attn_q", "attn_k", "attn_v", "attn_output", "indexer.q_proj", "indexer.k_proj", "ffn_gate_inp"})
                add(role, {p+role+".weight"});
            const std::string suffix = geometry.routed(layer) ? "_shexp" : "";
            for (const auto & role : {"ffn_gate", "ffn_up", "ffn_down"})
                add(role+suffix, {p+role+suffix+".weight"});
            add("packed_qkv", {p+"attn_q.weight",p+"attn_k.weight",p+"attn_v.weight"});
            add("packed_index_qk", {p+"indexer.q_proj.weight",p+"indexer.k_proj.weight"});
            add("packed_gu"+suffix, {p+"ffn_gate"+suffix+".weight",p+"ffn_up"+suffix+".weight"});
        }
        add("head", {"output.weight"});
        backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("CUDA backend init failed");
        const uint64_t msa_differences = dense_mmv_msa_isolation_probe(backend);
        if (msa_differences) throw std::runtime_error("unmarked MSA score changed after marking a separate projection");
        Json report = {{"scope","UD-Q2_K_XL actual unchanged GGUF dense weights, two deterministic synthetic activation families, scalar stock GGML reference vs stock batch and direct forced MMV; component only, not model logits or DSpark proof"},
                       {"model",argv[1]}, {"signature_count",selected.size()},
                       {"unmarked_msa_score",{{"values",4096},{"different",msa_differences},{"recaptured_after_mark",true}}},
                       {"cases",Json::array()}};
        for (const auto & projection : selected) {
            auto * context = ggml_init({1024*1024,nullptr,true});
            auto * weight = ggml_new_tensor_2d(context,ggml_type(projection.type),projection.columns,projection.rows);
            ggml_set_name(weight,projection.role.c_str());
            auto * buffer = ggml_backend_alloc_ctx_tensors(context,backend);
            if (!buffer) { ggml_free(context); throw std::runtime_error("weight allocation failed"); }
            ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            size_t offset = 0;
            for (const auto & name : projection.parts) {
                size_t shard = 0; const auto * info = model.find(name,&shard);
                const size_t bytes = tensor_payload_bytes(*info);
                ggml_backend_tensor_set(weight,model.shard(shard).tensor_data(*info),offset,bytes); offset += bytes;
            }
            try {
                for (const auto & result : dense_mmv_probe(backend,weight)) {
                    Json item = {{"role",projection.role},{"parts",projection.parts},{"type",ggml_type_name(weight->type)},
                        {"input",projection.columns},{"output",projection.rows},{"weight_bytes",projection.bytes},
                        {"columns",result.columns},{"activation_family",result.family},{"forced_mmv",result.forced_mmv},{"marked_mmv",result.marked_mmv},
                        {"stock_uses_mmv",result.stock_uses_mmv},{"values",result.values},{"different",result.different},
                        {"max_abs",result.max_abs},{"rms",result.rms},{"scalar_mmv_us",result.scalar_mmv_us},
                        {"batch_mmv_us",result.batch_mmv_us},{"stock_graph_us",result.stock_graph_us},{"marked_graph_us",result.marked_graph_us}};
                    report["cases"].push_back(item);
                    std::fprintf(stderr,"MMV probe role=%s type=%s K=%d family=%d forced=%d marked=%d diff=%llu max_abs=%.8g scalar_us=%.3f batch_us=%.3f\n",
                        projection.role.c_str(),ggml_type_name(weight->type),result.columns,result.family,int(result.forced_mmv),int(result.marked_mmv),
                        (unsigned long long)result.different,result.max_abs,result.scalar_mmv_us,result.batch_mmv_us);
                }
            } catch (...) { ggml_backend_synchronize(backend); ggml_backend_cuda_graph_cache_clear(backend); ggml_backend_buffer_free(buffer); ggml_free(context); throw; }
            ggml_backend_synchronize(backend); ggml_backend_cuda_graph_cache_clear(backend);
            ggml_backend_buffer_free(buffer); ggml_free(context);
        }
        ggml_backend_free(backend); backend = nullptr;
        std::ofstream output(argv[2]); output << report.dump(2) << '\n';
        std::printf("{\"complete\":true,\"signature_count\":%zu,\"cases\":%zu}\n",selected.size(),report["cases"].size());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"dense MMV probe: %s\n",error.what());
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
