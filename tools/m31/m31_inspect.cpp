#include "m31_geometry.hpp"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: m31_inspect ANY-SHARD.gguf (headers only, no CUDA)\n");
        return 2;
    }
    try {
        const auto model = strata::GgufModel::open(argv[1]);
        const auto g = strata::m31::read_geometry(model);
        std::printf("{\"layers\":%lld,\"embd\":%lld,\"vocab\":%lld,\"expert_bytes\":%llu,"
                    "\"other_bytes\":%llu,\"gpu_weight_bytes_without_embedding\":%llu,"
                    "\"f16_kv_bytes_32768\":%llu,\"nfc\":%s,\"routed_layers\":[",
                    (long long) g.layers, (long long) g.embd, (long long) g.vocab,
                    (unsigned long long) g.expert_bytes, (unsigned long long) g.other_bytes,
                    (unsigned long long) (g.other_bytes - g.embedding_bytes - g.expert_scale_bytes),
                    (unsigned long long) g.kv_bytes(32768), g.nfc ? "true" : "false");
        for (int64_t l = g.dense_lead; l < g.layers; ++l)
            std::printf("%s{\"layer\":%lld,\"gate\":%d,\"up\":%d,\"down\":%d}",
                        l == g.dense_lead ? "" : ",", (long long) l,
                        g.gate_types[l], g.up_types[l], g.down_types[l]);
        std::printf("]}\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
