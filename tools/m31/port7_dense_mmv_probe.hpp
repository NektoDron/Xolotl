#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdint>
#include <vector>
struct DenseMmvProbeResult {
    int columns = 0, family = 0;
    bool forced_mmv = false, marked_mmv = false, stock_uses_mmv = false;
    uint64_t values = 0, different = 0;
    double max_abs = 0, rms = 0;
    double scalar_mmv_us = 0, batch_mmv_us = 0, stock_graph_us = 0, marked_graph_us = 0;
};
std::vector<DenseMmvProbeResult> dense_mmv_probe(ggml_backend_t backend, ggml_tensor * weight);
uint64_t dense_mmv_msa_isolation_probe(ggml_backend_t backend);
