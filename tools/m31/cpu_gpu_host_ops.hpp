#pragma once
#include "cpu_gpu_host_executor.hpp"
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
struct GpuHostQuantLayout {
    ggml_type activation_type;
    size_t columns, rows, row_bytes, bytes;
};
GpuHostQuantLayout gpu_host_quant_layout(ggml_type weight_type, size_t columns, size_t rows);
// Calls the original ggml CPU trait->from_float once for each complete row.
// The caller owns/reuses output storage; no full-size temporary or per-call pool.
void gpu_host_quantize_rows(GpuHostExecutor &, ggml_type weight_type, size_t columns,
                            const float * input, size_t input_elements,
                            uint8_t * output, size_t output_bytes);
inline void gpu_host_quantize_rows(GpuHostExecutor & executor, ggml_type type, size_t columns,
                                   const std::vector<float> & input, std::vector<uint8_t> & output) {
    gpu_host_quantize_rows(executor, type, columns, input.data(), input.size(), output.data(), output.size());
}
// Exact scalar expression from the accepted GPUExperts host OAI prototype.
// Disjoint buffers or exact in-place output==gate/up are supported.
void gpu_host_oai_rows(GpuHostExecutor &, size_t columns, const float * gate,
                       const float * up, float * output, size_t elements);
inline void gpu_host_oai_rows(GpuHostExecutor & executor, size_t columns,
                              const std::vector<float> & gate, const std::vector<float> & up,
                              std::vector<float> & output) {
    if (gate.size() != up.size() || gate.size() != output.size())
        throw std::invalid_argument("GPU host OAI: inconsistent vector sizes");
    gpu_host_oai_rows(executor, columns, gate.data(), up.data(), output.data(), gate.size());
}
} // namespace strata::m31
