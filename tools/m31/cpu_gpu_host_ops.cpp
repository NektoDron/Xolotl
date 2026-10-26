#include "cpu_gpu_host_ops.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <span>

namespace strata::m31 {
namespace {
bool overlaps(const void * first, size_t first_size, const void * second, size_t second_size) {
    if (!first_size || !second_size) return false;
    const auto a = reinterpret_cast<uintptr_t>(first), b = reinterpret_cast<uintptr_t>(second);
    return a <= b ? b - a < first_size : a - b < second_size;
}
}
GpuHostQuantLayout gpu_host_quant_layout(ggml_type type, size_t columns, size_t rows) {
    if (type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_NVFP4)
        throw std::invalid_argument("GPU host quantization: unsupported expert type");
    const auto * weights = ggml_get_type_traits_cpu(type);
    if (!weights) throw std::invalid_argument("GPU host quantization: missing weight trait");
    const auto * activation = ggml_get_type_traits_cpu(weights->vec_dot_type);
    if (!activation || !activation->from_float || !columns ||
        columns > size_t(std::numeric_limits<int64_t>::max()) || columns % ggml_blck_size(weights->vec_dot_type))
        throw std::invalid_argument("GPU host quantization: invalid activation row");
    const size_t row_bytes = ggml_row_size(weights->vec_dot_type, int64_t(columns));
    if (rows > std::numeric_limits<size_t>::max() / row_bytes)
        throw std::overflow_error("GPU host quantization: output size overflow");
    return {weights->vec_dot_type, columns, rows, row_bytes, rows * row_bytes};
}
void gpu_host_quantize_rows(GpuHostExecutor & executor, ggml_type weight_type, size_t columns,
                            const float * input_data, size_t input_elements, uint8_t * output_data, size_t output_bytes) {
    if ((input_elements && !input_data) || (output_bytes && !output_data) || input_elements > SIZE_MAX / sizeof(float))
        throw std::invalid_argument("GPU host quantization: invalid buffer");
    const std::span<const float> input(input_data, input_elements);
    const std::span<uint8_t> output(output_data, output_bytes);
    if (!columns || input.size() % columns) throw std::invalid_argument("GPU host quantization: partial row");
    const auto layout = gpu_host_quant_layout(weight_type, columns, input.size() / columns);
    if (output.size() != layout.bytes || overlaps(input.data(), input.size_bytes(), output.data(), output.size_bytes()))
        throw std::invalid_argument("GPU host quantization: output size/overlap");
    const auto * quant_traits = ggml_get_type_traits_cpu(layout.activation_type);
    executor.parallel_rows(layout.rows, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row)
            quant_traits->from_float(input.data() + row * columns, output.data() + row * layout.row_bytes, int64_t(columns));
    });
}
void gpu_host_oai_rows(GpuHostExecutor & executor, size_t columns, const float * gate_data,
                       const float * up_data, float * output_data, size_t elements) {
    if ((elements && (!gate_data || !up_data || !output_data)) || elements > SIZE_MAX / sizeof(float))
        throw std::invalid_argument("GPU host OAI: invalid buffer");
    const std::span<const float> gate(gate_data, elements), up(up_data, elements);
    const std::span<float> output(output_data, elements);
    if (!columns || gate.size() % columns || gate.size() != up.size() || gate.size() != output.size())
        throw std::invalid_argument("GPU host OAI: invalid row shape");
    if ((output.data() != gate.data() && overlaps(gate.data(), gate.size_bytes(), output.data(), output.size_bytes())) ||
        (output.data() != up.data() && overlaps(up.data(), up.size_bytes(), output.data(), output.size_bytes())))
        throw std::invalid_argument("GPU host OAI: partially overlapping output");
    executor.parallel_rows(gate.size() / columns, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row)
            for (size_t column = 0; column < columns; ++column) {
                const size_t i = row * columns + column;
                const float a = std::min(gate[i], 7.f), b = std::clamp(up[i], -7.f, 7.f);
                const float glu = a / (1.f + expf(1.702f*(-a)));
                output[i] = glu*(b+1.f);
            }
    });
}
} // namespace strata::m31
