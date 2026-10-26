#include "gpu_cpu_quants.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::m31 {
namespace {
void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(std::string("M31 CPU-quants CUDA: ") + cudaGetErrorString(result));
}
__device__ float ue4m3(uint8_t x) {
    if (x == 0 || x == 127) return 0;
    const int e = (x >> 3) & 15, m = x & 7;
    return e ? ldexpf(1.f + float(m)/8.f, e-8) : ldexpf(float(m), -10);
}
// Eight lanes reproduce the eight AVX2 accumulators; the reduction uses the
// same (upper+lower), (lane0+lane2), (lane0+lane1) tree as hsum_float_8.
template<int Type>
__global__ void multiply_kernel(const uint8_t * pool, size_t stride, int columns,
                                int rows, int used, int n, const uint8_t * quant,
                                size_t quant_row_bytes, bool broadcast,
                                const int32_t * ids, float * output, bool tiled_fma) {
    const int lane = threadIdx.x & 7;
    const unsigned group_mask = 0xffu << (threadIdx.x & 24);
    const size_t item = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 8;
    const size_t count = size_t(rows) * used * n;
    if (item >= count) return;
    const int row = int(item % rows), route = int(item / rows);
    const int input_row = broadcast ? route / used : route;
    const auto * w = pool + size_t(ids[route]) * stride;
    const auto * a = quant + size_t(input_row) * quant_row_bytes;
    const bool tiled = Type != GGML_TYPE_NVFP4 && ids[size_t(used)*n+route] != 0;
    float acc = 0;
    if constexpr (Type == GGML_TYPE_IQ3_XXS) {
        const auto * x = reinterpret_cast<const block_iq3_xxs *>(w) + size_t(row) * (columns/256);
        const auto * y = reinterpret_cast<const block_q8_K *>(a);
        for (int b = 0; b < columns/256; ++b) {
            int isum = 0;
            for (int sub = 0; sub < 8; ++sub) {
                const uint8_t * gas = x[b].qs + 64 + sub*4;
                const uint32_t packed = uint32_t(gas[0]) | (uint32_t(gas[1])<<8) | (uint32_t(gas[2])<<16) | (uint32_t(gas[3])<<24);
                const int scale = 2*(packed >> 28)+1;
                const int group8 = lane/2;
                const uint8_t signs = ksigns_iq2xs[(packed >> (7*group8)) & 127];
                const uint32_t grid = iq3xxs_grid[x[b].qs[sub*8 + lane]];
                int partial = 0;
                for (int j = 0; j < 4; ++j) {
                    const int pos = lane*4+j;
                    const int v = int((grid >> (8*j)) & 255);
                    partial += v * int(y[b].qs[sub*32+pos]) * ((signs & (1 << (pos%8))) ? -1 : 1);
                }
                isum += scale * partial;
            }
            if (tiled) {
                int total = isum;
                total += __shfl_xor_sync(group_mask, total, 4, 8);
                total += __shfl_xor_sync(group_mask, total, 2, 8);
                total += __shfl_xor_sync(group_mask, total, 1, 8);
                const float weight_d = __fmul_rn(__half2float(x[b].d), 0.25f);
                const float product = __fmul_rn(__int2float_rn(total), weight_d);
                acc = tiled_fma ? fmaf(product, y[b].d, acc) : __fadd_rn(acc, __fmul_rn(product, y[b].d));
            } else {
                const float d = __half2float(x[b].d) * y[b].d;
                acc = fmaf(d, float(isum), acc);
            }
        }
    } else if constexpr (Type == GGML_TYPE_IQ4_XS) {
        const auto * x = reinterpret_cast<const block_iq4_xs *>(w) + size_t(row) * (columns/256);
        const auto * y = reinterpret_cast<const block_q8_K *>(a);
        for (int b = 0; b < columns/256; ++b) {
            int isum = 0;
            for (int sub = 0; sub < 8; ++sub) {
                const int scale = (((x[b].scales_l[sub/2] >> (4*(sub%2))) & 15) | (((x[b].scales_h >> (2*sub)) & 3) << 4)) - 32;
                for (int j = 0; j < 4; ++j) {
                    const int pos = lane*4+j;
                    const uint8_t q = x[b].qs[sub*16+(pos%16)];
                    const int value = kvalues_iq4nl[pos < 16 ? q & 15 : q >> 4];
                    isum += scale * value * int(y[b].qs[sub*32+pos]);
                }
            }
            if (tiled) {
                int total = isum;
                total += __shfl_xor_sync(group_mask, total, 4, 8);
                total += __shfl_xor_sync(group_mask, total, 2, 8);
                total += __shfl_xor_sync(group_mask, total, 1, 8);
                const float weight_d = __half2float(x[b].d);
                const float product = __fmul_rn(__int2float_rn(total), weight_d);
                acc = tiled_fma ? fmaf(product, y[b].d, acc) : __fadd_rn(acc, __fmul_rn(product, y[b].d));
            } else {
                const float d = __half2float(x[b].d) * y[b].d;
                acc = fmaf(d, float(isum), acc);
            }
        }
    } else if constexpr (Type == GGML_TYPE_NVFP4) {
        const auto * x = reinterpret_cast<const block_nvfp4 *>(w) + size_t(row) * (columns/64);
        const auto * y = reinterpret_cast<const block_q8_0 *>(a);
        for (int b = 0; b < columns/64; ++b) {
            for (int pair = 0; pair < 2; ++pair) {
                const int sub = pair*2 + lane/4;
                int isum = 0;
                for (int j = 0; j < 4; ++j) {
                    const int pos = (lane%4)*4+j;
                    const uint8_t q = x[b].qs[sub*8+(pos%8)];
                    const int value = kvalues_fp4[pos < 8 ? q & 15 : q >> 4];
                    isum += value * int(y[b*2+pair].qs[(sub%2)*16+pos]);
                }
                const float d = ue4m3(x[b].d[sub]) * __half2float(y[b*2+pair].d);
                acc = fmaf(d, float(isum), acc);
            }
        }
    }
    if (!tiled) {
        acc = __fadd_rn(acc, __shfl_xor_sync(group_mask, acc, 4, 8));
        acc = __fadd_rn(acc, __shfl_xor_sync(group_mask, acc, 2, 8));
        acc = __fadd_rn(acc, __shfl_xor_sync(group_mask, acc, 1, 8));
    }
    if (lane == 0) output[item] = Type == GGML_TYPE_IQ3_XXS && !tiled ? 0.25f*acc : acc;
}
}
struct GpuCpuQuants::Impl {
    uint8_t * quant = nullptr;
    int32_t * ids = nullptr;
    float * out = nullptr;
    size_t quant_capacity = 0, ids_capacity = 0, out_capacity = 0;
    size_t calls = 0;
    double quant_seconds = 0, roundtrip_seconds = 0;
    ~Impl() {
        const char * profile = std::getenv("M31_GPU_CPU_QUANTS_PROFILE");
        if (profile && std::string(profile) == "1")
            std::fprintf(stderr, "M31 CPU-quants diagnostic calls=%zu host_quant_ms=%.3f H2D_kernel_D2H_ms=%.3f scratch_bytes=%zu (correctness instrumentation, not M2 benchmark)\n",
                         calls, quant_seconds*1000, roundtrip_seconds*1000,
                         quant_capacity+ids_capacity+out_capacity);
        cudaFree(quant); cudaFree(ids); cudaFree(out);
    }
    template<class T> void reserve(T * & pointer, size_t & capacity, size_t bytes) {
        if (capacity >= bytes) return;
        cuda_check(cudaFree(pointer)); pointer = nullptr; capacity = 0;
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes)); capacity = bytes;
    }
};
GpuCpuQuants::GpuCpuQuants() : p_(std::make_unique<Impl>()) {}
GpuCpuQuants::~GpuCpuQuants() = default;
size_t GpuCpuQuants::scratch_bytes() const { return p_->quant_capacity+p_->ids_capacity+p_->out_capacity; }
std::vector<float> GpuCpuQuants::multiply(int type_int, const void * weights, size_t slot_stride,
                                        int columns, int rows, int used, int n,
                                        const std::vector<float> & input,
                                        const std::vector<int32_t> & ids) {
    if (!ggml_cpu_has_avx2() || !ggml_cpu_has_fma())
        throw std::runtime_error("M31 CPU-quants CUDA diagnostic requires an AVX2+FMA CPU ggml build");
    // GCC14 -O3 contracts even the VNNI epilogue's intrinsic mul+add into
    // vfmadd213ps (verified in the linked ggml-cpu tiled-kernel object).
    const bool tiled_fma = true;
    const auto type = ggml_type(type_int);
    if ((type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_NVFP4) ||
        !weights || columns <= 0 || rows <= 0 || used <= 0 || n <= 0 || ids.size() != size_t(n)*used ||
        (input.size() != size_t(columns)*n && input.size() != size_t(columns)*n*used))
        throw std::runtime_error("M31 CPU-quants CUDA: unsupported type or invalid dimensions");
    const auto * traits = ggml_get_type_traits_cpu(type);
    const auto * quant_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    if (!quant_traits || !quant_traits->from_float) throw std::runtime_error("missing CPU activation quantizer");
    const auto quant_start = std::chrono::steady_clock::now();
    const size_t qbytes = ggml_row_size(traits->vec_dot_type, columns);
    const size_t input_rows = input.size()/columns;
    std::vector<uint8_t> packed(qbytes*input_rows);
    for (size_t row = 0; row < input_rows; ++row)
        quant_traits->from_float(input.data()+row*columns, packed.data()+row*qbytes, columns);
    const auto quant_end = std::chrono::steady_clock::now();
    p_->quant_seconds += std::chrono::duration<double>(quant_end-quant_start).count();
    const size_t out_count = size_t(rows)*used*n;
    p_->reserve(p_->quant, p_->quant_capacity, packed.size());
    std::vector<int32_t> dispatch_ids(ids);
    dispatch_ids.resize(ids.size()*2);
    const char * tiled_env = std::getenv("GGML_CPU_TILED_MM");
    const char * tiled_force = std::getenv("GGML_CPU_TILED_MM_FORCE");
    const bool tiled_enabled = !tiled_env || std::atoi(tiled_env) != 0;
    const bool tiled_forced = tiled_force && std::atoi(tiled_force) == 1;
    // CPU MUL_MAT_ID chooses its accumulation algorithm per expert, at eight
    // assigned rows. Expert histograms refer to the original full layer batch.
    if (tiled_enabled && type != GGML_TYPE_NVFP4) {
        std::vector<int32_t> counts;
        for (auto id : ids) {
            if (id < 0) throw std::runtime_error("negative expert slot");
            if (size_t(id) >= counts.size()) counts.resize(size_t(id)+1);
            ++counts[size_t(id)];
        }
        for (size_t i = 0; i < ids.size(); ++i)
            dispatch_ids[ids.size()+i] = tiled_forced || counts[size_t(ids[i])] >= 8;
    }
    p_->reserve(p_->ids, p_->ids_capacity, dispatch_ids.size()*sizeof(int32_t));
    p_->reserve(p_->out, p_->out_capacity, out_count*sizeof(float));
    cuda_check(cudaMemcpy(p_->quant, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(p_->ids, dispatch_ids.data(), dispatch_ids.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    const int blocks = int((out_count*8+127)/128);
    const bool broadcast = input_rows == size_t(n);
#define M31_LAUNCH(T) multiply_kernel<T><<<blocks,128>>>(static_cast<const uint8_t *>(weights), slot_stride, columns, rows, used, n, p_->quant, qbytes, broadcast, p_->ids, p_->out, tiled_fma)
    if (type == GGML_TYPE_IQ3_XXS) { M31_LAUNCH(GGML_TYPE_IQ3_XXS); }
    else if (type == GGML_TYPE_IQ4_XS) { M31_LAUNCH(GGML_TYPE_IQ4_XS); }
    else { M31_LAUNCH(GGML_TYPE_NVFP4); }
#undef M31_LAUNCH
    cuda_check(cudaGetLastError());
    std::vector<float> result(out_count);
    cuda_check(cudaMemcpy(result.data(), p_->out, out_count*sizeof(float), cudaMemcpyDeviceToHost));
    p_->roundtrip_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-quant_end).count();
    ++p_->calls;
    return result;
}
}
