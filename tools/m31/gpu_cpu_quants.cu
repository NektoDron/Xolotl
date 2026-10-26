#include "gpu_cpu_quants.hpp"
#include "cpu_gpu_host_ops.hpp"
#include "gpu_pinned_scratch.hpp"
#include "prefill_route_worklist.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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
#include "gpu_cpu_quants_batched.cuh"
#include "gpu_cpu_quants_tensor.cuh"
// Eight lanes reproduce the eight AVX2 accumulators; the reduction uses the
// same (upper+lower), (lane0+lane2), (lane0+lane1) tree as hsum_float_8.
template<int Type>
__global__ void multiply_kernel(const uint8_t * pool, size_t stride, int columns,
                                int rows, int used, int n, const uint8_t * quant,
                                size_t quant_row_bytes, bool broadcast,
                                const int32_t * ids, const int32_t * worklist, float * output, bool tiled_fma) {
    const int lane = threadIdx.x & 7;
    const unsigned group_mask = 0xffu << (threadIdx.x & 24);
    const size_t item = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 8;
    const size_t count = size_t(rows) * used * n;
    if (item >= count) return;
    const int row = int(item % rows);
    const int route = worklist ? worklist[item / rows] : int(item / rows);
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
    if (lane == 0) output[size_t(route)*rows+row] = Type == GGML_TYPE_IQ3_XXS && !tiled ? 0.25f*acc : acc;
}
}
struct GpuCpuQuants::Impl {
    GpuHostExecutor host{[] {
        auto options = gpu_host_options_from_environment();
        // Parallel host work remains explicit opt-in; the prior serial path is default.
        if (!std::getenv("M31_GPU_HOST_THREADS")) options.threads = 1;
        return options;
    }()};
    uint8_t * quant = nullptr;
    int32_t * ids = nullptr, * worklist = nullptr, * tiles = nullptr, * tensor_tiles = nullptr;
    float * out = nullptr;
    size_t quant_capacity = 0, ids_capacity = 0, out_capacity = 0, worklist_capacity = 0, tiles_capacity = 0, tensor_tiles_capacity = 0;
    size_t calls = 0;
    // High-water storage: quantization overwrites all used bytes, so alternating
    // gate/down dimensions need no allocation or zero initialization per call.
    std::vector<uint8_t> host_quant;
    double quant_seconds = 0, roundtrip_seconds = 0, prepare_seconds = 0, upload_seconds = 0, kernel_seconds = 0, postlaunch_seconds = 0;
    size_t kernel_timed_calls = 0;
    cudaEvent_t kernel_begin = nullptr, kernel_end = nullptr;
    cudaStream_t stream = nullptr;
    bool pending = false;
    std::unique_ptr<GpuPinnedScratch> pinned;
    size_t pinned_calls = 0, pinned_fallbacks = 0;
    bool pinned_last = false, pinned_reported = false;
    Impl() {
        const char * requested = std::getenv("M31_GPU_CPU_QUANTS_STREAM");
        const char * pinned_env = std::getenv("M31_GPU_CPU_QUANTS_PINNED");
        const bool want_pinned = pinned_env && std::string(pinned_env) == "1";
        if (want_pinned || (requested && std::string(requested) == "1")) {
            cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            std::fprintf(stderr,"M31 CPU-quants: private nonblocking CUDA stream enabled\n");
        }
        if (want_pinned) {
            try { pinned = std::make_unique<GpuPinnedScratch>(stream); }
            catch (...) { cudaStreamDestroy(stream); stream = nullptr; throw; }
        }
    }
    void drain() noexcept {
        if (stream && pending) cudaStreamSynchronize(stream);
        pending = false;
    }
    void upload(void * destination,const void * source,size_t bytes) {
        if (stream) {
            pending = true; // even a failed API call may have queued work
            cuda_check(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyHostToDevice,stream));
        } else cuda_check(cudaMemcpy(destination,source,bytes,cudaMemcpyHostToDevice));
    }
    void read(void * destination,const void * source,size_t bytes) {
        if (stream) {
            pending = true;
            cuda_check(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyDeviceToHost,stream));
            const auto status = cudaStreamSynchronize(stream);
            if (status == cudaSuccess) pending = false;
            cuda_check(status);
        } else cuda_check(cudaMemcpy(destination,source,bytes,cudaMemcpyDeviceToHost));
    }
    bool timing() {
        const char * value = std::getenv("M31_GPU_CPU_QUANTS_TIMING");
        if (!value || std::string(value) != "1") return false;
        if (!kernel_begin) {
            cuda_check(cudaEventCreate(&kernel_begin));
            cuda_check(cudaEventCreate(&kernel_end));
        }
        return true;
    }
    ~Impl() {
        drain(); // host input/metadata and device allocations still belong to us
        const char * profile = std::getenv("M31_GPU_CPU_QUANTS_PROFILE");
        if (profile && std::string(profile) == "1")
            std::fprintf(stderr, "M31 CPU-quants diagnostic calls=%zu host_quant_ms=%.3f H2D_kernel_D2H_ms=%.3f scratch_bytes=%zu (correctness instrumentation, not M2 benchmark)\n",
                         calls, quant_seconds*1000, roundtrip_seconds*1000,
                         quant_capacity+ids_capacity+out_capacity+worklist_capacity);
        cudaFree(quant); cudaFree(ids); cudaFree(out); cudaFree(worklist); cudaFree(tiles); cudaFree(tensor_tiles);
        if (kernel_begin) cudaEventDestroy(kernel_begin);
        if (kernel_end) cudaEventDestroy(kernel_end);
        pinned.reset(); // borrowed private stream must still exist during pinned release
        if (stream) cudaStreamDestroy(stream);
    }
    template<class T> void reserve(T * & pointer, size_t & capacity, size_t bytes) {
        if (capacity >= bytes) return;
        cuda_check(cudaFree(pointer)); pointer = nullptr; capacity = 0;
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes)); capacity = bytes;
    }
};
GpuCpuQuants::GpuCpuQuants() : p_(std::make_unique<Impl>()) {}
GpuCpuQuants::~GpuCpuQuants() = default;
GpuHostExecutor & GpuCpuQuants::host_executor() { return p_->host; }
size_t GpuCpuQuants::scratch_bytes() const { return p_->quant_capacity+p_->ids_capacity+p_->out_capacity+p_->worklist_capacity+p_->tiles_capacity+p_->tensor_tiles_capacity; }
void GpuCpuQuants::print_profile(const char * label) const {
    const auto host_cfg = p_->host.options();
    const auto host_stats = p_->host.stats();
    std::fprintf(stderr,"M31 CPU-quants profile label=%s calls=%zu quant_ms=%.3f prepare_ms=%.3f H2D_wall_ms=%.3f kernel_event_ms=%.3f kernel_timed_calls=%zu kernel_D2H_wall_ms=%.3f roundtrip_ms=%.3f device_scratch_bytes=%zu host_quant_bytes=%zu\n",
        label ? label : "runtime",p_->calls,p_->quant_seconds*1000,p_->prepare_seconds*1000,p_->upload_seconds*1000,
        p_->kernel_seconds*1000,p_->kernel_timed_calls,p_->postlaunch_seconds*1000,p_->roundtrip_seconds*1000,scratch_bytes(),p_->host_quant.size());
    std::fprintf(stderr,"M31 CPU-quants host profile label=%s threads=%u serial_jobs=%zu parallel_jobs=%zu worker_ranges=%zu private_stream=%d\n",
        label ? label : "runtime",host_cfg.threads,host_stats.serial_jobs,host_stats.parallel_jobs,host_stats.worker_ranges,int(p_->stream != nullptr));
    if (p_->pinned) {
        const auto s = p_->pinned->stats();
        std::fprintf(stderr,"M31 CPU-quants pinned profile label=%s active_last=%d calls=%zu fallback_calls=%zu quant_bytes=%zu readback_bytes=%zu budget=%zu growths=%llu disabled=%d\n",
            label ? label : "runtime",int(p_->pinned_last),p_->pinned_calls,p_->pinned_fallbacks,s.quant_capacity,s.readback_capacity,s.budget,
            (unsigned long long)s.growths,int(s.disabled_after_allocation_failure));
    }
}
std::vector<float> GpuCpuQuants::multiply(int type_int, const void * weights, size_t slot_stride,
                                        int columns, int rows, int used, int n,
                                        const std::vector<float> & input,
                                        const std::vector<int32_t> & ids) {
    std::vector<float> result;
    multiply_into(type_int,weights,slot_stride,columns,rows,used,n,input,ids,result);
    return result;
}
void GpuCpuQuants::multiply_into(int type_int, const void * weights, size_t slot_stride,
                                int columns, int rows, int used, int n,
                                const std::vector<float> & input,
                                const std::vector<int32_t> & ids, std::vector<float> & result) {
    if (&input == &result) throw std::invalid_argument("M31 CPU-quants CUDA: input/output vectors must differ");
    const char * sort_env = std::getenv("M31_GPU_EXPERT_SORT_ROUTES");
    const char * batched_env = std::getenv("M31_GPU_CPU_QUANTS_BATCHED");
    const char * tensor_env = std::getenv("M31_GPU_CPU_QUANTS_TENSOR");
    const bool tensor = n >= 32 && type_int != GGML_TYPE_NVFP4 && tensor_env && std::string(tensor_env) == "1";
    const bool batched = tensor || (n >= 32 && type_int != GGML_TYPE_NVFP4 && batched_env && std::string(batched_env) == "1");
    const bool sort_by_expert = batched || (n >= 32 && sort_env && std::string(sort_env) == "1");
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
    const size_t packed_bytes = qbytes*input_rows;
    const size_t out_count = size_t(rows)*used*n;
    bool pinned_active = false;
    if (p_->pinned) {
        pinned_active = p_->pinned->ensure(packed_bytes,out_count*sizeof(float));
        if (pinned_active) ++p_->pinned_calls;
        else ++p_->pinned_fallbacks;
        if (!p_->pinned_reported || pinned_active != p_->pinned_last) {
            const auto s = p_->pinned->stats();
            std::fprintf(stderr,"M31 CPU-quants pinned actual active=%d quant_bytes=%zu readback_bytes=%zu budget=%zu disabled=%d\n",
                int(pinned_active),s.quant_capacity,s.readback_capacity,s.budget,int(s.disabled_after_allocation_failure));
            p_->pinned_reported = true;
        }
        p_->pinned_last = pinned_active;
    }
    uint8_t * packed = nullptr;
    if (pinned_active) packed = p_->pinned->quant();
    else {
        if (p_->host_quant.size() < packed_bytes) p_->host_quant.resize(packed_bytes);
        packed = p_->host_quant.data();
    }
    gpu_host_quantize_rows(p_->host, type, size_t(columns), input.data(), input.size(), packed, packed_bytes);
    const auto quant_end = std::chrono::steady_clock::now();
    p_->quant_seconds += std::chrono::duration<double>(quant_end-quant_start).count();
    p_->reserve(p_->quant, p_->quant_capacity, packed_bytes);
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
    std::vector<int32_t> tiles, tensor_tiles, scheduled_routes;
    if (sort_by_expert) {
        int32_t maximum = 0;
        for (const auto id : ids) maximum = std::max(maximum, id);
        const auto order = prefill_route_worklist(ids, size_t(maximum)+1, used, tiled_enabled, tiled_forced);
        p_->reserve(p_->worklist, p_->worklist_capacity, order.route.size()*sizeof(int32_t));
        scheduled_routes = order.route;
        if (batched) {
            size_t begin = 0;
            for (size_t slot = 0; slot < order.histogram.size(); ++slot) {
                const int count = order.histogram[slot];
                const bool tensor_expert = tensor && tiled_enabled && (tiled_forced || count >= 8);
                const int group = tensor_expert ? 64 : 16;
                auto & target = tensor_expert ? tensor_tiles : tiles;
                for (int part = 0; part < count; part += group) {
                    target.push_back(int32_t(slot));
                    target.push_back(int32_t(begin)+part);
                    target.push_back(std::min(group,count-part));
                }
                begin += size_t(count);
            }
            if (!tiles.empty()) p_->reserve(p_->tiles,p_->tiles_capacity,tiles.size()*sizeof(int32_t));
            if (!tensor_tiles.empty()) p_->reserve(p_->tensor_tiles,p_->tensor_tiles_capacity,tensor_tiles.size()*sizeof(int32_t));
        }
    }
    const int blocks = int((out_count*8+127)/128);
    const bool broadcast = input_rows == size_t(n);
    // Construct AFTER all local upload-source vectors. On exceptions this guard
    // drains before those vectors can be destroyed, preserving async lifetimes.
    struct AsyncDrain { Impl * owner; ~AsyncDrain() { owner->drain(); } } async_drain{p_.get()};
    const auto upload_start = std::chrono::steady_clock::now();
    p_->prepare_seconds += std::chrono::duration<double>(upload_start-quant_end).count();
    p_->upload(p_->quant, packed, packed_bytes);
    p_->upload(p_->ids, dispatch_ids.data(), dispatch_ids.size()*sizeof(int32_t));
    if (sort_by_expert)
        p_->upload(p_->worklist, scheduled_routes.data(), scheduled_routes.size()*sizeof(int32_t));
    if (batched && !tiles.empty())
        p_->upload(p_->tiles,tiles.data(),tiles.size()*sizeof(int32_t));
    if (!tensor_tiles.empty())
        p_->upload(p_->tensor_tiles,tensor_tiles.data(),tensor_tiles.size()*sizeof(int32_t));
    const auto launch_start = std::chrono::steady_clock::now();
    p_->upload_seconds += std::chrono::duration<double>(launch_start-upload_start).count();
    const bool timed = p_->timing();
    if (timed) cuda_check(cudaEventRecord(p_->kernel_begin,p_->stream));
#define M31_LAUNCH(T) multiply_kernel<T><<<blocks,128,0,p_->stream>>>(static_cast<const uint8_t *>(weights), slot_stride, columns, rows, used, n, p_->quant, qbytes, broadcast, p_->ids, sort_by_expert ? p_->worklist : nullptr, p_->out, tiled_fma)
    if (batched) {
      if (!tiles.empty()) {
        const dim3 grid((rows+15)/16,unsigned(tiles.size()/3));
#define M31_BATCH(T) multiply_batched_kernel<T><<<grid,256,0,p_->stream>>>(static_cast<const uint8_t *>(weights),slot_stride,columns,rows,used,n,p_->quant,qbytes,broadcast,p_->ids,p_->worklist,p_->tiles,p_->out)
        if (type == GGML_TYPE_IQ3_XXS) { M31_BATCH(GGML_TYPE_IQ3_XXS); }
        else { M31_BATCH(GGML_TYPE_IQ4_XS); }
#undef M31_BATCH
      }
      if (!tensor_tiles.empty()) {
        const dim3 grid((rows+15)/16,unsigned(tensor_tiles.size()/3));
#define M31_TENSOR(T) multiply_tensor_kernel<T><<<grid,128,0,p_->stream>>>(static_cast<const uint8_t *>(weights),slot_stride,columns,rows,used,n,p_->quant,qbytes,broadcast,p_->worklist,p_->tensor_tiles,p_->out)
        if (type == GGML_TYPE_IQ3_XXS) { M31_TENSOR(GGML_TYPE_IQ3_XXS); }
        else { M31_TENSOR(GGML_TYPE_IQ4_XS); }
#undef M31_TENSOR
      }
    }
    else if (type == GGML_TYPE_IQ3_XXS) { M31_LAUNCH(GGML_TYPE_IQ3_XXS); }
    else if (type == GGML_TYPE_IQ4_XS) { M31_LAUNCH(GGML_TYPE_IQ4_XS); }
    else { M31_LAUNCH(GGML_TYPE_NVFP4); }
#undef M31_LAUNCH
    cuda_check(cudaGetLastError());
    if (timed) cuda_check(cudaEventRecord(p_->kernel_end,p_->stream));
    if (result.size() != out_count) result.resize(out_count);
    p_->read(pinned_active ? p_->pinned->readback() : result.data(),p_->out,out_count*sizeof(float));
    if (pinned_active) std::memcpy(result.data(),p_->pinned->readback(),out_count*sizeof(float));
    const auto result_end = std::chrono::steady_clock::now();
    p_->postlaunch_seconds += std::chrono::duration<double>(result_end-launch_start).count();
    if (timed) {
        float elapsed = 0;
        if (!p_->stream) cuda_check(cudaEventSynchronize(p_->kernel_end));
        cuda_check(cudaEventElapsedTime(&elapsed,p_->kernel_begin,p_->kernel_end));
        p_->kernel_seconds += elapsed*0.001;
        ++p_->kernel_timed_calls;
    }
    p_->roundtrip_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-quant_end).count();
    ++p_->calls;
}
}
