#include "gpu_prepared_pair.hpp"
#include "cpu_gpu_host_ops.hpp"
#include "prefill_route_worklist.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::m31 {
namespace {
void prepared_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("M31 prepared pair: ")+cudaGetErrorString(result));
}
#include "gpu_cpu_quants_batched.cuh"
__global__ void prepared_oai(float * gate,const float * up,size_t count,int rows,
                             const int32_t * ids,const float * gs,const float * us) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const int expert = ids[i/size_t(rows)];
    const float gv = gs ? __fmul_rn(gate[i],gs[expert]) : gate[i];
    const float uv = us ? __fmul_rn(up[i],us[expert]) : up[i];
    const float a = 7.f < gv ? 7.f : gv;
    const float b = uv < -7.f ? -7.f : (7.f < uv ? 7.f : uv);
    const float glu = __fdiv_rn(a,__fadd_rn(1.f,expf(__fmul_rn(1.702f,-a))));
    gate[i] = __fmul_rn(glu,__fadd_rn(b,1.f));
}
void prepared_require(bool value,const char * message) {
    if (!value) throw std::runtime_error(std::string("M31 prepared pair: ")+message);
}
}
struct GpuPreparedPair::Impl {
    GpuHostExecutor & host;
    uint8_t * quant = nullptr;
    int32_t * ids = nullptr, * routes = nullptr, * device_tiles = nullptr;
    float * output = nullptr;
    float * scales = nullptr;
    size_t quant_capacity = 0, ids_capacity = 0, routes_capacity = 0, tiles_capacity = 0, output_capacity = 0, scales_capacity = 0;
    std::vector<uint8_t> host_quant;
    std::vector<int32_t> tiles, histogram, physical_slots;
    std::vector<uint8_t> submitted;
    int gate_type = 0, up_type = 0, columns = 0, rows = 0, used = 0, n = 0;
    size_t quant_row_bytes = 0, out_count = 0;
    bool broadcast = false, active = false;
    explicit Impl(GpuHostExecutor & executor):host(executor) {}
    template<class T> void reserve(T * & pointer,size_t & capacity,size_t bytes) {
        if (capacity >= bytes) return;
        prepared_cuda(cudaFree(pointer)); pointer = nullptr; capacity = 0;
        prepared_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer),bytes)); capacity = bytes;
    }
    void drain() noexcept { if (active) cudaStreamSynchronize(nullptr); active = false; }
    ~Impl() {
        drain();
        cudaFree(quant); cudaFree(ids); cudaFree(routes); cudaFree(device_tiles); cudaFree(output); cudaFree(scales);
    }
};
GpuPreparedPair::GpuPreparedPair(GpuHostExecutor & host):p_(std::make_unique<Impl>(host)) {}
GpuPreparedPair::~GpuPreparedPair() = default;
bool GpuPreparedPair::supported(int gate,int up,int n) {
    return n >= 32 && (gate == GGML_TYPE_IQ3_XXS || gate == GGML_TYPE_IQ4_XS) &&
                     (up == GGML_TYPE_IQ3_XXS || up == GGML_TYPE_IQ4_XS);
}
size_t GpuPreparedPair::scratch_bytes() const {
    return p_->quant_capacity+p_->ids_capacity+p_->routes_capacity+p_->tiles_capacity+p_->output_capacity+p_->scales_capacity;
}
void GpuPreparedPair::abort() noexcept { p_->drain(); }
void GpuPreparedPair::prepare(int gate,int up,int columns,int rows,int used,int n,
                             const std::vector<float> & input,const std::vector<int32_t> & ids) {
    prepared_require(!p_->active,"previous batch requires read_into/abort");
    prepared_require(ggml_cpu_has_avx2() && ggml_cpu_has_fma(),"CPU quant contract requires AVX2/FMA");
    prepared_require(supported(gate,up,n) && columns > 0 && columns%256 == 0 && rows > 0 && used > 0 &&
        used <= 4096 && n <= 8192 && ids.size() == size_t(used)*n &&
        (input.size() == size_t(columns)*n || input.size() == size_t(columns)*n*used),"unsupported shape/type");
    int32_t maximum = -1;
    for (auto id:ids) { prepared_require(id >= 0 && id < 1048576,"invalid logical expert"); maximum = std::max(maximum,id); }
    const char * mode = std::getenv("GGML_CPU_TILED_MM");
    const char * force = std::getenv("GGML_CPU_TILED_MM_FORCE");
    const bool tiled = !mode || std::atoi(mode) != 0, forced = force && std::atoi(force) == 1;
    const auto order = prefill_route_worklist(ids,size_t(maximum)+1,used,tiled,forced);
    p_->gate_type = gate; p_->up_type = up; p_->columns = columns; p_->rows = rows; p_->used = used; p_->n = n;
    p_->broadcast = input.size() == size_t(columns)*n;
    p_->out_count = size_t(rows)*ids.size();
    prepared_require(p_->out_count <= std::numeric_limits<size_t>::max()/(2*sizeof(float)),"output size overflow");
    p_->histogram = order.histogram; p_->submitted.assign(order.histogram.size(),0); p_->tiles.clear();
    p_->physical_slots.assign(order.histogram.size(),-1);
    size_t begin = 0;
    for (size_t expert = 0; expert < order.histogram.size(); ++expert) {
        const int count = order.histogram[expert];
        for (int part = 0; part < count; part += 16) {
            p_->tiles.push_back(int32_t(expert)); p_->tiles.push_back(int32_t(begin)+part);
            p_->tiles.push_back(std::min(16,count-part));
        }
        begin += size_t(count);
    }
    prepared_require(p_->tiles.size()/3 <= 65535,"too many route tiles");
    const auto layout = gpu_host_quant_layout(ggml_type(gate),size_t(columns),input.size()/columns);
    prepared_require(layout.activation_type == GGML_TYPE_Q8_K &&
        ggml_get_type_traits_cpu(ggml_type(up))->vec_dot_type == GGML_TYPE_Q8_K,"activation quant mismatch");
    p_->quant_row_bytes = layout.row_bytes;
    if (p_->host_quant.size() < layout.bytes) p_->host_quant.resize(layout.bytes);
    gpu_host_quantize_rows(p_->host,ggml_type(gate),size_t(columns),input.data(),input.size(),p_->host_quant.data(),layout.bytes);
    std::vector<int32_t> dispatch(ids); dispatch.insert(dispatch.end(),order.tiled.begin(),order.tiled.end());
    p_->reserve(p_->quant,p_->quant_capacity,layout.bytes);
    p_->reserve(p_->ids,p_->ids_capacity,dispatch.size()*sizeof(int32_t));
    p_->reserve(p_->routes,p_->routes_capacity,order.route.size()*sizeof(int32_t));
    p_->reserve(p_->device_tiles,p_->tiles_capacity,p_->tiles.size()*sizeof(int32_t));
    p_->reserve(p_->output,p_->output_capacity,p_->out_count*2*sizeof(float));
    prepared_cuda(cudaMemcpy(p_->quant,p_->host_quant.data(),layout.bytes,cudaMemcpyHostToDevice));
    prepared_cuda(cudaMemcpy(p_->ids,dispatch.data(),dispatch.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    prepared_cuda(cudaMemcpy(p_->routes,order.route.data(),order.route.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    p_->active = true;
}
void GpuPreparedPair::enqueue(const void * gate,const void * up,size_t stride,
                             const std::vector<int32_t> & logical,const std::vector<int32_t> & slots,
                             ggml_backend_event_t ready) {
    prepared_require(p_->active && gate && up && !logical.empty() && logical.size() == slots.size(),"invalid enqueue");
    prepared_require(stride >= std::max(ggml_row_size(ggml_type(p_->gate_type),p_->columns),
        ggml_row_size(ggml_type(p_->up_type),p_->columns))*size_t(p_->rows),"weight slot too short");
    std::vector<int32_t> remap(p_->histogram.size(),-1), selected;
    for (size_t i = 0; i < logical.size(); ++i) {
        const auto id = logical[i];
        prepared_require(id >= 0 && size_t(id) < p_->histogram.size() && p_->histogram[size_t(id)] &&
            !p_->submitted[size_t(id)] && remap[size_t(id)] == -1 && slots[i] >= 0,"duplicate/unknown logical expert or invalid slot");
        remap[size_t(id)] = slots[i];
        for (size_t j = 0; j < i; ++j) prepared_require(slots[j] != slots[i],"aliased physical slots in group");
        for (auto previous:p_->physical_slots) prepared_require(previous != slots[i],"physical slot reused across groups");
    }
    for (size_t tile = 0; tile < p_->tiles.size(); tile += 3) {
        const auto slot = remap[size_t(p_->tiles[tile])];
        if (slot < 0) continue;
        selected.push_back(slot); selected.push_back(p_->tiles[tile+1]); selected.push_back(p_->tiles[tile+2]);
    }
    if (ready) {
        // This bridge is deliberately tied to the pinned ggml CUDA backend,
        // whose event->context is cudaEvent_t (ggml-cuda.cu event_record/wait).
        int current = -1; prepared_cuda(cudaGetDevice(&current));
        const auto registry = ggml_backend_dev_backend_reg(ready->device);
        prepared_require(registry && std::string(ggml_backend_reg_name(registry)) == "CUDA" &&
            std::string(ggml_backend_dev_name(ready->device)) == "CUDA"+std::to_string(current) && ready->context,
            "ready event must belong to the current CUDA device");
        prepared_cuda(cudaStreamWaitEvent(nullptr,static_cast<cudaEvent_t>(ready->context),0));
    }
    // Synchronous metadata copy also protects this reusable descriptor buffer
    // from an earlier raw-stream consumer. Weight DMA uses a different stream.
    prepared_cuda(cudaMemcpy(p_->device_tiles,selected.data(),selected.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    const dim3 grid((p_->rows+15)/16,unsigned(selected.size()/3));
#define M31_PREPARED_LAUNCH(T,W,O) multiply_batched_kernel<T><<<grid,256>>>(static_cast<const uint8_t *>(W),stride,p_->columns,p_->rows,p_->used,p_->n,p_->quant,p_->quant_row_bytes,p_->broadcast,p_->ids,p_->routes,p_->device_tiles,O)
    if (p_->gate_type == GGML_TYPE_IQ3_XXS) { M31_PREPARED_LAUNCH(GGML_TYPE_IQ3_XXS,gate,p_->output); }
    else { M31_PREPARED_LAUNCH(GGML_TYPE_IQ4_XS,gate,p_->output); }
    prepared_cuda(cudaGetLastError());
    if (p_->up_type == GGML_TYPE_IQ3_XXS) { M31_PREPARED_LAUNCH(GGML_TYPE_IQ3_XXS,up,p_->output+p_->out_count); }
    else { M31_PREPARED_LAUNCH(GGML_TYPE_IQ4_XS,up,p_->output+p_->out_count); }
#undef M31_PREPARED_LAUNCH
    prepared_cuda(cudaGetLastError());
    for (size_t i = 0; i < logical.size(); ++i) {
        p_->submitted[size_t(logical[i])] = 1;
        p_->physical_slots[size_t(logical[i])] = slots[i];
    }
}
void GpuPreparedPair::read_into(std::vector<float> & gate,std::vector<float> & up) {
    prepared_require(p_->active && &gate != &up,"invalid read_into or aliased output vectors");
    for (size_t expert = 0; expert < p_->histogram.size(); ++expert)
        prepared_require(!p_->histogram[expert] || p_->submitted[expert],"not all expert groups have been submitted");
    if (gate.size() != p_->out_count) gate.resize(p_->out_count);
    if (up.size() != p_->out_count) up.resize(p_->out_count);
    prepared_cuda(cudaMemcpy(gate.data(),p_->output,p_->out_count*sizeof(float),cudaMemcpyDeviceToHost));
    prepared_cuda(cudaMemcpy(up.data(),p_->output+p_->out_count,p_->out_count*sizeof(float),cudaMemcpyDeviceToHost));
    p_->active = false;
}
void GpuPreparedPair::read_oai_into(std::vector<float> & hidden,const std::vector<float> & gs,const std::vector<float> & us) {
    prepared_require(p_->active,"read_oai_into outside a batch");
    for (size_t expert = 0; expert < p_->histogram.size(); ++expert)
        prepared_require(!p_->histogram[expert] || p_->submitted[expert],"not all expert groups have been submitted");
    const size_t experts = p_->histogram.size();
    prepared_require((gs.empty() || gs.size() >= experts) && (us.empty() || us.size() >= experts),"scale array too short");
    if (!gs.empty() || !us.empty()) p_->reserve(p_->scales,p_->scales_capacity,experts*2*sizeof(float));
    if (!gs.empty()) prepared_cuda(cudaMemcpy(p_->scales,gs.data(),experts*sizeof(float),cudaMemcpyHostToDevice));
    if (!us.empty()) prepared_cuda(cudaMemcpy(p_->scales+experts,us.data(),experts*sizeof(float),cudaMemcpyHostToDevice));
    prepared_oai<<<unsigned((p_->out_count+255)/256),256>>>(p_->output,p_->output+p_->out_count,p_->out_count,p_->rows,
        p_->ids,gs.empty() ? nullptr : p_->scales,us.empty() ? nullptr : p_->scales+experts);
    prepared_cuda(cudaGetLastError());
    if (hidden.size() != p_->out_count) hidden.resize(p_->out_count);
    prepared_cuda(cudaMemcpy(hidden.data(),p_->output,p_->out_count*sizeof(float),cudaMemcpyDeviceToHost));
    p_->active = false;
}
}
