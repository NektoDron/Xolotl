#include "port7_dense_mmv_probe.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
static void ck(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(std::string("MMV CUDA: ")+cudaGetErrorString(value));
}
struct Batch {
    ggml_backend_t backend;
    ggml_context * metadata = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * output = nullptr;
    Batch(ggml_backend_t b, ggml_tensor * weight, int columns, bool marked = false) : backend(b) {
        metadata = ggml_init({2*1024*1024,nullptr,true});
        input = ggml_new_tensor_3d(metadata,GGML_TYPE_F32,weight->ne[0],columns,weight->ne[2]);
        ggml_set_input(input);
        output = ggml_mul_mat(metadata,weight,input); ggml_set_output(output);
        if (marked) {
#ifdef GGML_CUDA_HAS_EXACT_MMV_MARKER
            if (!ggml_cuda_mul_mat_set_exact_mmv(output) || !ggml_cuda_mul_mat_is_exact_mmv(output))
                throw std::runtime_error("supported projection marker rejected");
#else
            throw std::runtime_error("marker API not compiled");
#endif
        }
        graph = ggml_new_graph(metadata); ggml_build_forward_expand(graph,output);
        buffer = ggml_backend_alloc_ctx_tensors(metadata,backend);
        if (!buffer) throw std::runtime_error("MMV input/output allocation failed");
    }
    ~Batch() {
        ggml_backend_synchronize(backend); ggml_backend_cuda_graph_cache_clear(backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (metadata) ggml_free(metadata);
    }
    void compute() {
        if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MMV stock graph failed");
    }
    std::vector<float> read() const {
        std::vector<float> data(size_t(ggml_nelements(output)));
        ggml_backend_tensor_get(output,data.data(),0,data.size()*sizeof(float));
        return data;
    }
};
static void mmv(ggml_backend_cuda_context & context, const ggml_tensor * weight,
         const ggml_tensor * input, ggml_tensor * output) {
    if (ggml_is_quantized(weight->type)) ggml_cuda_mul_mat_vec_q(context,weight,input,nullptr,output);
    else ggml_cuda_mul_mat_vec_f(context,weight,input,nullptr,output);
}
template<class Function> double time_cuda(cudaStream_t stream, Function operation) {
    cudaEvent_t begin = nullptr, end = nullptr;
    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
    try {
        constexpr int repeats = 5;
        operation(); ck(cudaStreamSynchronize(stream));
        ck(cudaEventRecord(begin,stream));
        for (int k = 0; k < repeats; ++k) operation();
        ck(cudaEventRecord(end,stream)); ck(cudaEventSynchronize(end));
        float elapsed = 0; ck(cudaEventElapsedTime(&elapsed,begin,end));
        ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
        return elapsed*1000/repeats;
    } catch (...) {
        cudaStreamSynchronize(stream);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        throw;
    }
}
static DenseMmvProbeResult compare(const std::vector<float> & expected, const std::vector<float> & actual) {
    if (expected.size() != actual.size()) throw std::runtime_error("MMV output length mismatch");
    DenseMmvProbeResult result; result.values = actual.size(); double sq = 0;
    for (size_t k = 0; k < actual.size(); ++k) {
        if (!std::isfinite(actual[k]) || !std::isfinite(expected[k])) throw std::runtime_error("MMV nonfinite output");
        if (std::memcmp(&expected[k],&actual[k],sizeof(float))) ++result.different;
        const double d = double(actual[k])-expected[k];
        result.max_abs = std::max(result.max_abs,std::abs(d)); sq += d*d;
    }
    result.rms = std::sqrt(sq/std::max<size_t>(1,actual.size()));
    return result;
}
}

std::vector<DenseMmvProbeResult> dense_mmv_probe(ggml_backend_t backend, ggml_tensor * weight) {
    auto & context = *static_cast<ggml_backend_cuda_context *>(backend->context);
    const int cc = ggml_cuda_info().devices[context.device].cc;
    const int warp = ggml_cuda_info().devices[context.device].warp_size;
    std::vector<DenseMmvProbeResult> results;
    for (int columns : {2,3,4,8}) {
        Batch scalar(backend,weight,1), batch(backend,weight,columns);
        std::unique_ptr<Batch> marked;
#ifdef GGML_CUDA_HAS_EXACT_MMV_MARKER
        if (columns <= 4) marked = std::make_unique<Batch>(backend,weight,columns,true);
#endif
        const bool stock_mmv = ggml_is_quantized(weight->type) ? ggml_cuda_should_use_mmvq(weight->type,cc,columns)
            : ggml_cuda_should_use_mmvf(weight->type,cc,warp,weight->ne,weight->nb,columns);
        for (int family = 0; family < 2; ++family) {
            std::vector<float> input(size_t(weight->ne[0])*columns);
            uint32_t state = 123456789;
            for (size_t k = 0; k < input.size(); ++k) {
                state = state*1664525u+1013904223u;
                float value = float(int((state>>8)&65535)-32768)/8192.f;
                if (family) value = k%32==0 ? value*32 : value*0.00003125f;
                input[k] = value;
            }
            std::vector<float> reference;
            reference.reserve(size_t(weight->ne[1])*columns);
            for (int k = 0; k < columns; ++k) {
                ggml_backend_tensor_set(scalar.input,input.data()+size_t(k)*weight->ne[0],0,size_t(weight->ne[0])*4);
                scalar.compute(); auto row = scalar.read(); reference.insert(reference.end(),row.begin(),row.end());
            }
            ggml_backend_tensor_set(batch.input,input.data(),0,input.size()*4);
            for (int k = 0; k < 4; ++k) batch.compute();
            auto stock = compare(reference,batch.read());
            mmv(context,weight,batch.input,batch.output); ggml_backend_synchronize(backend);
            auto forced = compare(reference,batch.read());
            DenseMmvProbeResult marked_result;
            if (marked) {
                ggml_backend_tensor_set(marked->input,input.data(),0,input.size()*4);
                for (int k = 0; k < 4; ++k) marked->compute();
                marked_result = compare(reference,marked->read());
                marked_result.marked_mmv = true;
            }
            const double scalar_us = time_cuda(context.stream(),[&] {
                for (int k = 0; k < columns; ++k) {
                    auto x = *batch.input, y = *batch.output;
                    x.ne[1] = y.ne[1] = 1;
                    x.nb[2] = x.nb[3] = x.nb[1]; y.nb[2] = y.nb[3] = y.nb[1];
                    x.data = static_cast<char *>(batch.input->data)+size_t(k)*batch.input->nb[1];
                    y.data = static_cast<char *>(batch.output->data)+size_t(k)*batch.output->nb[1];
                    mmv(context,weight,&x,&y);
                }
            });
            const double batch_us = time_cuda(context.stream(),[&] { mmv(context,weight,batch.input,batch.output); });
            const double stock_us = time_cuda(context.stream(),[&] {
                if (ggml_backend_graph_compute_async(backend,batch.graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("MMV timed graph failed");
            });
            const double marked_us = marked ? time_cuda(context.stream(),[&] {
                if (ggml_backend_graph_compute_async(backend,marked->graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("MMV marked timed graph failed");
            }) : 0;
            for (auto * result : {&stock,&forced,&marked_result}) {
                result->columns = columns; result->family = family;
                result->stock_uses_mmv = stock_mmv;
                result->scalar_mmv_us = scalar_us; result->batch_mmv_us = batch_us; result->stock_graph_us = stock_us;
                result->marked_graph_us = marked_us;
            }
            forced.forced_mmv = true;
            results.push_back(stock); results.push_back(forced);
            if (marked) results.push_back(marked_result);
        }
    }
    return results;
}

uint64_t dense_mmv_msa_isolation_probe(ggml_backend_t backend) {
    auto * metadata = ggml_init({1024*1024,nullptr,true});
    auto * score_w = ggml_new_tensor_3d(metadata,GGML_TYPE_F32,128,256,4);
    auto * projection_w = ggml_new_tensor_2d(metadata,GGML_TYPE_F32,128,16);
    auto * buffer = ggml_backend_alloc_ctx_tensors(metadata,backend);
    if (!buffer) { ggml_free(metadata); throw std::runtime_error("MSA isolation weight allocation"); }
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    try {
        for (auto * w : {score_w,projection_w}) {
            std::vector<float> values(size_t(ggml_nelements(w)));
            for (size_t k = 0; k < values.size(); ++k) values[k] = float(int(k*37%233)-116)/29.f;
            ggml_backend_tensor_set(w,values.data(),0,values.size()*4);
        }
        uint64_t differences = 0;
        {
            Batch score(backend,score_w,4);
            std::vector<float> input(size_t(ggml_nelements(score.input)));
            for (size_t k = 0; k < input.size(); ++k) input[k] = float(int(k*17%149)-74)/31.f;
            ggml_backend_tensor_set(score.input,input.data(),0,input.size()*4);
            for (int k = 0; k < 4; ++k) score.compute();
            const auto reference = score.read();
#ifdef GGML_CUDA_HAS_EXACT_MMV_MARKER
            if (ggml_cuda_mul_mat_is_exact_mmv(score.output)) throw std::runtime_error("fresh MSA score marked");
            auto rejected = *score.output;
            std::array<unsigned char,sizeof(ggml_tensor)> before{};
            std::memcpy(before.data(),&rejected,before.size());
            if (ggml_cuda_mul_mat_set_exact_mmv(&rejected) || std::memcmp(before.data(),&rejected,sizeof(rejected)))
                throw std::runtime_error("MSA channel node accepted/mutated by marker");
            {
                Batch projection(backend,projection_w,4,true);
                ggml_backend_tensor_set(projection.input,input.data(),0,ggml_nbytes(projection.input));
                for (int k = 0; k < 4; ++k) projection.compute();
            }
#endif
            // Remove the old executable: the second score must choose its stock
            // dispatcher anew, not merely replay a pre-marker capture.
            ggml_backend_cuda_graph_cache_clear(backend);
            for (int k = 0; k < 4; ++k) score.compute();
            differences = compare(reference,score.read()).different;
        }
        ggml_backend_synchronize(backend); ggml_backend_buffer_free(buffer); ggml_free(metadata);
        return differences;
    } catch (...) {
        ggml_backend_synchronize(backend); ggml_backend_cuda_graph_cache_clear(backend);
        ggml_backend_buffer_free(buffer); ggml_free(metadata); throw;
    }
}
