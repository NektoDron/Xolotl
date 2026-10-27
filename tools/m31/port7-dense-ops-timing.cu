#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
double dense_ops_graph_us(ggml_backend_t backend, ggml_cgraph * graph, int repeats);
double dense_ops_graph_us(ggml_backend_t backend, ggml_cgraph * graph, int repeats) {
    auto & context=*static_cast<ggml_backend_cuda_context *>(backend->context);
    auto stream=context.stream(); cudaEvent_t begin=nullptr,end=nullptr;
    auto ck=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    try {
        ck(cudaEventRecord(begin,stream));
        for(int i=0;i<repeats;++i)
            if(ggml_backend_graph_compute_async(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("ops graph submission failed");
        ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,begin,end));
        ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return ms*1000/repeats;
    }catch(...){cudaStreamSynchronize(stream);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);throw;}
}
