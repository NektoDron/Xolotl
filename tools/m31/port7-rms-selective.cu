#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/binbcast.cuh"
#include <stdexcept>

// Exercise the stock norm-only fusion entry point, independently of the broad
// CUDA graph matcher. The library, reduction geometry and FP flags are unchanged.
static void run(ggml_backend_t backend, ggml_tensor * norm, ggml_tensor * mul, bool fused) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if(fused) ggml_cuda_op_rms_norm_fused(ctx,norm,mul);
    else {ggml_cuda_op_rms_norm(ctx,norm);ggml_cuda_op_mul(ctx,mul);}
}
void selective_rms_compute(ggml_backend_t backend,ggml_tensor * norm,ggml_tensor * mul,bool fused) {
    run(backend,norm,mul,fused);ggml_backend_synchronize(backend);
}
double selective_rms_us(ggml_backend_t backend,ggml_tensor * norm,ggml_tensor * mul,bool fused) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    cudaStream_t stream=ctx.stream();cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t a,b;
    auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    check(cudaStreamSynchronize(stream));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    run(backend,norm,mul,fused);check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    for(int i=0;i<8;++i)check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(a,stream));
    for(int i=0;i<256;++i)check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));
    float ms=0;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(a));check(cudaEventDestroy(b));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));return double(ms)*1000/256;
}
