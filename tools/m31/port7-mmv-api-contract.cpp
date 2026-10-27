#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static int checks = 0;
static void check(bool value,const char * reason) {
    if (!value) throw std::runtime_error(reason);
    ++checks;
}
static void rejected(ggml_tensor * tensor) {
    std::array<unsigned char,sizeof(ggml_tensor)> before{};
    if (tensor) std::memcpy(before.data(),tensor,before.size());
    check(!ggml_cuda_mul_mat_set_exact_mmv(tensor),"unsupported node accepted");
    if (tensor) check(!std::memcmp(before.data(),tensor,before.size()),"rejection mutated node");
}
int main() {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    try {
        // Metadata only: no CUDA device/backend/runtime initialization.
        context = ggml_init({8*1024*1024,nullptr,true});
        check(context != nullptr,"metadata allocation");
        std::vector<unsigned char> weight_bytes(4*1024*1024+64);
        auto * weight_base = reinterpret_cast<unsigned char *>((reinterpret_cast<uintptr_t>(weight_bytes.data())+63)&~uintptr_t(63));
        weights_buffer = ggml_backend_cpu_buffer_from_ptr(weight_base,4*1024*1024);
        check(weights_buffer != nullptr,"CPU immutable weight wrapper");
        ggml_backend_buffer_set_usage(weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto bind = [&](ggml_tensor * weight) {
            check(ggml_backend_tensor_alloc(weights_buffer,weight,weight_base) == GGML_STATUS_SUCCESS,
                "bind read-only weight metadata");
            return weight;
        };
        rejected(nullptr); check(!ggml_cuda_mul_mat_is_exact_mmv(nullptr),"null getter");
        for (auto type : {GGML_TYPE_F32,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0}) for (int k : {1,2,3,4}) {
            auto * w = bind(ggml_new_tensor_2d(context,type,6144,128));
            auto * x = ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,k);
            auto * node = ggml_mul_mat(context,w,x); ggml_prec_set_acc(node,GGML_PREC_F32);
            check(!ggml_cuda_mul_mat_is_exact_mmv(node),"fresh node marked");
            const auto params = std::array<int32_t,2>{node->op_params[0],node->op_params[1]};
            check(ggml_cuda_mul_mat_set_exact_mmv(node),"supported projection rejected");
            check(ggml_cuda_mul_mat_is_exact_mmv(node),"marker getter");
            check((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,"marker lacks fusion output fence");
            check(node->op_params[0] == params[0] && node->op_params[1] == params[1],"precision/hint modified");
            std::array<unsigned char,sizeof(ggml_tensor)> before{};std::memcpy(before.data(),node,before.size());
            check(ggml_cuda_mul_mat_set_exact_mmv(node),"idempotent marker rejected");
            check(!std::memcmp(before.data(),node,before.size()),"idempotent marker mutated node");
        }
        auto * w = bind(ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,128));
        for (int k : {5,8}) rejected(ggml_mul_mat(context,w,ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,k)));
        for (auto type : {GGML_TYPE_F16,GGML_TYPE_Q4_K})
            rejected(ggml_mul_mat(context,bind(ggml_new_tensor_2d(context,type,6144,128)),ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,4)));
        // Actual engine score geometry: CAST(index KV F16→F32)[128,nk] x
        // flattened index heads[128,4], all ne2=1. This is a computed operand,
        // not an immutable projection weight, even before graph allocation.
        auto * index_kv = ggml_new_tensor_2d(context,GGML_TYPE_F16,128,256);
        auto * computed_keys = ggml_cast(context,index_kv,GGML_TYPE_F32);
        rejected(ggml_mul_mat(context,computed_keys,ggml_new_tensor_2d(context,GGML_TYPE_F32,128,4)));
        rejected(ggml_mul_mat(context,ggml_new_tensor_2d(context,GGML_TYPE_F32,128,256),
                            ggml_new_tensor_2d(context,GGML_TYPE_F32,128,4)));
        // In particular, an MSA score GEMM with RHS4 and channel-axis batching
        // must not accidentally acquire projection dispatch.
        auto * score_w = ggml_new_tensor_3d(context,GGML_TYPE_F32,128,256,4);
        auto * score_x = ggml_new_tensor_3d(context,GGML_TYPE_F32,128,4,4);
        rejected(ggml_mul_mat(context,score_w,score_x));
        auto * padded = ggml_new_tensor_2d(context,GGML_TYPE_F32,6176,4);
        auto * strided = ggml_view_2d(context,padded,6144,4,padded->nb[1],0);
        rejected(ggml_mul_mat(context,w,strided));
        auto * hinted = ggml_mul_mat(context,w,ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,4));
        ggml_mul_mat_set_hint(hinted,GGML_HINT_SRC0_IS_HADAMARD); rejected(hinted);
        auto * foreign = ggml_mul_mat(context,w,ggml_new_tensor_2d(context,GGML_TYPE_F32,6144,4));
        foreign->op_params[15] = 17; rejected(foreign);
        rejected(ggml_add(context,ggml_new_tensor_2d(context,GGML_TYPE_F32,128,4),ggml_new_tensor_2d(context,GGML_TYPE_F32,128,4)));
        ggml_backend_buffer_free(weights_buffer); weights_buffer = nullptr;
        ggml_free(context); context = nullptr;
        std::printf("{\"pass\":true,\"checks\":%d,\"scope\":\"CPU metadata-only marker API boundaries, idempotence, precision/hint preservation, MSA-channel and unsupported-shape rejection; no CUDA initialization\"}\n",checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"MMV marker API contract: %s\n",error.what());
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (context) ggml_free(context);
        return 1;
    }
}
