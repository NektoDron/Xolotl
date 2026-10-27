#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::m31 {
// Return nullptr when the two original role spans cannot be reinterpreted as a
// single [columns,2*ff,slots] weight tensor. No weights are copied/repacked and
// no vendor kernel changes are needed. Caller excludes external gate/up scales.
// The first half of each expert is gate, the second half up. CUDA SWIGLU_OAI's
// src1=null form natively handles this layout with row stride 2*ff.
inline ggml_tensor * native_grouped_gate_up(ggml_context *context,
        ggml_tensor *gate,ggml_tensor *up,ggml_tensor *input,ggml_tensor *ids,
        bool force_q8_activations=false) {
    if(!context||!gate||!up||!input||!ids||gate->type!=up->type||
       gate->ne[0]!=up->ne[0]||gate->ne[1]!=up->ne[1]||gate->ne[2]!=up->ne[2]||
       gate->ne[3]!=1||up->ne[3]!=1||gate->ne[1]<=0||
       gate->ne[1]>std::numeric_limits<int64_t>::max()/2||
       gate->nb[0]!=up->nb[0]||gate->nb[1]!=up->nb[1]||gate->nb[2]!=up->nb[2]||
       !gate->view_src||gate->view_src!=up->view_src||gate->buffer!=up->buffer||
       !gate->data||!up->data)return nullptr;
    const size_t row=ggml_row_size(gate->type,gate->ne[0]);
    if(gate->nb[1]!=row||size_t(gate->ne[1])>std::numeric_limits<size_t>::max()/row)return nullptr;
    const size_t half=row*size_t(gate->ne[1]);
    if(half>std::numeric_limits<size_t>::max()/2||2*half>gate->nb[2]||
       gate->view_offs>std::numeric_limits<size_t>::max()-half||
       up->view_offs!=gate->view_offs+half||
       static_cast<const uint8_t*>(up->data)!=static_cast<const uint8_t*>(gate->data)+half)
        return nullptr;
    auto *joined=ggml_new_tensor_3d(context,gate->type,gate->ne[0],2*gate->ne[1],gate->ne[2]);
    joined->nb[2]=gate->nb[2];joined->nb[3]=gate->nb[3];
    const size_t available=ggml_nbytes(gate->view_src);
    if(gate->view_offs>available||ggml_nbytes(joined)>available-gate->view_offs)return nullptr;
    joined->view_src=gate->view_src;joined->view_offs=gate->view_offs;
    if(ggml_backend_view_init(joined)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("M31 native GU: borrowed concatenated weight view initialization failed");
    ggml_set_name(joined,"m31.native_gate_up.weight");
    auto *products=ggml_mul_mat_id(context,joined,input,ids);
    if(force_q8_activations)ggml_prec_set_src(products,GGML_PREC_Q8,1);
    ggml_set_name(products,"m31.native_gate_up.products");
    auto *activation=ggml_swiglu_oai(context,products,nullptr,1.702f,7.f);
    ggml_set_name(activation,"m31.native_gate_up.activation");
    return activation;
}
} // namespace strata::m31
