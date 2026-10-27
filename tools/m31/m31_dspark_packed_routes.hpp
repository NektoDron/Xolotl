#pragma once
// Isolated proposal only. Engine integration must remain behind an explicit flag.
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace strata::m31 {
struct DSparkPackedRoutesLayout {
    int64_t embd=0,width=0;int tokens=0,used=0,experts=0;
    size_t elements=0,bytes=0;
    DSparkPackedRoutesLayout(int64_t d,int n,int k,int e=128):embd(d),tokens(n),used(k),experts(e){
        if(d<=0||n<=0||n>8||k<=0||k>e||e<=0||e>128||d>(std::numeric_limits<int64_t>::max()-2*int64_t(k))/2)
            throw std::invalid_argument("M31 packed routes: invalid geometry");
        width=2*d+2*int64_t(k);
        if(uint64_t(width)>std::numeric_limits<size_t>::max()/(size_t(n)*sizeof(float)))
            throw std::invalid_argument("M31 packed routes: host byte size overflow");
        elements=size_t(width)*size_t(n);bytes=elements*sizeof(float);
    }
};
struct DSparkPackedRoutes {
    DSparkPackedRoutesLayout layout;
    ggml_tensor * tensor=nullptr;
};
inline DSparkPackedRoutes dspark_pack_routes(ggml_context * c,ggml_tensor * norm,
    ggml_tensor * residual,ggml_tensor * weights,ggml_tensor * ids,int experts=128){
    auto check=[](bool value,const char *reason){if(!value)throw std::invalid_argument(reason);};
    check(c&&norm&&residual&&weights&&ids,"M31 packed routes: missing tensor");
    check(norm->type==GGML_TYPE_F32&&residual->type==GGML_TYPE_F32&&weights->type==GGML_TYPE_F32&&ids->type==GGML_TYPE_I32,
          "M31 packed routes: F32 metadata and I32 ids required");
    check(norm->ne[1]<=8&&norm->ne[1]>0&&weights->ne[0]>0&&weights->ne[0]<=128,"M31 packed routes: invalid tensor geometry");
    DSparkPackedRoutesLayout layout(norm->ne[0],int(norm->ne[1]),int(weights->ne[0]),experts);
    for(auto * t:{norm,residual,weights,ids})check(t->ne[1]==layout.tokens&&t->ne[2]==1&&t->ne[3]==1,"M31 packed routes: expected matrices [width,n]");
    check(residual->ne[0]==layout.embd&&ids->ne[0]==layout.used,"M31 packed routes: widths differ");
    auto dense=[&](ggml_tensor *t){return ggml_is_contiguous(t)?t:ggml_cont(c,t);};
    auto encoded=ggml_cast(c,dense(ids),GGML_TYPE_F32);
    auto packed=ggml_concat(c,dense(norm),dense(residual),0);
    packed=ggml_concat(c,packed,dense(weights),0);
    packed=ggml_concat(c,packed,encoded,0);
    check(packed->type==GGML_TYPE_F32&&packed->ne[0]==layout.width&&packed->ne[1]==layout.tokens&&ggml_is_contiguous(packed),
          "M31 packed routes: invalid packed tensor");
    return {layout,packed};
}
struct DSparkPackedRoutesHost {
    std::vector<float> norm,residual,weights;
    std::vector<int32_t> ids;
};
inline void dspark_unpack_routes(const DSparkPackedRoutesLayout & l,
    const std::vector<float> & packed,DSparkPackedRoutesHost & output){
    if(packed.size()!=l.elements||&packed==&output.norm||&packed==&output.residual||&packed==&output.weights)
        throw std::invalid_argument("M31 packed routes: invalid host buffer/alias");
    // Reject malformed IDs before mutating the previous successful result.
    for(int token=0;token<l.tokens;++token)for(int rank=0;rank<l.used;++rank){
        const float id=packed[size_t(token)*size_t(l.width)+size_t(2*l.embd+l.used+rank)];
        if(!std::isfinite(id)||id<0||id>=float(l.experts)||id!=float(int32_t(id)))
            throw std::invalid_argument("M31 packed routes: nonintegral/out-of-range id");
    }
    output.norm.resize(size_t(l.embd)*size_t(l.tokens));output.residual.resize(output.norm.size());
    output.weights.resize(size_t(l.used)*size_t(l.tokens));output.ids.resize(output.weights.size());
    for(int token=0;token<l.tokens;++token){
        const auto *row=packed.data()+size_t(token)*size_t(l.width);
        std::memcpy(output.norm.data()+size_t(token)*size_t(l.embd),row,size_t(l.embd)*sizeof(float));
        std::memcpy(output.residual.data()+size_t(token)*size_t(l.embd),row+size_t(l.embd),size_t(l.embd)*sizeof(float));
        std::memcpy(output.weights.data()+size_t(token)*size_t(l.used),row+size_t(2*l.embd),size_t(l.used)*sizeof(float));
        for(int rank=0;rank<l.used;++rank)output.ids[size_t(token)*size_t(l.used)+size_t(rank)]=int32_t(row[size_t(2*l.embd+l.used+rank)]);
    }
}
inline void dspark_read_packed_routes(const DSparkPackedRoutes & p,
    std::vector<float> & scratch,DSparkPackedRoutesHost & output){
    if(!p.tensor||p.tensor->type!=GGML_TYPE_F32||!ggml_is_contiguous(p.tensor)||ggml_nbytes(p.tensor)!=p.layout.bytes)
        throw std::invalid_argument("M31 packed routes: invalid output tensor");
    scratch.resize(p.layout.elements);
    ggml_backend_tensor_get(p.tensor,scratch.data(),0,p.layout.bytes); // Exactly one synchronous get.
    dspark_unpack_routes(p.layout,scratch,output);
}
} // namespace strata::m31
