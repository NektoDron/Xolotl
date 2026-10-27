#pragma once
// Source-only proposal: immutable exact F32 maps replace only the n1 idx->tok/row
// expansion. Sorted picks, KV/mask gathers and flash attention remain caller-owned.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
class MsaSortedLookupMaps {
    ggml_context * context_=nullptr;
    ggml_backend_buffer_t buffer_=nullptr;
    ggml_tensor * token_=nullptr,*row_=nullptr;
    int64_t capacity_=0,block_=0,heads_=0;
    static void require(bool ok,const char * why){if(!ok)throw std::invalid_argument(why);}
    void release()noexcept{if(buffer_)ggml_backend_buffer_free(buffer_);if(context_)ggml_free(context_);buffer_=nullptr;context_=nullptr;}
public:
    struct Indices {ggml_tensor * token,*row;};
    // capacity is the padded KV capacity, not the current logical position.
    // Keep this owner alive until ALL borrowing graph/capture owners are destroyed.
    MsaSortedLookupMaps(ggml_backend_t backend,int64_t capacity,int64_t block,int64_t index_heads,int64_t kv_heads)
        :capacity_(capacity),block_(block),heads_(index_heads){
        static_assert(std::numeric_limits<float>::radix==2&&std::numeric_limits<float>::digits==24);
        require(backend&&capacity>0&&capacity<=(1LL<<20)&&block>0&&capacity%block==0,"invalid lookup capacity/block");
        require(index_heads>0&&index_heads==kv_heads&&capacity*index_heads<=(1LL<<24),"lookup requires equal index/KV heads and exactly representable F32 row IDs");
        try{
            context_=ggml_init({16*1024,nullptr,true});require(context_,"lookup metadata allocation failed");
            token_=ggml_new_tensor_3d(context_,GGML_TYPE_F32,block,capacity/block,index_heads);
            row_=ggml_new_tensor_3d(context_,GGML_TYPE_F32,block,capacity/block,index_heads);
            ggml_set_name(token_,"m31.sorted_token_lookup");ggml_set_name(row_,"m31.sorted_row_lookup");
            buffer_=ggml_backend_alloc_ctx_tensors_from_buft(context_,ggml_backend_get_default_buffer_type(backend));
            require(buffer_,"lookup backend allocation failed");ggml_backend_buffer_set_usage(buffer_,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> values(size_t(capacity*index_heads));
            for(int64_t h=0;h<index_heads;++h)for(int64_t t=0;t<capacity;++t)values[size_t(h*capacity+t)]=float(t);
            ggml_backend_tensor_set(token_,values.data(),0,values.size()*sizeof(float));
            for(int64_t h=0;h<index_heads;++h)for(int64_t t=0;t<capacity;++t)values[size_t(h*capacity+t)]=float(t*kv_heads+h);
            ggml_backend_tensor_set(row_,values.data(),0,values.size()*sizeof(float));
        }catch(...){release();throw;}
    }
    ~MsaSortedLookupMaps(){release();}
    MsaSortedLookupMaps(const MsaSortedLookupMaps&)=delete;
    MsaSortedLookupMaps&operator=(const MsaSortedLookupMaps&)=delete;
    ggml_tensor * token_source()const{return token_;}
    ggml_tensor * row_source()const{return row_;}
    size_t resident_bytes()const{return buffer_?ggml_backend_buffer_get_size(buffer_):0;}
    // Optional source aliases allow an Engine Graph::borrow() wrapper without
    // transferring ownership. idx is the original score-sorted I32 top_k view;
    // its head stride may exceed top*sizeof(int32_t), exactly as ggml_top_k.
    Indices build(ggml_context * graph,ggml_tensor * idx,ggml_tensor * token_alias=nullptr,ggml_tensor * row_alias=nullptr)const{
        require(graph&&idx&&idx->type==GGML_TYPE_I32&&idx->ne[0]>0&&idx->ne[0]<=capacity_/block_&&idx->ne[1]==heads_&&idx->ne[2]==1&&idx->ne[3]==1,"invalid n1 sorted lookup index shape");
        auto ts=token_alias?token_alias:token_;auto rs=row_alias?row_alias:row_;
        for(auto source:{ts,rs})require(source&&source->type==GGML_TYPE_F32&&source->ne[0]==block_&&source->ne[1]==capacity_/block_&&source->ne[2]==heads_&&source->ne[3]==1,"invalid lookup source alias");
        // Public get_rows contract: [block,blocks,heads] x [top,heads]
        // -> contiguous [block,top,heads]. No block/head reordering occurs.
        auto tokens=ggml_get_rows(graph,ts,idx),rows=ggml_get_rows(graph,rs,idx);
        const int64_t count=block_*idx->ne[0]*heads_;
        return {ggml_cast(graph,ggml_reshape_1d(graph,tokens,count),GGML_TYPE_I32),
                ggml_cast(graph,ggml_reshape_1d(graph,rows,count),GGML_TYPE_I32)};
    }
};
} // namespace strata::m31
