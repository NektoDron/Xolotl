#pragma once
// Independent proposal, not included by Engine. Coalesce adjacent complete F32
// columns before allocation. Storage identity alone is insufficient: the producer
// path must contain metadata views only, or an inplace write could be skipped.
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
namespace strata::m31 {
inline bool dspark_coalesced_metadata_path(const ggml_tensor * row,const ggml_tensor * owner) {
    for(int depth=0;depth<64;++depth){
        if(row==owner)return true;
        if(!row||row->view_src!=owner)return false;
        switch(row->op){
            case GGML_OP_VIEW:case GGML_OP_RESHAPE:case GGML_OP_PERMUTE:case GGML_OP_TRANSPOSE:break;
            default:return false; // Includes inplace arithmetic and CPY writes.
        }
        row=row->src[0];
    }
    return false;
}
inline ggml_tensor * dspark_try_coalesce_rows(ggml_context * c,
        const std::vector<ggml_tensor *> & rows,int first,int count) {
    if(!c||first<0||count<2||count>4||size_t(first)>rows.size()||size_t(count)>rows.size()-size_t(first))return nullptr;
    auto * sample=rows[size_t(first)];
    if(!sample||sample->type!=GGML_TYPE_F32||sample->ne[0]<=0||sample->ne[1]!=1||sample->ne[2]!=1||sample->ne[3]!=1)return nullptr;
    const int64_t width=sample->ne[0];
    if(uint64_t(width)>std::numeric_limits<size_t>::max()/sizeof(float))return nullptr;
    const size_t stride=size_t(width)*sizeof(float),offset=sample->view_offs;
    auto * owner=sample->view_src;
    if(!owner||owner->view_src||owner->type!=GGML_TYPE_F32||owner->ne[0]!=width||owner->nb[0]!=sizeof(float)||owner->nb[1]!=stride||!ggml_is_contiguous(owner))return nullptr;
    if(offset%stride||stride>std::numeric_limits<size_t>::max()/size_t(count))return nullptr;
    const size_t bytes=stride*size_t(count),extent=ggml_nbytes(owner);
    if(offset>extent||bytes>extent-offset)return nullptr;
    for(int i=0;i<count;++i){auto * row=rows[size_t(first+i)];
        if(!row||row->type!=GGML_TYPE_F32||row->ne[0]!=width||row->ne[1]!=1||row->ne[2]!=1||row->ne[3]!=1||row->nb[0]!=sizeof(float)||row->nb[1]!=stride||!ggml_is_contiguous(row)||row->view_src!=owner||row->view_offs!=offset+size_t(i)*stride||!dspark_coalesced_metadata_path(row,owner))return nullptr;
    }
    return ggml_view_2d(c,owner,width,count,stride,offset);
}
} // namespace strata::m31
