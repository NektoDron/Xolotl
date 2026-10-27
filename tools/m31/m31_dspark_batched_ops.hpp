#pragma once
// Private stock-op helpers, included inside strata::m31 after Geometry and ggml
// declarations. No environment/state changes. Every group is at most four
// independent query rows; MSA score products and FA are deliberately absent.
inline void dspark_ops_check(bool value,const char * reason) {
    if(!value)throw std::runtime_error(std::string("M31 batched ops: ")+reason);
}
inline ggml_tensor * dspark_ops_join_rows(ggml_context * c,
    const std::vector<ggml_tensor *> & rows,int first,int count) {
    dspark_ops_check(count>=1&&count<=4&&first>=0&&size_t(first+count)<=rows.size(),"invalid row group");
    auto * sample=rows[size_t(first)];
    dspark_ops_check(sample&&sample->type==GGML_TYPE_F32&&sample->ne[2]==1&&sample->ne[3]==1,"expected F32 query row");
    ggml_tensor * joined=nullptr;
    for(int i=0;i<count;++i){auto * row=rows[size_t(first+i)];
        dspark_ops_check(row&&row->type==GGML_TYPE_F32&&row->ne[0]==sample->ne[0]&&row->ne[1]==sample->ne[1]&&row->ne[2]==1&&row->ne[3]==1,"incompatible query rows");
        if(!ggml_is_contiguous(row))row=ggml_cont(c,row);
        joined=joined?ggml_concat(c,joined,row,1):row;
    }
    return ggml_reshape_3d(c,joined,sample->ne[0],sample->ne[1],count);
}
inline void dspark_ops_split_rows(ggml_context * c,ggml_tensor * tensor,int count,
    std::vector<ggml_tensor *> & rows) {
    dspark_ops_check(tensor&&tensor->type==GGML_TYPE_F32&&tensor->ne[2]==count&&ggml_is_contiguous(tensor),"invalid batched output");
    for(int i=0;i<count;++i)rows.push_back(ggml_view_3d(c,tensor,tensor->ne[0],tensor->ne[1],1,
        tensor->nb[1],tensor->nb[2],size_t(i)*tensor->nb[2]));
}
inline std::vector<ggml_tensor *> dspark_ops_rms_rows(ggml_context * c,
    const std::vector<ggml_tensor *> & rows,ggml_tensor * weight,float eps) {
    dspark_ops_check(!rows.empty()&&weight&&weight->type==GGML_TYPE_F32,"missing RMS rows/weights");
    std::vector<ggml_tensor *> result;result.reserve(rows.size());
    for(int first=0;first<int(rows.size());first+=4){const int count=std::min(4,int(rows.size())-first);
        auto * input=dspark_ops_join_rows(c,rows,first,count);
        dspark_ops_check(weight->ne[0]==input->ne[0]&&ggml_nelements(weight)==input->ne[0],"RMS weight shape");
        auto * output=ggml_mul(c,ggml_rms_norm(c,input,eps),weight);dspark_ops_split_rows(c,output,count,result);
    }
    return result;
}
inline std::vector<ggml_tensor *> dspark_ops_norm_rope_rows(ggml_context * c,
    const std::vector<ggml_tensor *> & rows,ggml_tensor * weight,float eps,
    ggml_tensor * positions,const Geometry & g) {
    dspark_ops_check(!rows.empty()&&positions&&positions->type==GGML_TYPE_I32&&
        positions->ne[0]>=int64_t(rows.size())&&positions->ne[1]==1&&positions->ne[2]==1&&positions->ne[3]==1&&ggml_is_contiguous(positions),"position vector shape");
    dspark_ops_check(weight&&weight->type==GGML_TYPE_F32,"missing Q/K normalization weights");
    std::vector<ggml_tensor *> result;result.reserve(rows.size());
    for(int first=0;first<int(rows.size());first+=4){const int count=std::min(4,int(rows.size())-first);
        auto * input=dspark_ops_join_rows(c,rows,first,count);
        dspark_ops_check(weight->ne[0]==input->ne[0]&&ggml_nelements(weight)==input->ne[0],"Q/K weight shape");
        auto * norm=ggml_mul(c,ggml_rms_norm(c,input,eps),weight);
        auto * pos=ggml_view_1d(c,positions,count,size_t(first)*sizeof(int32_t));
        auto * output=ggml_rope_ext(c,norm,pos,nullptr,int(g.rope_dim),GGML_ROPE_TYPE_NEOX,
            int(g.context),g.rope_base,1,0,1,32,1);
        dspark_ops_split_rows(c,output,count,result);
    }
    return result;
}
inline ggml_tensor * dspark_ops_glu(ggml_context * c,ggml_tensor * gate,ggml_tensor * up=nullptr) {
    dspark_ops_check(gate&&gate->type==GGML_TYPE_F32&&gate->ne[1]>=1&&gate->ne[1]<=4&&
        gate->ne[2]==1&&gate->ne[3]==1&&ggml_is_contiguous(gate),"expected contiguous K<=4 GU matrix");
    if(up)dspark_ops_check(up->type==GGML_TYPE_F32&&ggml_are_same_shape(gate,up)&&ggml_is_contiguous(up),"gate/up shape");
    else dspark_ops_check(gate->ne[0]%2==0,"packed gate/up row split");
    return ggml_swiglu_oai(c,gate,up,1.702f,7.f);
}
struct DSparkBatchedRouterRows {
    std::vector<ggml_tensor *> ids,weights;
};
inline DSparkBatchedRouterRows dspark_ops_router_rows(ggml_context * c,
    const std::vector<ggml_tensor *> & logits,ggml_tensor * bias,const Geometry & g) {
    dspark_ops_check(!logits.empty()&&bias&&bias->type==GGML_TYPE_F32&&ggml_nelements(bias)==g.experts,"router bias shape");
    DSparkBatchedRouterRows result;result.ids.reserve(logits.size());result.weights.reserve(logits.size());
    for(int first=0;first<int(logits.size());first+=4){const int count=std::min(4,int(logits.size())-first);
        auto * input=dspark_ops_join_rows(c,logits,first,count);
        dspark_ops_check(input->ne[0]==g.experts&&input->ne[1]==1,"router logits shape");
        auto * probs=ggml_sigmoid(c,ggml_reshape_2d(c,input,g.experts,count));
        auto * ids=ggml_top_k(c,ggml_add(c,probs,bias),int(g.used));
        auto * weights=ggml_get_rows(c,ggml_reshape_3d(c,probs,1,g.experts,count),ids);
        weights=ggml_reshape_2d(c,weights,g.used,count);
        if(g.norm_routes){auto * sum=ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY);weights=ggml_div(c,weights,sum);}
        if(g.route_scale!=1)weights=ggml_scale(c,weights,g.route_scale);
        for(int i=0;i<count;++i){result.ids.push_back(ggml_view_2d(c,ids,g.used,1,ids->nb[1],size_t(i)*ids->nb[1]));
            result.weights.push_back(ggml_view_2d(c,weights,g.used,1,weights->nb[1],size_t(i)*weights->nb[1]));}
    }
    return result;
}
