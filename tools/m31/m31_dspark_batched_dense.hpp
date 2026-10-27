#pragma once
// Private graph helpers. Only individually marked dense weight projections use
// the exact K<=4 MMV dispatcher; MSA's F32 score matmul is never marked.
ggml_tensor * dspark_exact_mm(Graph & b,ggml_tensor * weight,ggml_tensor * input) {
    auto output=ggml_mul_mat(b.c,weight,input);
#if defined(M31_ENGINE_CUDA) && defined(GGML_CUDA_HAS_EXACT_MMV_MARKER)
    require(ggml_cuda_mul_mat_set_exact_mmv(output),"dense projection is not supported by exact K<=4 MMV");
#else
    require(false,"batched dense verifier requires the exact MMV CUDA marker API");
#endif
    return output;
}
ggml_tensor * dspark_join(Graph & b,const std::vector<ggml_tensor *> & rows,int first,int count) {
    if(b.dspark_coalesce_inputs)
        if(auto joined=dspark_try_coalesce_rows(b.c,rows,first,count))return joined;
    ggml_tensor * joined=nullptr;
    for(int i=0;i<count;++i){auto row=rows[size_t(first+i)];
        if(!b.dspark_view_inputs||!ggml_is_contiguous(row))row=ggml_cont(b.c,row);
        joined=joined?ggml_concat(b.c,joined,row,1):row;
    }
    return joined;
}
ggml_tensor * dspark_column(Graph & b,ggml_tensor * matrix,int column,int64_t rows,int64_t row_offset=0) {
    auto view=ggml_view_2d(b.c,matrix,rows,1,matrix->nb[1],size_t(column)*matrix->nb[1]+size_t(row_offset)*sizeof(float));
    return b.dspark_view_inputs&&ggml_is_contiguous(view)?view:ggml_cont(b.c,view);
}
std::vector<ggml_tensor *> dspark_projection(Graph & b,ggml_tensor * weight,const std::vector<ggml_tensor *> & rows) {
    std::vector<ggml_tensor *> result;result.reserve(rows.size());
    for(int first=0;first<int(rows.size());first+=4){
        const int count=std::min(4,int(rows.size())-first);
        auto output=dspark_exact_mm(b,weight,dspark_join(b,rows,first,count));
        for(int i=0;i<count;++i)result.push_back(dspark_column(b,output,i,output->ne[0]));
    }
    return result;
}
inline ggml_tensor * dspark_ops_glu(ggml_context *,ggml_tensor *,ggml_tensor *);
std::vector<ggml_tensor *> dspark_ffn(Graph & b,const Weights & weights,
    const std::vector<ggml_tensor *> & normalized,const std::string & prefix,const std::string & suffix,bool batch_ops=false) {
    std::vector<ggml_tensor *> result;result.reserve(normalized.size());
    for(int first=0;first<int(normalized.size());first+=4){
        const int count=std::min(4,int(normalized.size())-first);
        auto input=dspark_join(b,normalized,first,count);
        auto gate_weight=weights.get(prefix+"ffn_gate"+suffix+".weight");
        const int64_t width=gate_weight->ne[1];
        auto bundle=weights.maybe(prefix+"m31_ffn_gu"+suffix+".weight");
        ggml_tensor * gate=nullptr,*up=nullptr,*joined=nullptr;
        if(bundle)joined=dspark_exact_mm(b,bundle,input);
        else {gate=dspark_exact_mm(b,gate_weight,input);up=dspark_exact_mm(b,weights.get(prefix+"ffn_up"+suffix+".weight"),input);}
        ggml_tensor * activation=nullptr;
        if(batch_ops)activation=dspark_ops_glu(b.c,joined?joined:gate,joined?nullptr:up);
        else {
            std::vector<ggml_tensor *> activations;
            for(int i=0;i<count;++i){
                auto a=dspark_column(b,joined?joined:gate,i,width);
                auto u=dspark_column(b,joined?joined:up,i,width,joined?width:0);
                activations.push_back(ggml_swiglu_oai(b.c,a,u,1.702f,7.0f));
            }
            activation=dspark_join(b,activations,0,count);
        }
        auto output=dspark_exact_mm(b,weights.get(prefix+"ffn_down"+suffix+".weight"),activation);
        for(int i=0;i<count;++i)result.push_back(dspark_column(b,output,i,output->ne[0]));
    }
    return result;
}
