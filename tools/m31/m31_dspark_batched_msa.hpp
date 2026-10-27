#pragma once
// Private context helper. Caller completes all n1 KV SET_ROWS before the shared
// F16->F32 keys CAST, then groups only queries having the same n1 Nk bucket.
// Score keeps heads in ne1 and queries in ne2. It is NEVER an exact-MMV weight
// projection. Sorted sparse gather and FA remain outside this helper.
struct DSparkMSADebug {
    ggml_tensor * score=nullptr,*masked=nullptr,*block_max=nullptr,*biased=nullptr,*ids=nullptr;
};
inline void dspark_msa_check(bool ok,const char * why){
    if(!ok)throw std::runtime_error(std::string("M31 batched MSA: ")+why);
}
inline std::vector<ggml_tensor *> dspark_msa_select_rows(ggml_context * c,ggml_tensor * keys,
        const std::vector<ggml_tensor *> & queries,const std::vector<ggml_tensor *> & masks,
        const std::vector<ggml_tensor *> & biases,const Geometry & g,DSparkMSADebug * debug=nullptr) {
    const size_t count=queries.size();
    dspark_msa_check(c&&keys&&keys->type==GGML_TYPE_F32&&keys->ne[0]==g.index_dim&&
        keys->ne[2]==1&&keys->ne[3]==1&&ggml_is_contiguous(keys),"expected shared contiguous F32 keys[D,Nk]");
    const int64_t nk=keys->ne[1];
    dspark_msa_check(count>=1&&count<=4&&masks.size()==count&&biases.size()==count,"expected one same-Nk group of 1..4 rows");
    dspark_msa_check(g.block>0&&nk>0&&nk%g.block==0&&g.index_heads>0&&g.top_blocks>0&&
        nk<INT32_MAX/int64_t(count)/g.index_heads,"unsupported Nk/block/index kernel geometry");
    const int64_t blocks=nk/g.block,top=std::min(g.top_blocks,blocks);
    const auto mask_type=masks[0]?masks[0]->type:GGML_TYPE_COUNT;
    dspark_msa_check(mask_type==GGML_TYPE_F16||mask_type==GGML_TYPE_F32,"expected F16/F32 causal masks");
    for(size_t i=0;i<count;++i){auto * q=queries[i];auto * m=masks[i];auto * bias=biases[i];
        dspark_msa_check(q&&q->type==GGML_TYPE_F32&&q->ne[0]==g.index_dim&&q->ne[1]==g.index_heads&&q->ne[2]==1&&q->ne[3]==1&&ggml_is_contiguous(q),"query shape/stride mismatch");
        dspark_msa_check(m&&m->type==mask_type&&m->ne[0]==nk&&m->ne[1]==1&&m->ne[2]==1&&m->ne[3]==1&&ggml_is_contiguous(m),"mask Nk/type/stride mismatch");
        dspark_msa_check(bias&&bias->type==GGML_TYPE_F32&&bias->ne[0]==blocks&&bias->ne[1]==1&&bias->ne[2]==1&&bias->ne[3]==1&&ggml_is_contiguous(bias),"block bias shape/stride mismatch");
    }
    auto join=[&](const std::vector<ggml_tensor *> & rows){ggml_tensor * result=nullptr;
        for(auto * row:rows)result=result?ggml_concat(c,result,row,2):row;return result;};
    auto * query=join(queries);auto * mask=join(masks);auto * bias=join(biases);
    auto * score=ggml_mul_mat(c,keys,query);ggml_prec_set_acc(score,GGML_PREC_F32);
    if(mask_type!=GGML_TYPE_F32)mask=ggml_cast(c,mask,GGML_TYPE_F32);
    auto * masked=ggml_add(c,score,mask);
    auto * pooled=ggml_pool_2d(c,masked,GGML_OP_POOL_MAX,int(g.block),1,int(g.block),1,0,0);
    auto * biased=ggml_add(c,pooled,bias);auto * ids=ggml_top_k(c,biased,int(top));
    if(debug)*debug=DSparkMSADebug{score,masked,pooled,biased,ids};
    std::vector<ggml_tensor *> result;result.reserve(count);
    for(size_t i=0;i<count;++i)result.push_back(ggml_view_3d(c,ids,top,g.index_heads,1,ids->nb[1],ids->nb[2],i*ids->nb[2]));
    return result;
}
