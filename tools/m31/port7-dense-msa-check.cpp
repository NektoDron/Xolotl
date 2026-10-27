// Reuse the private GGUF loader/stock graph owner from the earlier component.
// Its renamed entrypoint is never called; no production engine code is linked.
#define main dense_ops_unused_main
#include "port7-dense-ops-check.cpp"
#undef main
#ifdef M31_MSA_PRODUCTION_HELPER
namespace strata::m31 {
#include "m31_dspark_batched_msa.hpp"
}
#endif

static std::vector<float> encode_rows(ggml_backend_t backend,const m31::Geometry & g,
        ggml_tensor * weight,const std::vector<float> & input,int heads,int first) {
    const int count=int(input.size()/size_t(g.index_dim*heads));
    GraphOwner b(backend);
    auto * x=ggml_new_tensor_3d(b.c,GGML_TYPE_F32,g.index_dim,heads,1);
    auto * pos=ggml_new_tensor_1d(b.c,GGML_TYPE_I32,1);
    auto * norm=ggml_mul(b.c,ggml_rms_norm(b.c,x,g.rms_eps),weight);
    auto * out=ggml_rope_ext(b.c,norm,pos,nullptr,int(g.rope_dim),GGML_ROPE_TYPE_NEOX,
        int(g.context),g.rope_base,1,0,1,32,1);
    b.finish({out});std::vector<float> result(input.size());
    for(int row=0;row<count;++row){int32_t position=first+row;
        ggml_backend_tensor_set(x,input.data()+size_t(row*g.index_dim*heads),0,size_t(g.index_dim*heads)*4);
        ggml_backend_tensor_set(pos,&position,0,4);b.compute();
        ggml_backend_tensor_get(out,result.data()+size_t(row*g.index_dim*heads),0,size_t(g.index_dim*heads)*4);
    }
    return result;
}
static std::vector<ggml_fp16_t> half_keys(ggml_backend_t backend,const m31::Geometry & g,const std::vector<float> & encoded) {
    GraphOwner b(backend);auto * x=ggml_new_tensor_2d(b.c,GGML_TYPE_F32,g.index_dim,int64_t(encoded.size())/g.index_dim);
    auto * y=ggml_cast(b.c,x,GGML_TYPE_F16);b.finish({y});ggml_backend_tensor_set(x,encoded.data(),0,encoded.size()*4);b.compute();
    std::vector<ggml_fp16_t> result(encoded.size());ggml_backend_tensor_get(y,result.data(),0,result.size()*2);return result;
}
class MsaGraph:public GraphOwner {
public:
    ggml_tensor * keys=nullptr,*query=nullptr,*mask=nullptr,*bias=nullptr;
    std::vector<ggml_tensor *> outputs;const std::vector<std::string> names{"cast_keys","score","masked","block_max","biased","block_ids"};
    MsaGraph(ggml_backend_t backend,const m31::Geometry & g,int nk,int count,int start,bool scalar,
            const std::vector<ggml_fp16_t> & key_rows,const std::vector<float> & queries,ggml_type mask_type=GGML_TYPE_F16):GraphOwner(backend) {
        keys=ggml_new_tensor_2d(c,GGML_TYPE_F16,g.index_dim,nk);
        query=ggml_new_tensor_3d(c,GGML_TYPE_F32,g.index_dim,g.index_heads,count);
        mask=ggml_new_tensor_3d(c,mask_type,nk,1,count);
        const int64_t blocks=nk/g.block,top=std::min(g.top_blocks,blocks);
        bias=ggml_new_tensor_3d(c,GGML_TYPE_F32,blocks,1,count);
        auto * cast=ggml_cast(c,keys,GGML_TYPE_F32);outputs.push_back(cast);
        auto build=[&](ggml_tensor * q,ggml_tensor * m,ggml_tensor * bi) {
            auto * score=ggml_mul_mat(c,cast,q);ggml_prec_set_acc(score,GGML_PREC_F32);
            // No exact-MMV marker: computed KV is deliberately outside that API.
            auto * masked=ggml_add(c,score,m->type==GGML_TYPE_F32?m:ggml_cast(c,m,GGML_TYPE_F32));
            auto * pooled=ggml_pool_2d(c,masked,GGML_OP_POOL_MAX,int(g.block),1,int(g.block),1,0,0);
            auto * biased=ggml_add(c,pooled,bi);auto * ids=ggml_top_k(c,biased,int(top));
            return std::vector<ggml_tensor *>{score,masked,pooled,biased,ids};
        };
        if(scalar)for(int row=0;row<count;++row){
            auto * q=ggml_view_3d(c,query,g.index_dim,g.index_heads,1,query->nb[1],query->nb[2],size_t(row)*query->nb[2]);
            auto * m=ggml_view_3d(c,mask,nk,1,1,mask->nb[1],mask->nb[2],size_t(row)*mask->nb[2]);
            auto * bi=ggml_view_3d(c,bias,blocks,1,1,bias->nb[1],bias->nb[2],size_t(row)*bias->nb[2]);
            const auto values=build(q,m,bi);
            if(row==0)outputs.insert(outputs.end(),values.begin(),values.end());
            else for(size_t i=0;i<values.size();++i)outputs[i+1]=ggml_concat(c,outputs[i+1],values[i],2);
        }else{
#ifdef M31_MSA_PRODUCTION_HELPER
            std::vector<ggml_tensor *> queries,masks,biases;
            for(int row=0;row<count;++row){queries.push_back(ggml_view_3d(c,query,g.index_dim,g.index_heads,1,query->nb[1],query->nb[2],size_t(row)*query->nb[2]));
                masks.push_back(ggml_view_3d(c,mask,nk,1,1,mask->nb[1],mask->nb[2],size_t(row)*mask->nb[2]));
                biases.push_back(ggml_view_3d(c,bias,blocks,1,1,bias->nb[1],bias->nb[2],size_t(row)*bias->nb[2]));}
            m31::DSparkMSADebug debug;const auto selected=m31::dspark_msa_select_rows(c,cast,queries,masks,biases,g,&debug);
            need(selected.size()==size_t(count),"production MSA row count");
            ggml_tensor * ids=nullptr;for(auto * row:selected)ids=ids?ggml_concat(c,ids,row,2):row;
            outputs.insert(outputs.end(),{debug.score,debug.masked,debug.block_max,debug.biased,ids});
#else
            const auto values=build(query,mask,bias);outputs.insert(outputs.end(),values.begin(),values.end());
#endif
        }
        for(auto & out:outputs)out=ggml_cont(c,out);finish(outputs);
        std::vector<ggml_fp16_t> repeated(size_t(g.index_dim)*size_t(nk));
        for(size_t i=0;i<repeated.size();++i)repeated[i]=key_rows[i%key_rows.size()];
        std::vector<ggml_fp16_t> causal(size_t(nk)*size_t(count));
        std::vector<float> causal_float(size_t(nk)*size_t(count));
        std::vector<float> block_bias(size_t(blocks)*size_t(count));
        for(int row=0;row<count;++row){const int position=start+row;
            for(int key=0;key<nk;++key){const size_t cell=size_t(row)*size_t(nk)+size_t(key);causal_float[cell]=key<=position?0.f:-INFINITY;causal[cell]=ggml_fp32_to_fp16(causal_float[cell]);}
            for(int64_t block=0;block<blocks;++block){const int64_t current=position/g.block;
                block_bias[size_t(row)*size_t(blocks)+size_t(block)]=block<=current&&block>current-g.local_blocks?1.e30f:0.f;}
        }
        ggml_backend_tensor_set(keys,repeated.data(),0,repeated.size()*2);ggml_backend_tensor_set(query,queries.data(),0,size_t(g.index_dim*g.index_heads*count)*4);
        if(mask_type==GGML_TYPE_F16)ggml_backend_tensor_set(mask,causal.data(),0,causal.size()*2);
        else ggml_backend_tensor_set(mask,causal_float.data(),0,causal_float.size()*4);
        ggml_backend_tensor_set(bias,block_bias.data(),0,block_bias.size()*4);
    }
};
static Json compare_msa(MsaGraph & reference,MsaGraph & batch) {
    Json result=Json::array();
    for(size_t i=0;i<reference.outputs.size();++i){auto * a=reference.outputs[i];auto * b=batch.outputs[i];
        need(a->type==b->type&&ggml_nelements(a)==ggml_nelements(b),"MSA output shape mismatch");
        const size_t n=size_t(ggml_nelements(a));std::vector<uint32_t> expected(n),actual(n);
        ggml_backend_tensor_get(a,expected.data(),0,n*4);ggml_backend_tensor_get(b,actual.data(),0,n*4);
        size_t different=0,nonfinite_different=0;double maximum=0;
        for(size_t k=0;k<n;++k){different+=expected[k]!=actual[k];if(a->type==GGML_TYPE_F32){float x,y;std::memcpy(&x,&expected[k],4);std::memcpy(&y,&actual[k],4);
            need(!std::isnan(x)&&!std::isnan(y),"MSA NaN output");if(std::isfinite(x)&&std::isfinite(y))maximum=std::max(maximum,std::abs(double(x)-y));else nonfinite_different+=expected[k]!=actual[k];}}
        result.push_back({{"name",reference.names[i]},{"type",ggml_type_name(a->type)},{"values",n},{"different",different},{"nonfinite_different",nonfinite_different},{"max_abs",maximum}});
    }
    return result;
}
int main(int argc,char ** argv){ggml_backend_t backend=nullptr;try{
    need(argc==4,"usage: authorized GPU0 probe MODEL FIXTURE OUT.json");
    need(std::getenv("M31_CARD_SESSION")&&std::strcmp(std::getenv("M31_CARD_SESSION"),"1")==0&&std::getenv("CUDA_VISIBLE_DEVICES")&&std::strcmp(std::getenv("CUDA_VISIBLE_DEVICES"),"0")==0,"requires authorized GPU0 session");
    need(!fs::exists(argv[3]),"output exists");const fs::path fixtures=argv[2];Json fixture;std::ifstream manifest(fixtures/"manifest.json");manifest>>fixture;
    const auto model=GgufModel::open(argv[1]);const auto g=m31::read_geometry(model);backend=ggml_backend_cuda_init(0);need(backend,"CUDA init");
    Json report={{"scope","Immutable real-derived F16 index key fixtures, n1 stock score vs stock [D,heads4,K] channel batch. CAST shared keys, score, F16 causal mask to F32, block MAX/local bias/topK. No marker, no FA, no Engine changes. Long keys cycle 128 real-derived rows; timings are component-only."},{"fixture",fixture},{"cases",Json::array()}};
#ifdef M31_MSA_PRODUCTION_HELPER
    report["production_helper"]=true;
    report["scope"]="Actual production context MSA helper vs independent n1 scalar stages, K1/2/3/4, F16 and F32 masks, immutable real-derived F16 keys cycled to long Nk. Shared F32 CAST, head4 retained, no score marker or FA/Engine changes. Exact-MmvAPIv3 backend; operator proof only, timings component-only.";
#endif
    {Weights weights(model,backend);for(const auto & item:fixture["layers"]){const int layer=item["layer"];const std::string prefix="blk."+std::to_string(layer)+".";
        const auto hidden=floats(fixtures/item["attn_norm"].get<std::string>());
        const auto raw_keys=project_n1(backend,weights.get(prefix+"indexer.k_proj.weight"),hidden);
        const auto keys=half_keys(backend,g,encode_rows(backend,g,weights.get(prefix+"indexer.k_norm.weight"),raw_keys,1,0));
        const std::vector<float> selected(hidden.begin()+17*g.embd,hidden.begin()+21*g.embd);
        const auto raw_queries=project_n1(backend,weights.get(prefix+"indexer.q_proj.weight"),selected);
        for(int nk:{256,4096,33280,65536}){
            const std::vector<int> starts=nk==256?std::vector<int>{252}:nk==4096?std::vector<int>{254}:nk==33280?std::vector<int>{32766}:std::vector<int>{32766,65532};
            for(int start:starts){const auto queries=encode_rows(backend,g,weights.get(prefix+"indexer.q_norm.weight"),raw_queries,int(g.index_heads),start);
            std::vector<int> counts{2,3,4};std::vector<ggml_type> mask_types{GGML_TYPE_F16};
#ifdef M31_MSA_PRODUCTION_HELPER
            counts.insert(counts.begin(),1);mask_types.push_back(GGML_TYPE_F32);
#endif
            for(auto mask_type:mask_types)for(int count:counts){
                MsaGraph scalar(backend,g,nk,count,start,true,keys,queries,mask_type),batch(backend,g,nk,count,start,false,keys,queries,mask_type);
                for(int repeat=0;repeat<2;++repeat){scalar.compute();batch.compute();}
                auto outputs=compare_msa(scalar,batch);bool exact=true;for(const auto & out:outputs)exact=exact&&out["different"].get<size_t>()==0;
                const double scalar_us=dense_ops_graph_us(backend,scalar.graph,8),batch_us=dense_ops_graph_us(backend,batch.graph,8);
                report["cases"].push_back({{"layer",layer},{"K",count},{"nk",nk},{"mask_type",ggml_type_name(mask_type)},{"position_start",start},{"ncols",g.index_heads},{"outputs",outputs},{"exact",exact},{"scalar_graph_us",scalar_us},{"batch_graph_us",batch_us}});
                std::fprintf(stderr,"msa layer=%d K=%d nk=%d pos=%d exact=%d scalar_us=%.3f batch_us=%.3f\n",layer,count,nk,start,int(exact),scalar_us,batch_us);
            }
            }
        }
    }}
    ggml_backend_free(backend);backend=nullptr;std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';return 0;
}catch(const std::exception & e){std::fprintf(stderr,"MSA component: %s\n",e.what());if(backend)ggml_backend_free(backend);return 1;}}
