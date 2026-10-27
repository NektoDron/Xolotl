// MiniMax DSpark dense backbone and Markov head, from llama.cpp PR #31192
// src/models/dflash.cpp / conversion/minimax.py (MIT). This is opt-in draft math:
// all returned tokens must be verified by the unchanged target model.
#include "m31_dspark.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml-alloc.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace strata::m31 {
namespace {
void need(bool ok, const std::string & why) { if (!ok) throw std::runtime_error("M31 DSpark: " + why); }
// The target may enable a separately validated n1 fusion policy. The draft's
// n>1 backbone keeps its own unfused policy, including exceptional exits.
struct DraftFusionGuard {
    ggml_backend_t backend=nullptr;
    int previous=-2;
    explicit DraftFusionGuard(ggml_backend_t be):backend(be) {
#if defined(M31_ENGINE_CUDA) && defined(GGML_CUDA_HAS_FUSION_OVERRIDE)
        if(ggml_backend_is_cuda(be)) {
            const int current=ggml_backend_cuda_get_fusion_override(be);
            need(current>=-1,"cannot read draft backend fusion policy");
            const auto disabled=std::getenv("GGML_CUDA_DISABLE_FUSION");
            const bool inherited_off=current==-1&&disabled&&std::string(disabled)=="1";
            // An unnecessary 0/-1 override would invalidate every retained
            // target capture despite an already unfused accepted profile.
            if(current!=0&&!inherited_off){
                need(ggml_backend_cuda_set_fusion_override(be,0),"cannot select unfused draft policy");
                previous=current;
            }
        }
#endif
    }
    void restore() {
#if defined(M31_ENGINE_CUDA) && defined(GGML_CUDA_HAS_FUSION_OVERRIDE)
        if(previous>=-1){const int mode=previous;previous=-2;need(ggml_backend_cuda_set_fusion_override(backend,mode),"cannot restore target fusion policy");}
#endif
    }
    ~DraftFusionGuard() {
#if defined(M31_ENGINE_CUDA) && defined(GGML_CUDA_HAS_FUSION_OVERRIDE)
        if(previous>=-1)(void)ggml_backend_cuda_set_fusion_override(backend,previous);
#endif
    }
};
struct Graph {
    ggml_context * c = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_gallocr_t a = nullptr;
    ggml_backend_t be = nullptr;
    ggml_tensor * input = nullptr, * positions = nullptr, * slots = nullptr, * mask = nullptr;
    ggml_tensor * anchor = nullptr;
    std::vector<ggml_tensor *> ids, confidence;
    explicit Graph(ggml_backend_t b):be(b) {
        c = ggml_init({32*1024*1024,nullptr,true});
        need(c,"graph metadata allocation failed");
        g = ggml_new_graph_custom(c,8192,false);
        a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        need(a,"graph allocator failed");
    }
    ~Graph() {
#ifdef M31_ENGINE_CUDA
        ggml_backend_cuda_graph_cache_clear(be);
#endif
        if(a)ggml_gallocr_free(a);
        if(c)ggml_free(c);
    }
    ggml_tensor * inp(ggml_type t,int64_t n,int64_t m=1) {
        auto x=ggml_new_tensor_2d(c,t,n,m);ggml_set_input(x);return x;
    }
    void out(ggml_tensor * x) { ggml_set_output(x);ggml_build_forward_expand(g,x); }
    void alloc() { need(ggml_gallocr_alloc_graph(a,g),"graph allocation failed"); }
    void compute() { DraftFusionGuard policy(be);need(ggml_backend_graph_compute(be,g)==GGML_STATUS_SUCCESS,"graph compute failed");policy.restore(); }
};
}
struct DSpark::Impl {
    DSparkConfig cfg;
    std::unique_ptr<GgufModel> side,target;
    ggml_context * wc=nullptr,*sc=nullptr;
    ggml_backend_buffer_t wb=nullptr,sb=nullptr;
    std::map<std::string,ggml_tensor*> w;
    std::map<int,std::unique_ptr<Graph>> append_graphs,draft_graphs;
    std::vector<ggml_tensor*> keys,values;
    std::vector<int> layers;
    std::vector<int64_t> filled;
    const TensorInfo * embedding=nullptr;
    size_t embedding_shard=0;
    int D=0,H=0,HK=0,DH=0,ROT=0,L=0,WINDOW=0,BLOCK=0,VOCAB=0,MASK=0,CAP=0;
    float eps=0,base=0;
    int64_t pos=0;
    uint64_t bytes=0;
    double ms=0;
    bool ready=false,poisoned=false;
    ~Impl() {
        draft_graphs.clear();append_graphs.clear();
        if(sb)ggml_backend_buffer_free(sb);if(sc)ggml_free(sc);
        if(wb)ggml_backend_buffer_free(wb);if(wc)ggml_free(wc);
    }
    ggml_tensor * W(const std::string & name) const {
        auto it=w.find(name);need(it!=w.end(),"missing weight "+name);return it->second;
    }
    ggml_tensor * norm(ggml_context * c,ggml_tensor * x,ggml_tensor * wt) {
        return ggml_mul(c,ggml_rms_norm(c,x,eps),wt);
    }
    ggml_tensor * rope(ggml_context * c,ggml_tensor * x,ggml_tensor * positions) {
        return ggml_rope_ext(c,x,positions,nullptr,ROT,GGML_ROPE_TYPE_NEOX,1048576,base,1,0,1,32,1);
    }
    ggml_tensor * write(Graph & g,int layer,ggml_tensor * x,ggml_tensor * slots,bool key) {
        auto cache=key?keys[size_t(layer)]:values[size_t(layer)];
        x=ggml_cont_2d(g.c,x,DH*HK,x->ne[2]);
        auto result=ggml_set_rows(g.c,cache,x,slots);
        ggml_build_forward_expand(g.g,result);
        return ggml_reshape_3d(g.c,result,DH,HK,CAP);
    }
    Graph & build_append(int n) {
        auto & owner=append_graphs[n];if(owner)return *owner;
        owner=std::make_unique<Graph>(cfg.backend);auto & g=*owner;auto c=g.c;
        g.input=g.inp(GGML_TYPE_F32,D*int(layers.size()),n);
        g.positions=g.inp(GGML_TYPE_I32,n);g.slots=g.inp(GGML_TYPE_I64,n);
        auto x=norm(c,ggml_mul_mat(c,W("fc.weight"),g.input),W("enc.output_norm.weight"));
        for(int l=0;l<L;++l) {
            const auto prefix="blk."+std::to_string(l)+".";
            auto k=ggml_reshape_3d(c,ggml_mul_mat(c,W(prefix+"attn_k.weight"),x),DH,HK,n);
            auto v=ggml_reshape_3d(c,ggml_mul_mat(c,W(prefix+"attn_v.weight"),x),DH,HK,n);
            k=rope(c,norm(c,k,W(prefix+"attn_k_norm.weight")),g.positions);
            write(g,l,k,g.slots,true);write(g,l,v,g.slots,false);
        }
        g.alloc();return g;
    }
    Graph & build_draft(int n) {
        auto & owner=draft_graphs[n];if(owner)return *owner;
        owner=std::make_unique<Graph>(cfg.backend);auto & g=*owner;auto c=g.c;
        g.input=g.inp(GGML_TYPE_F32,D,n);g.positions=g.inp(GGML_TYPE_I32,n);
        g.slots=g.inp(GGML_TYPE_I64,n);g.anchor=g.inp(GGML_TYPE_I32,1);
        g.mask=g.inp(GGML_TYPE_F16,CAP,32);auto x=g.input;
        for(int l=0;l<L;++l) {
            const auto prefix="blk."+std::to_string(l)+".";
            auto a=norm(c,x,W(prefix+"attn_norm.weight"));
            auto q=ggml_reshape_3d(c,ggml_mul_mat(c,W(prefix+"attn_q.weight"),a),DH,H,n);
            auto k=ggml_reshape_3d(c,ggml_mul_mat(c,W(prefix+"attn_k.weight"),a),DH,HK,n);
            auto v=ggml_reshape_3d(c,ggml_mul_mat(c,W(prefix+"attn_v.weight"),a),DH,HK,n);
            q=rope(c,norm(c,q,W(prefix+"attn_q_norm.weight")),g.positions);
            k=rope(c,norm(c,k,W(prefix+"attn_k_norm.weight")),g.positions);
            auto kc=write(g,l,k,g.slots,true),vc=write(g,l,v,g.slots,false);
            auto fa=ggml_flash_attn_ext(c,ggml_permute(c,q,0,2,1,3),
                ggml_permute(c,kc,0,2,1,3),ggml_permute(c,vc,0,2,1,3),g.mask,
                1.0f/std::sqrt(float(DH)),0,0);
            ggml_prec_set_acc(fa,GGML_PREC_F32);
            auto attention=ggml_mul_mat(c,W(prefix+"attn_output.weight"),ggml_reshape_2d(c,fa,DH*H,n));
            auto residual=ggml_add(c,attention,x);
            a=norm(c,residual,W(prefix+"ffn_norm.weight"));
            auto gate=ggml_mul_mat(c,W(prefix+"ffn_gate.weight"),a);
            auto up=ggml_mul_mat(c,W(prefix+"ffn_up.weight"),a);
            auto act=ggml_swiglu_oai(c,gate,up,1.702f,7.0f);
            x=ggml_add(c,ggml_mul_mat(c,W(prefix+"ffn_down.weight"),act),residual);
        }
        auto final=norm(c,x,W("output_norm.weight"));
        auto logits=ggml_mul_mat(c,cfg.shared_head,final);
        auto prev=g.anchor;
        for(int i=0;i<n;++i) {
            auto markov=ggml_get_rows(c,W("markov_w1.weight"),prev);
            auto bias=ggml_mul_mat(c,W("markov_w2.weight"),markov);
            auto column=ggml_view_2d(c,logits,VOCAB,1,logits->nb[1],size_t(i)*logits->nb[1]);
            auto scores=ggml_add(c,column,bias);
            auto id=ggml_argmax(c,scores);g.ids.push_back(id);g.out(id);
            auto h=ggml_view_2d(c,final,D,1,final->nb[1],size_t(i)*final->nb[1]);
            auto conf=ggml_mul_mat(c,W("conf_proj.weight"),ggml_concat(c,h,markov,0));
            conf=ggml_sigmoid(c,ggml_add(c,conf,W("conf_proj.bias")));
            g.confidence.push_back(conf);g.out(conf);prev=id;
        }
        g.alloc();return g;
    }
};
DSpark::DSpark():p_(new Impl){}
DSpark::~DSpark()=default;
int64_t DSpark::position()const{return p_->pos;}
int DSpark::feature_dim()const{return p_->D*int(p_->layers.size());}
int DSpark::block_size()const{return p_->BLOCK;}
uint64_t DSpark::weight_bytes()const{return p_->bytes;}
uint64_t DSpark::resident_bytes()const{
    uint64_t bytes=p_->wb?ggml_backend_buffer_get_size(p_->wb):0;
    if(p_->sb)bytes+=ggml_backend_buffer_get_size(p_->sb);
    for(const auto & [key,graph]:p_->append_graphs)if(graph)bytes+=ggml_gallocr_get_buffer_size(graph->a,0);
    for(const auto & [key,graph]:p_->draft_graphs)if(graph)bytes+=ggml_gallocr_get_buffer_size(graph->a,0);
    return bytes;
}
double DSpark::last_ms()const{return p_->ms;}
const std::vector<int>&DSpark::target_layers()const{return p_->layers;}
bool DSpark::init(const DSparkConfig & cfg,std::string & error) {
    try {
        need(!p_->ready,"already initialized");need(cfg.backend&&cfg.shared_head,"backend/shared target head required");
        need(cfg.max_append>0&&cfg.max_append<=4096,"max_append out of range");
        auto & p=*p_;p.cfg=cfg;
        size_t free_before=0,total_before=0;
        ggml_backend_dev_memory(ggml_backend_get_device(cfg.backend),&free_before,&total_before);
        p.side=std::make_unique<GgufModel>(GgufModel::open(cfg.sidecar));
        p.target=std::make_unique<GgufModel>(GgufModel::open(cfg.target));
        const auto & f=p.side->meta();
        auto integer=[&](const std::string & key) {
            auto v=f.get(key);need(v&&v->is_num()&&std::isfinite(v->num())&&v->num()>0&&v->num()<=1048576&&std::floor(v->num())==v->num(),"invalid "+key);return int(v->num());
        };
        auto real=[&](const std::string & key) { auto v=f.get(key);need(v&&v->is_num()&&std::isfinite(v->num())&&v->num()>0,"invalid "+key);return float(v->num()); };
        auto arch=f.get("general.architecture");need(arch&&arch->s=="dflash","sidecar architecture is not dflash");
        p.D=integer("dflash.embedding_length");p.H=integer("dflash.attention.head_count");p.HK=integer("dflash.attention.head_count_kv");
        p.DH=integer("dflash.attention.key_length");p.ROT=integer("dflash.rope.dimension_count");p.L=integer("dflash.block_count");
        p.WINDOW=integer("dflash.attention.sliding_window");p.BLOCK=integer("dflash.block_size");p.MASK=integer("tokenizer.ggml.mask_token_id");
        p.eps=real("dflash.attention.layer_norm_rms_epsilon");p.base=real("dflash.rope.freq_base");
        const auto * tls=f.get("dflash.target_layers");need(tls&&tls->type==MetaType::ARRAY,"missing target_layers");
        for(const auto & v:tls->items) {need(v.is_num()&&v.num()>=1&&v.num()<=60&&std::floor(v.num())==v.num(),"invalid target layer");p.layers.push_back(int(v.num()));}
        need(p.layers==std::vector<int>({4,18,32,46,60}),"unsupported target layer contract");
        need(p.D==6144&&p.H==64&&p.HK==4&&p.DH==128&&p.ROT==64&&p.L==5&&p.BLOCK==7&&p.WINDOW==4096,"unsupported draft geometry");
        auto anchor=f.get("dflash.sample_from_anchor");need(anchor&&anchor->type==MetaType::BOOL&&anchor->u,"only sample_from_anchor=true is supported");
        auto act=f.get("dflash.hidden_activation");need(act&&act->s=="swigluoai","unsupported FFN activation");
        p.CAP=(p.WINDOW+p.BLOCK+255)/256*256;
        p.embedding=p.target->find("token_embd.weight",&p.embedding_shard);need(p.embedding,"target embedding missing");
        need(p.embedding->shape.size()==2&&p.embedding->shape[0]==uint64_t(p.D),"target embedding dimensions differ");
        p.VOCAB=int(p.embedding->shape[1]);need(cfg.shared_head->ne[0]==p.D&&cfg.shared_head->ne[1]==p.VOCAB,"target head shape mismatch");
        need(p.MASK<p.VOCAB,"mask token outside vocabulary");
        p.wc=ggml_init({4*1024*1024,nullptr,true});need(p.wc,"weight metadata failed");
        for(size_t sh=0;sh<p.side->size();++sh)for(const auto & ti:p.side->shard(sh).tensors()) {
            need(ti.type<GGML_TYPE_COUNT&&p.side->in_bounds(ti,sh),"unsupported/out-of-bounds "+ti.name);
            int64_t ne[4]={1,1,1,1};need(ti.shape.size()<=4,"weight rank too large");
            for(size_t i=0;i<ti.shape.size();++i)ne[i]=int64_t(ti.shape[i]);
            p.w[ti.name]=ggml_new_tensor(p.wc,ggml_type(ti.type),int(ti.shape.size()),ne);
        }
        p.wb=ggml_backend_alloc_ctx_tensors_from_buft(p.wc,ggml_backend_get_default_buffer_type(cfg.backend));need(p.wb,"weight allocation failed");
        for(size_t sh=0;sh<p.side->size();++sh)for(const auto & ti:p.side->shard(sh).tensors()) {
            auto t=p.W(ti.name);ggml_backend_tensor_set(t,p.side->shard(sh).tensor_data(ti),0,ggml_nbytes(t));p.bytes+=ggml_nbytes(t);
        }
        p.sc=ggml_init({1024*1024,nullptr,true});need(p.sc,"state metadata failed");
        for(int l=0;l<p.L;++l) {p.keys.push_back(ggml_new_tensor_2d(p.sc,GGML_TYPE_F16,p.DH*p.HK,p.CAP));p.values.push_back(ggml_new_tensor_2d(p.sc,GGML_TYPE_F16,p.DH*p.HK,p.CAP));}
        p.sb=ggml_backend_alloc_ctx_tensors_from_buft(p.sc,ggml_backend_get_default_buffer_type(cfg.backend));need(p.sb,"KV ring allocation failed");
        ggml_backend_buffer_clear(p.sb,0);p.filled.assign(size_t(p.WINDOW),-1);p.ready=true;
        size_t free_after=0,total_after=0;
        ggml_backend_dev_memory(ggml_backend_get_device(cfg.backend),&free_after,&total_after);
        std::fprintf(stderr,"M31 DSpark memory free_before=%zu free_after=%zu total_bytes=%zu weight_bytes=%llu ring_bytes=%zu\n",free_before,free_after,total_after,(unsigned long long)p.bytes,ggml_backend_buffer_get_size(p.sb));
        return true;
    }catch(const std::exception & e){error=e.what();return false;}
}
void DSpark::reset(){p_->pos=0;p_->poisoned=false;std::fill(p_->filled.begin(),p_->filled.end(),-1);}
bool DSpark::append(const float * features,int64_t first,int count,std::string & error) {
    try {
        auto & p=*p_;need(p.ready&&!p.poisoned,"not initialized or reset required");
        need(features&&count>0&&first>=p.pos&&first<=INT32_MAX-count,"invalid/non-monotonic feature positions");
        p.poisoned=true;auto started=std::chrono::steady_clock::now();
        for(int begin=0;begin<count;) {
            int n=std::min(count-begin,p.cfg.max_append);auto & g=p.build_append(n);
            std::vector<int32_t> positions(size_t(n),0);std::vector<int64_t> slots(size_t(n),0);
            for(int i=0;i<n;++i){positions[size_t(i)]=int32_t(first+begin+i);slots[size_t(i)]=(first+begin+i)%p.WINDOW;}
            ggml_backend_tensor_set(g.input,features+size_t(begin)*feature_dim(),0,ggml_nbytes(g.input));
            ggml_backend_tensor_set(g.positions,positions.data(),0,ggml_nbytes(g.positions));
            ggml_backend_tensor_set(g.slots,slots.data(),0,ggml_nbytes(g.slots));g.compute();
            for(int i=0;i<n;++i)p.filled[size_t(slots[size_t(i)])]=positions[size_t(i)];begin+=n;
        }
        p.pos=first+count;p.poisoned=false;p.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return true;
    }catch(const std::exception&e){error=e.what();return false;}
}
bool DSpark::draft(int32_t anchor,int depth,float p_min,std::vector<int32_t>&ids,std::vector<float>&confidence,std::string&error) {
    ids.clear();confidence.clear();
    try {
        auto & p=*p_;need(p.ready&&!p.poisoned,"not initialized or reset required");
        need(anchor>=0&&anchor<p.VOCAB&&depth>=1&&depth<=p.BLOCK&&p_min>=0&&p_min<=1&&std::isfinite(p_min)&&p.pos<=INT32_MAX-depth,"invalid draft arguments");
        auto started=std::chrono::steady_clock::now();auto & g=p.build_draft(depth);
        std::vector<float> embeddings(size_t(p.D)*depth);std::vector<int32_t> positions(size_t(depth),0);std::vector<int64_t> slots(size_t(depth),0);
        auto * trait=ggml_get_type_traits(ggml_type(p.embedding->type));need(p.embedding->type==GGML_TYPE_F32||trait->to_float,"embedding decoder unavailable");
        auto * data=p.target->shard(p.embedding_shard).tensor_data(*p.embedding);
        const size_t row_bytes=ggml_row_size(ggml_type(p.embedding->type),p.D);
        for(int i=0;i<depth;++i) {
            const auto token=i? p.MASK:anchor;auto dst=embeddings.data()+size_t(i)*p.D;
            if(p.embedding->type==GGML_TYPE_F32)std::memcpy(dst,data+size_t(token)*row_bytes,size_t(p.D)*sizeof(float));
            else trait->to_float(data+size_t(token)*row_bytes,dst,p.D);
            positions[size_t(i)]=int32_t(p.pos+i);slots[size_t(i)]=p.WINDOW+i;
        }
        std::vector<ggml_fp16_t> mask(size_t(p.CAP)*32,ggml_fp32_to_fp16(-INFINITY));
        for(int q=0;q<depth;++q) {
            const auto query=p.pos+q;
            for(int s=0;s<p.WINDOW;++s)if(p.filled[size_t(s)]>=0&&p.filled[size_t(s)]>=query-p.WINDOW+1&&p.filled[size_t(s)]<p.pos)mask[size_t(q)*p.CAP+s]=0;
            for(int j=0;j<depth;++j)mask[size_t(q)*p.CAP+p.WINDOW+j]=0;
        }
        ggml_backend_tensor_set(g.input,embeddings.data(),0,ggml_nbytes(g.input));ggml_backend_tensor_set(g.positions,positions.data(),0,ggml_nbytes(g.positions));
        ggml_backend_tensor_set(g.slots,slots.data(),0,ggml_nbytes(g.slots));ggml_backend_tensor_set(g.anchor,&anchor,0,sizeof(anchor));ggml_backend_tensor_set(g.mask,mask.data(),0,ggml_nbytes(g.mask));
        // Draft writes only scratch slots. Failure cannot corrupt accepted ring KV.
        g.compute();
        for(int i=0;i<depth;++i){int32_t id;float conf;ggml_backend_tensor_get(g.ids[size_t(i)],&id,0,sizeof(id));ggml_backend_tensor_get(g.confidence[size_t(i)],&conf,0,sizeof(conf));if(conf<p_min)break;ids.push_back(id);confidence.push_back(conf);}
        p.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return true;
    }catch(const std::exception&e){error=e.what();return false;}
}
} // namespace strata::m31
