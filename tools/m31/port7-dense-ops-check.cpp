#include "m31_geometry.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using Json=nlohmann::json;
namespace fs=std::filesystem;
namespace strata::m31 {
#include "m31_dspark_batched_ops.hpp"
}
double dense_ops_graph_us(ggml_backend_t backend,ggml_cgraph * graph,int repeats);
static void need(bool value,const char * reason){if(!value)throw std::runtime_error(reason);}
static std::vector<float> floats(const fs::path & path){
    const auto bytes=fs::file_size(path);need(bytes&&bytes%4==0,"invalid F32 fixture");
    std::vector<float> out(size_t(bytes/4));std::ifstream in(path,std::ios::binary);
    in.read(reinterpret_cast<char *>(out.data()),std::streamsize(bytes));need(bool(in),"short F32 fixture");return out;
}
struct GraphOwner {
    ggml_backend_t backend;ggml_context * c=nullptr;ggml_cgraph * graph=nullptr;ggml_backend_buffer_t buffer=nullptr;
    explicit GraphOwner(ggml_backend_t b):backend(b){c=ggml_init({8*1024*1024,nullptr,true});need(c,"graph metadata");graph=ggml_new_graph(c);}
    ~GraphOwner(){ggml_backend_synchronize(backend);ggml_backend_cuda_graph_cache_clear(backend);if(buffer)ggml_backend_buffer_free(buffer);if(c)ggml_free(c);}
    void finish(const std::vector<ggml_tensor *> & outputs){
        for(auto * out:outputs){ggml_set_output(out);ggml_build_forward_expand(graph,out);}
        buffer=ggml_backend_alloc_ctx_tensors(c,backend);need(buffer,"graph tensor allocation");
    }
    void compute(){need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"stock op compute");}
};
class Weights {
    struct Owner{ggml_context * c=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_tensor * tensor=nullptr;
        ~Owner(){if(buffer)ggml_backend_buffer_free(buffer);if(c)ggml_free(c);}};
    const GgufModel & model;ggml_backend_t backend;
    std::map<std::string,std::unique_ptr<Owner>> values;
public:
    Weights(const GgufModel & m,ggml_backend_t b):model(m),backend(b){}
    ggml_tensor * get(const std::vector<std::string> & names){
        std::string key;for(const auto & n:names)key+=n+";";
        auto found=values.find(key);if(found!=values.end())return found->second->tensor;
        std::vector<const TensorInfo *> parts;std::vector<size_t> shards;
        for(const auto & n:names){size_t shard=0;auto info=model.find(n,&shard);need(info&&model.in_bounds(*info,shard),"missing/invalid weight");parts.push_back(info);shards.push_back(shard);}
        auto owned=std::make_unique<Owner>();owned->c=ggml_init({64*1024,nullptr,true});need(owned->c,"weight metadata");
        if(parts.size()==1){int64_t shape[4]={1,1,1,1};for(size_t i=0;i<parts[0]->shape.size();++i)shape[i]=int64_t(parts[0]->shape[i]);
            owned->tensor=ggml_new_tensor(owned->c,ggml_type(parts[0]->type),int(parts[0]->shape.size()),shape);
        }else{int64_t rows=0;for(auto * part:parts){need(part->shape.size()==2&&part->shape[0]==parts[0]->shape[0]&&part->type==parts[0]->type,"incompatible unchanged weight bundle");rows+=int64_t(part->shape[1]);}
            owned->tensor=ggml_new_tensor_2d(owned->c,ggml_type(parts[0]->type),int64_t(parts[0]->shape[0]),rows);}
        owned->buffer=ggml_backend_alloc_ctx_tensors(owned->c,backend);need(owned->buffer,"weight allocation");ggml_backend_buffer_set_usage(owned->buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t offset=0;for(size_t i=0;i<parts.size();++i){const size_t bytes=tensor_payload_bytes(*parts[i]);ggml_backend_tensor_set(owned->tensor,model.shard(shards[i]).tensor_data(*parts[i]),offset,bytes);offset+=bytes;}
        auto * result=owned->tensor;values.emplace(key,std::move(owned));return result;
    }
    ggml_tensor * get(const std::string & name){return get(std::vector<std::string>{name});}
};
static std::vector<float> project_n1(ggml_backend_t backend,ggml_tensor * weight,const std::vector<float> & input){
    const int count=int(input.size()/size_t(weight->ne[0]));need(input.size()==size_t(count)*size_t(weight->ne[0]),"projection fixture dimensions");
    GraphOwner b(backend);auto x=ggml_new_tensor_2d(b.c,GGML_TYPE_F32,weight->ne[0],1);ggml_set_input(x);auto out=ggml_mul_mat(b.c,weight,x);b.finish({out});
    std::vector<float> result(size_t(count)*size_t(weight->ne[1]));
    for(int row=0;row<count;++row){ggml_backend_tensor_set(x,input.data()+size_t(row)*size_t(weight->ne[0]),0,size_t(weight->ne[0])*4);b.compute();ggml_backend_tensor_get(out,result.data()+size_t(row)*size_t(weight->ne[1]),0,size_t(weight->ne[1])*4);}
    return result;
}
static std::vector<float> slice_rows(const std::vector<float> & input,int64_t stride,int64_t offset,int64_t width,int count=4){
    need(input.size()>=size_t(stride)*size_t(count)&&offset+width<=stride,"projection slice dimensions");std::vector<float> out(size_t(width)*size_t(count));
    for(int row=0;row<count;++row)std::copy_n(input.data()+size_t(row)*size_t(stride)+size_t(offset),size_t(width),out.data()+size_t(row)*size_t(width));return out;
}
struct Scenario {
    std::string name,kind;int layer=0;int64_t dimension=0,heads=1;ggml_tensor * weight=nullptr,*bias=nullptr;
    std::vector<float> x,u;
};
class OpsGraph:public GraphOwner {
public:
    ggml_tensor * input=nullptr,*up=nullptr,*positions=nullptr;
    std::vector<ggml_tensor *> outputs;std::vector<std::string> names;
    OpsGraph(ggml_backend_t backend,const m31::Geometry & g,const Scenario & s,int count,int start,bool scalar):GraphOwner(backend){
#ifdef M31_OPS_PRODUCTION_HELPERS
        const bool production=true;
#else
        const bool production=false;
#endif
        const bool packed=s.kind=="glu_packed";const int64_t input_d=packed?2*s.dimension:s.dimension;
        input=ggml_new_tensor_3d(c,GGML_TYPE_F32,input_d,s.heads,count);ggml_set_input(input);
        if(s.kind=="glu"){up=ggml_new_tensor_3d(c,GGML_TYPE_F32,input_d,s.heads,count);ggml_set_input(up);}
        positions=ggml_new_tensor_1d(c,GGML_TYPE_I32,count);ggml_set_input(positions);
        auto build=[&](ggml_tensor * x,ggml_tensor * u,ggml_tensor * pos,int columns,bool scalar_row){
            std::vector<ggml_tensor *> result;
            if(s.kind=="rms"||s.kind=="norm_rope"){
                auto norm=ggml_rms_norm(c,x,g.rms_eps);auto weighted=ggml_mul(c,norm,s.weight);
                result={norm,weighted};names={"rms","weighted"};
                if(s.kind=="norm_rope"){
                    result.push_back(ggml_rope_ext(c,weighted,pos,nullptr,int(g.rope_dim),GGML_ROPE_TYPE_NEOX,int(g.context),g.rope_base,1,0,1,32,1));names.push_back("rope");}
            }else if(s.kind=="glu") {result={ggml_swiglu_oai(c,x,u,1.702f,7.f)};names={"glu"};}
            else if(s.kind=="glu_packed"){
                if(scalar_row){auto gate=ggml_cont(c,ggml_view_3d(c,x,s.dimension,s.heads,1,x->nb[1],x->nb[2],0));
                    auto other=ggml_cont(c,ggml_view_3d(c,x,s.dimension,s.heads,1,x->nb[1],x->nb[2],size_t(s.dimension)*4));result={ggml_swiglu_oai(c,gate,other,1.702f,7.f)};}
                else result={ggml_swiglu_oai(c,x,nullptr,1.702f,7.f)};
                names={"glu_packed"};
            }else if(s.kind=="router"){
                auto probs=ggml_sigmoid(c,ggml_reshape_2d(c,x,g.experts,columns));
                auto ids=ggml_top_k(c,ggml_add(c,probs,s.bias),int(g.used));
                auto weights=ggml_get_rows(c,ggml_reshape_3d(c,probs,1,g.experts,columns),ids);
                weights=ggml_reshape_2d(c,weights,g.used,columns);
                if(g.norm_routes){auto sum=ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY);weights=ggml_div(c,weights,sum);}
                if(g.route_scale!=1)weights=ggml_scale(c,weights,g.route_scale);
                result={probs,ggml_cont(c,ids),weights};names={"probabilities","ids","weights"};
            }else throw std::runtime_error("unknown component kind");
            if(production){
                if(s.kind=="rms"){result={result[1]};names={"weighted"};}
                if(s.kind=="norm_rope"){result={result.back()};names={"rope"};}
                if(s.kind=="router"){result={result[1],result[2]};names={"ids","weights"};}
            }
            return result;
        };
        if(scalar){for(int row=0;row<count;++row){
            auto x=ggml_view_3d(c,input,input_d,s.heads,1,input->nb[1],input->nb[2],size_t(row)*input->nb[2]);
            auto u=up?ggml_view_3d(c,up,input_d,s.heads,1,up->nb[1],up->nb[2],size_t(row)*up->nb[2]):nullptr;
            auto pos=ggml_view_1d(c,positions,1,size_t(row)*4);auto values=build(x,u,pos,1,true);
            if(outputs.empty())outputs=values;
            else for(size_t i=0;i<values.size();++i)outputs[i]=ggml_concat(c,outputs[i],values[i],s.kind=="router"?1:2);
        }}else if(production){
            std::vector<ggml_tensor *> rows;
            for(int row=0;row<count;++row)rows.push_back(ggml_view_3d(c,input,input_d,s.heads,1,input->nb[1],input->nb[2],size_t(row)*input->nb[2]));
            auto join=[&](const std::vector<ggml_tensor *> & values,int axis){ggml_tensor * out=nullptr;for(auto * value:values)out=out?ggml_concat(c,out,value,axis):value;return out;};
            if(s.kind=="rms") {outputs={join(m31::dspark_ops_rms_rows(c,rows,s.weight,g.rms_eps),2)};names={"weighted"};}
            else if(s.kind=="norm_rope") {outputs={join(m31::dspark_ops_norm_rope_rows(c,rows,s.weight,g.rms_eps,positions,g),2)};names={"rope"};}
            else if(s.kind=="router") {auto r=m31::dspark_ops_router_rows(c,rows,s.bias,g);outputs={join(r.ids,1),join(r.weights,1)};names={"ids","weights"};}
            else if(s.heads==1&&(s.kind=="glu"||s.kind=="glu_packed")){
                auto a=ggml_reshape_2d(c,input,input_d,count);auto other=up?ggml_reshape_2d(c,up,input_d,count):nullptr;
                outputs={ggml_reshape_3d(c,m31::dspark_ops_glu(c,a,other),s.dimension,1,count)};names={s.kind=="glu"?"glu":"glu_packed"};
            }else outputs=build(input,up,positions,count,false); // routed-rank shape tests raw stock API only
        }else outputs=build(input,up,positions,count,false);
        for(auto & out:outputs)out=ggml_cont(c,out);finish(outputs);
        const size_t bytes=size_t(input_d)*size_t(s.heads)*size_t(count)*4;need(s.x.size()*4>=bytes,"short real input");ggml_backend_tensor_set(input,s.x.data(),0,bytes);
        if(up){need(s.u.size()*4>=bytes,"short real GLU up");ggml_backend_tensor_set(up,s.u.data(),0,bytes);}
        std::vector<int32_t> absolute(size_t(count),0);for(int i=0;i<count;++i)absolute[size_t(i)]=start+i;
        ggml_backend_tensor_set(positions,absolute.data(),0,absolute.size()*4);
    }
};
static Json compare(OpsGraph & reference,OpsGraph & batch){
    Json outputs=Json::array();
    for(size_t i=0;i<reference.outputs.size();++i){auto * a=reference.outputs[i];auto * b=batch.outputs[i];
        need(a->type==b->type&&ggml_nelements(a)==ggml_nelements(b),"component output shape mismatch");
        const size_t values=size_t(ggml_nelements(a));std::vector<uint32_t> expected(values),actual(values);
        ggml_backend_tensor_get(a,expected.data(),0,values*4);ggml_backend_tensor_get(b,actual.data(),0,values*4);
        size_t different=0;double max_abs=0;
        for(size_t k=0;k<values;++k){different+=expected[k]!=actual[k];if(a->type==GGML_TYPE_F32){float x,y;std::memcpy(&x,&expected[k],4);std::memcpy(&y,&actual[k],4);need(std::isfinite(x)&&std::isfinite(y),"nonfinite component output");max_abs=std::max(max_abs,std::abs(double(x)-y));}}
        outputs.push_back({{"name",reference.names[i]},{"type",ggml_type_name(a->type)},{"values",values},{"different",different},{"max_abs",max_abs}});
    }
    return outputs;
}
int main(int argc,char ** argv){ggml_backend_t backend=nullptr;try{
    need(argc==4,"usage: aictl GPU0 component MODEL FIXTURE OUT.json");
    need(std::getenv("M31_CARD_SESSION")&&std::strcmp(std::getenv("M31_CARD_SESSION"),"1")==0&&std::getenv("CUDA_VISIBLE_DEVICES")&&std::strcmp(std::getenv("CUDA_VISIBLE_DEVICES"),"0")==0,"requires authorized GPU0 session");
    need(!fs::exists(argv[3]),"output exists");const fs::path fixtures=argv[2];std::ifstream input(fixtures/"manifest.json");Json fixture;input>>fixture;
    const auto model=GgufModel::open(argv[1]);const auto g=m31::read_geometry(model);backend=ggml_backend_cuda_init(0);need(backend,"CUDA init");
    Json report={{"scope","Stock unary/reduction operators, real captured model inputs and GGUF weights, scalar n1 subgraphs vs K-row batch. RoPE stress positions replay same inputs independently of original KV history. CUDA-event latency is component only."},{"fixture",fixture},{"cases",Json::array()}};
    {Weights weights(model,backend);for(const auto & row:fixture["layers"]){const int layer=row["layer"];const std::string prefix="blk."+std::to_string(layer)+".";
        const auto attn_norm=floats(fixtures/row["attn_norm"].get<std::string>());const auto ffn_norm=floats(fixtures/row["ffn_norm"].get<std::string>());
        std::vector<Scenario> scenarios;
        auto add_rms=[&](const std::string & name,const std::string & file,const std::string & weight){Scenario s;s.name=name;s.kind="rms";s.layer=layer;s.dimension=g.embd;s.weight=weights.get(prefix+weight);s.x=floats(fixtures/file);scenarios.push_back(std::move(s));};
        if(row.contains("attn_hidden"))add_rms("attn_rms",row["attn_hidden"],"attn_norm.weight");add_rms("ffn_rms",row["ffn_input"],"ffn_norm.weight");
        const auto qkv=project_n1(backend,weights.get(std::vector<std::string>{prefix+"attn_q.weight",prefix+"attn_k.weight",prefix+"attn_v.weight"}),attn_norm);
        const auto index=project_n1(backend,weights.get(std::vector<std::string>{prefix+"indexer.q_proj.weight",prefix+"indexer.k_proj.weight"}),attn_norm);
        auto add_rope=[&](const std::string & name,const std::vector<float> & raw,int64_t stride,int64_t offset,int64_t heads,const std::string & weight){Scenario s;s.name=name;s.kind="norm_rope";s.layer=layer;s.dimension=g.head_dim;s.heads=heads;s.weight=weights.get(prefix+weight);s.x=slice_rows(raw,stride,offset,g.head_dim*heads);scenarios.push_back(std::move(s));};
        add_rope("q_norm_rope",qkv,g.head_dim*(g.heads+2*g.kv_heads),0,g.heads,"attn_q_norm.weight");add_rope("k_norm_rope",qkv,g.head_dim*(g.heads+2*g.kv_heads),g.head_dim*g.heads,g.kv_heads,"attn_k_norm.weight");
        add_rope("index_q_norm_rope",index,g.index_dim*(g.index_heads+1),0,g.index_heads,"indexer.q_norm.weight");add_rope("index_k_norm_rope",index,g.index_dim*(g.index_heads+1),g.index_dim*g.index_heads,1,"indexer.k_norm.weight");
        Scenario glu;glu.name="glu";glu.kind="glu";glu.layer=layer;
        if(g.routed(layer)){glu.dimension=g.expert_ff;glu.heads=g.used;glu.x=floats(fixtures/row["gate"].get<std::string>());glu.u=floats(fixtures/row["up"].get<std::string>());}
        else{glu.dimension=g.dense_ff;const auto gu=project_n1(backend,weights.get(std::vector<std::string>{prefix+"ffn_gate.weight",prefix+"ffn_up.weight"}),ffn_norm);glu.x=slice_rows(gu,2*g.dense_ff,0,g.dense_ff);glu.u=slice_rows(gu,2*g.dense_ff,g.dense_ff,g.dense_ff);}
        scenarios.push_back(glu);Scenario packed=glu;packed.name="glu_packed";packed.kind="glu_packed";packed.x.resize(glu.x.size()*2);packed.u.clear();
        for(int token=0;token<4;++token)for(int head=0;head<glu.heads;++head){const size_t src=(size_t(token)*size_t(glu.heads)+size_t(head))*size_t(glu.dimension),dst=2*src;std::copy_n(glu.x.data()+src,size_t(glu.dimension),packed.x.data()+dst);std::copy_n(glu.u.data()+src,size_t(glu.dimension),packed.x.data()+dst+size_t(glu.dimension));}scenarios.push_back(std::move(packed));
        if(g.routed(layer)){Scenario single=glu;single.name="glu_rank0";single.heads=1;
            single.x=slice_rows(glu.x,glu.dimension*glu.heads,0,glu.dimension);single.u=slice_rows(glu.u,glu.dimension*glu.heads,0,glu.dimension);scenarios.push_back(single);
            Scenario one=single;one.kind="glu_packed";one.name="glu_packed_rank0";one.x.resize(single.x.size()*2);one.u.clear();
            for(int token=0;token<4;++token){const size_t src=size_t(token)*size_t(glu.dimension),dst=2*src;std::copy_n(single.x.data()+src,size_t(glu.dimension),one.x.data()+dst);std::copy_n(single.u.data()+src,size_t(glu.dimension),one.x.data()+dst+size_t(glu.dimension));}scenarios.push_back(std::move(one));}
        if(g.routed(layer)){Scenario s;s.name="router";s.kind="router";s.layer=layer;s.dimension=g.experts;s.bias=weights.get(prefix+"exp_probs_b.bias");s.x=project_n1(backend,weights.get(prefix+"ffn_gate_inp.weight"),ffn_norm);scenarios.push_back(std::move(s));}
        for(const auto & s:scenarios)for(int count:{1,2,3,4}){std::vector<int> starts{row["first_source_token"]};if(s.kind=="norm_rope")starts=fixture["position_starts"].get<std::vector<int>>();for(int start:starts){
            OpsGraph scalar(backend,g,s,count,start,true),batch(backend,g,s,count,start,false);for(int repeat=0;repeat<4;++repeat){scalar.compute();batch.compute();}
            auto outputs=compare(scalar,batch);bool exact=true;for(const auto & out:outputs)exact=exact&&out["different"].get<uint64_t>()==0;
            const double scalar_us=dense_ops_graph_us(backend,scalar.graph,32),batch_us=dense_ops_graph_us(backend,batch.graph,32);
            bool production_helper=false;
#ifdef M31_OPS_PRODUCTION_HELPERS
            production_helper=!(s.heads>1&&(s.kind=="glu"||s.kind=="glu_packed"));
#endif
            report["cases"].push_back({{"layer",layer},{"role",s.name},{"K",count},{"position_start",start},{"dimension",s.dimension},{"heads",s.heads},{"production_helper",production_helper},{"outputs",outputs},{"exact",exact},{"scalar_graph_us",scalar_us},{"batch_graph_us",batch_us}});
            std::fprintf(stderr,"ops layer=%d role=%s K=%d position=%d exact=%d scalar_us=%.3f batch_us=%.3f\n",layer,s.name.c_str(),count,start,int(exact),scalar_us,batch_us);
        }}
    }}
    ggml_backend_free(backend);backend=nullptr;std::ofstream output(argv[3]);output<<report.dump(2)<<'\n';std::printf("{\"complete\":true,\"cases\":%zu}\n",report["cases"].size());return 0;
}catch(const std::exception & e){std::fprintf(stderr,"stock ops component: %s\n",e.what());if(backend)ggml_backend_free(backend);return 1;}}
