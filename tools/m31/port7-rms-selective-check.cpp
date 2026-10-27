#define main previous_ops_main
#include "port7-dense-ops-check.cpp"
#undef main
void selective_rms_compute(ggml_backend_t,ggml_tensor *,ggml_tensor *,bool);
double selective_rms_us(ggml_backend_t,ggml_tensor *,ggml_tensor *,bool);
int main(int argc,char **argv){ggml_backend_t backend=nullptr;try{
    need(argc==4,"usage MODEL FIXTURE OUT.json");need(std::getenv("M31_CARD_SESSION")&&std::string(std::getenv("M31_CARD_SESSION"))=="1","requires card session");need(!fs::exists(argv[3]),"output exists");
    auto model=GgufModel::open(argv[1]);auto g=m31::read_geometry(model);fs::path dir=argv[2];Json fixture;{std::ifstream in(dir/"manifest.json");in>>fixture;}
    backend=ggml_backend_cuda_init(0);need(backend,"backend");Json report={{"pass",true},{"cases",Json::array()},{"scope","Only stock rms_norm_fused vs stock rms_norm followed by mul; original CUDA archive and compile flags; graph-replay event times are isolated, not token generation."}};size_t compared=0;
    {Weights weights(model,backend);for(const auto & row:fixture["layers"]){int layer=row["layer"];std::string p="blk."+std::to_string(layer)+".";std::vector<Scenario> scenarios;
        auto add=[&](std::string name,int64_t dim,int64_t heads,std::vector<float> x,std::string weight){Scenario s;s.name=name;s.dimension=dim;s.heads=heads;s.x=std::move(x);s.weight=weights.get(p+weight);scenarios.push_back(std::move(s));};
        auto an=floats(dir/row["attn_norm"].get<std::string>());
        add("ffn",g.embd,1,floats(dir/row["ffn_input"].get<std::string>()),"ffn_norm.weight");
        if(row.contains("attn_hidden"))add("attn",g.embd,1,floats(dir/row["attn_hidden"].get<std::string>()),"attn_norm.weight");
        auto qkv=project_n1(backend,weights.get(std::vector<std::string>{p+"attn_q.weight",p+"attn_k.weight",p+"attn_v.weight"}),an);
        auto iqk=project_n1(backend,weights.get(std::vector<std::string>{p+"indexer.q_proj.weight",p+"indexer.k_proj.weight"}),an);
        add("q",g.head_dim,g.heads,slice_rows(qkv,g.head_dim*(g.heads+2*g.kv_heads),0,g.head_dim*g.heads),"attn_q_norm.weight");
        add("k",g.head_dim,g.kv_heads,slice_rows(qkv,g.head_dim*(g.heads+2*g.kv_heads),g.head_dim*g.heads,g.head_dim*g.kv_heads),"attn_k_norm.weight");
        add("iq",g.index_dim,g.index_heads,slice_rows(iqk,g.index_dim*(g.index_heads+1),0,g.index_dim*g.index_heads),"indexer.q_norm.weight");
        add("ik",g.index_dim,1,slice_rows(iqk,g.index_dim*(g.index_heads+1),g.index_dim*g.index_heads,g.index_dim),"indexer.k_norm.weight");
        for(auto &s:scenarios)for(int token=0;token<4;++token){GraphOwner b(backend);auto x=ggml_new_tensor_3d(b.c,GGML_TYPE_F32,s.dimension,s.heads,1);auto norm=ggml_rms_norm(b.c,x,g.rms_eps);auto mul=ggml_mul(b.c,norm,s.weight);b.finish({mul});size_t n=size_t(s.dimension*s.heads);ggml_backend_tensor_set(x,s.x.data()+token*n,0,n*4);
            selective_rms_compute(backend,norm,mul,false);std::vector<uint32_t> ref(n),got(n);ggml_backend_tensor_get(mul,ref.data(),0,n*4);selective_rms_compute(backend,norm,mul,true);ggml_backend_tensor_get(mul,got.data(),0,n*4);size_t diff=0;for(size_t i=0;i<n;++i)diff+=ref[i]!=got[i];compared+=n;
            Json rec={{"layer",layer},{"role",s.name},{"token",token},{"values",n},{"different",diff}};if(token==0){rec["unfused_us"]=selective_rms_us(backend,norm,mul,false);rec["fused_us"]=selective_rms_us(backend,norm,mul,true);}report["cases"].push_back(rec);if(diff)report["pass"]=false;
        }
    }}
    report["values"]=compared;std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';out.flush();need(bool(out),"report write");std::printf("cases=%zu values=%zu pass=%d\n",report["cases"].size(),compared,int(report["pass"].get<bool>()));ggml_backend_free(backend);return report["pass"].get<bool>()?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"RMS selective: %s\n",e.what());if(backend)ggml_backend_free(backend);return 2;}}
