// Opt-in end-to-end DSpark research driver. Greedy only; never used by v8 server.
#include "m31_engine.hpp"
#include "m31_dspark.hpp"
#include "m31_run.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <set>

using namespace strata::m31;
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int32_t argmax(const float * x,int n){return int32_t(std::max_element(x,x+n)-x);}
void check(bool ok,const std::string & why){if(!ok)throw std::runtime_error(why);}
struct Args {
    run::Args target;
    std::string sidecar,mode="spec",verifier="n1";
    int depth=7,count=128;
    float p_min=0;
    std::vector<std::pair<int,float>> sweep;
};
Args parse(int argc,char ** argv) {
    Args a;std::vector<char*> rest{argv[0]};
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--sidecar"||key=="--depth"||key=="--generate"||key=="--p-min"||key=="--mode"||key=="--verifier"||key=="--sweep") {
            check(i+1<argc,"missing "+key);const std::string value=argv[++i];
            if(key=="--sidecar")a.sidecar=value;
            else if(key=="--mode")a.mode=value;
            else if(key=="--verifier")a.verifier=value;
            else if(key=="--sweep") {
                std::istringstream list(value);std::string item;
                while(std::getline(list,item,',')){
                    const auto colon=item.find(':');check(colon!=std::string::npos,"sweep syntax is depth:p_min,...");
                    a.sweep.emplace_back(std::stoi(item.substr(0,colon)),std::stof(item.substr(colon+1)));
                }
            }
            else if(key=="--p-min")a.p_min=std::stof(value);
            else if(key=="--depth")a.depth=std::stoi(value);
            else a.count=std::stoi(value);
        } else rest.push_back(argv[i]);
    }
    a.target=run::parse(int(rest.size()),rest.data());
    check(a.depth>=1&&a.depth<=7&&a.count>0&&a.count<=4096&&a.p_min>=0&&a.p_min<=1,"invalid draft/generation parameters");
    check(a.mode=="spec"||a.mode=="serial"||a.mode=="parity"||a.mode=="fixture","mode must be spec|serial|parity|fixture");
    check(a.sweep.empty()||a.mode=="spec","sweep requires mode spec");
    check(a.sweep.size()<=24,"too many sweep arms");
    std::set<std::string> arm_ids;
    for(const auto & [depth,minimum]:a.sweep){
        check(depth>=1&&depth<=7&&std::isfinite(minimum)&&minimum>=0&&minimum<=1,"invalid sweep arm");
        check(arm_ids.insert(std::to_string(depth)+":"+std::to_string(int(std::lround(1000*minimum)))).second,"duplicate sweep arm");
    }
    check(a.verifier=="naive"||a.verifier=="n1","verifier must be naive|n1");
    check(a.mode!="spec"||!a.sidecar.empty(),"spec requires --sidecar");
    check(a.mode!="spec"||a.target.backend=="cuda","DSpark target driver currently requires CUDA");
    check(a.target.gpu_experts_min_tokens>8,"short draft verification must stay on CPU routed FFN (threshold>8)");
    return a;
}
void ids_file(const run::fs::path & file,const std::vector<int32_t>& ids){std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(ids.data()),std::streamsize(ids.size()*4));check(bool(f),"cannot write ids");}
}
int main(int argc,char ** argv) {
    ggml_backend_t gpu=nullptr;
    bool parity_failed=false;
    try {
        auto a=parse(argc,argv);const auto & t=a.target;
        check(run::fs::create_directory(t.out),"output directory exists");
        if(t.backend=="cuda") {
#ifdef M31_HAVE_CUDA
            gpu=ggml_backend_cuda_init(0);check(gpu,"cannot initialize GPU0");
#else
            throw std::runtime_error("built without CUDA");
#endif
        }
        {
            Engine engine;Config cfg;
            cfg.backend=gpu;cfg.threads=t.threads;cfg.context=t.context;cfg.max_chunk=t.chunk;
            cfg.arena_gib=t.arena_gib;cfg.gpu_experts_gib=t.gpu_experts_gib;cfg.gpu_experts_min_tokens=t.gpu_experts_min_tokens;
            cfg.reuse_allocations=t.reuse_allocations;cfg.persistent_cpu_pool=t.persistent_cpu_pool;cfg.cpu_residual=t.cpu_residual;
            cfg.cpu_decode_cache=t.cpu_decode_cache;cfg.dense_decode_cache=t.dense_decode_cache;cfg.profile=t.profile;
            if(a.mode=="spec"||a.mode=="fixture")cfg.draft_feature_layers={4,18,32,46,60};
            std::string error;check(engine.init(t.model,cfg,error),error);const int vocab=int(engine.geometry().vocab);
            std::ofstream summary(run::fs::path(t.out)/"summary.jsonl");
            std::unique_ptr<DSpark> shared_draft;
            std::vector<std::tuple<run::fs::path,int,float,std::string>> tasks;
            for(const auto & file:run::cases(t)) {
                if(a.sweep.empty())tasks.emplace_back(file,a.depth,a.p_min,file.stem().string());
                else for(const auto & [depth,minimum]:a.sweep)
                    tasks.emplace_back(file,depth,minimum,file.stem().string()+"_k"+std::to_string(depth)+"_p"+std::to_string(int(std::lround(1000*minimum))));
            }
            for(const auto & [file,depth,minimum,case_name]:tasks) {
                a.depth=depth;a.p_min=minimum;
                const auto prompt=run::read_ids(file);check(prompt.size()+size_t(a.count+8)<=size_t(t.context),"prompt+generation exceeds context");
                engine.reset();
                const auto capture_reuse=std::getenv("M31_DSPARK_CAPTURE_REUSE");
                if((a.mode=="spec"||a.mode=="fixture")&&capture_reuse&&std::string(capture_reuse)=="1")
                    check(engine.set_draft_feature_window(std::max<int64_t>(0,int64_t(prompt.size())-4096),error),error);
                if(shared_draft){shared_draft->reset();check(engine.set_external_resident_bytes(shared_draft->resident_bytes(),error),error);}
                std::vector<float> logits,seed;
                std::vector<std::tuple<int,int,int64_t>> prefix_chunks;
                int64_t seed_first=0;int seed_rows=0;
                const size_t width=size_t(engine.geometry().embd)*5;
                const auto prefill_start=Clock::now();
                for(int first=0;first<int(prompt.size());) {
                    int n=engine.prefill_chunk(std::min(t.chunk,int(prompt.size())-first),error);check(n>0,error);
                    check(engine.forward(prompt.data()+first,n,error),error);
                    if(a.mode=="spec"||a.mode=="fixture") {
                        int64_t pos;int rows;std::vector<float> values;check(engine.draft_features(pos,rows,values,error),error);
                        if(rows==0)check(values.empty()&&pos==engine.position(),"invalid empty prefill feature slice");
                        else {
                        if(seed.empty()||pos!=seed_first+seed_rows){seed=std::move(values);seed_first=pos;seed_rows=rows;}
                        else {seed.insert(seed.end(),values.begin(),values.end());seed_rows+=rows;}
                        if(seed_rows>4096){const int drop=seed_rows-4096;seed.erase(seed.begin(),seed.begin()+size_t(drop)*width);seed_first+=drop;seed_rows-=drop;}
                        }
                    }
                    const int64_t keys=(int64_t(first)+n+std::max<int64_t>(256,engine.geometry().block)-1)/std::max<int64_t>(256,engine.geometry().block)*std::max<int64_t>(256,engine.geometry().block);
                    prefix_chunks.emplace_back(first,n,keys);
                    first+=n;
                    if(first==int(prompt.size()))check(engine.logits(n-1,1,logits,error),error);
                }
                const double prefill_ms=ms(prefill_start);const int64_t boundary=engine.position();
                int32_t next=argmax(logits.data(),vocab);std::vector<int32_t> output;
                summary<<"{\"case\":"<<run::quote(case_name)<<",\"input_case\":"<<run::quote(file.stem().string())<<",\"mode\":\"prefill_receipt\",\"context\":"<<t.context<<",\"max_chunk\":"<<t.chunk<<",\"draft_resident_bytes\":"<<(shared_draft?shared_draft->resident_bytes():0)<<",\"chunks\":[";
                for(size_t i=0;i<prefix_chunks.size();++i){const auto [position,count,keys]=prefix_chunks[i];summary<<(i?",":"")<<"["<<position<<","<<count<<","<<keys<<"]";}
                summary<<"]}\n";summary.flush();
                if(a.mode=="fixture") {
                    std::ofstream raw(run::fs::path(t.out)/(case_name+".features.f32"),std::ios::binary);
                    raw.write(reinterpret_cast<const char*>(seed.data()),std::streamsize(seed.size()*sizeof(float)));
                    check(bool(raw),"cannot write features");
                    summary<<"{\"case\":"<<run::quote(case_name)<<",\"mode\":\"fixture\",\"first_position\":"<<seed_first<<",\"rows\":"<<seed_rows<<",\"feature_dim\":"<<width<<",\"anchor\":"<<next<<",\"prefill_ms\":"<<prefill_ms<<"}\n";
                } else if(a.mode=="serial"||a.mode=="parity") {
                    std::vector<float> reference;reference.reserve(size_t(vocab)*a.count);
                    const auto serial_start=Clock::now();
                    for(int i=0;i<a.count;++i){output.push_back(next);check(engine.forward(&next,1,error),error);check(engine.logits(0,1,logits,error),error);if(a.mode=="parity")reference.insert(reference.end(),logits.begin(),logits.end());next=argmax(logits.data(),vocab);}
                    const double serial_ms=ms(serial_start);
                    summary<<"{\"case\":"<<run::quote(case_name)<<",\"mode\":\"serial\",\"tokens\":"<<output.size()<<",\"prefill_ms\":"<<prefill_ms<<",\"decode_ms\":"<<serial_ms<<",\"tps\":"<<1000*output.size()/serial_ms<<"}\n";
                    if(a.mode=="parity") {
                        check(engine.rewind(boundary,error),error);int different=0;double max_abs=0,kl_sum=0,kl_max=0;const auto verify_start=Clock::now();
                        for(int first=0;first<a.count;) {
                            int n=std::min(a.depth+1,a.count-first);
                            if(a.verifier=="n1")check(engine.verify(output.data()+first,n,logits,error),error);
                            else {check(engine.forward(output.data()+first,n,error),error);check(engine.logits(0,n,logits,error),error);}
                            for(int row=0;row<n;++row) {
                                const float * baseline=reference.data()+size_t(first+row)*vocab,*actual=logits.data()+size_t(row)*vocab;
                                different+=argmax(baseline,vocab)!=argmax(actual,vocab);
                                double s0=0,s1=0;const double m0=*std::max_element(baseline,baseline+vocab),m1=*std::max_element(actual,actual+vocab);
                                for(int v=0;v<vocab;++v){max_abs=std::max(max_abs,std::abs(double(baseline[v])-actual[v]));s0+=std::exp(baseline[v]-m0);s1+=std::exp(actual[v]-m1);}
                                const double z0=m0+std::log(s0),z1=m1+std::log(s1);double kl=0;
                                for(int v=0;v<vocab;++v)kl+=std::exp(baseline[v]-z0)*(baseline[v]-z0-actual[v]+z1);
                                kl_sum+=kl;kl_max=std::max(kl_max,kl);
                            }first+=n;
                        }
                        summary<<"{\"case\":"<<run::quote(case_name)<<",\"mode\":\"batch_parity\",\"verifier\":"<<run::quote(a.verifier)<<",\"depth\":"<<a.depth<<",\"rows\":"<<a.count<<",\"exact_greedy\":"<<(different?"false":"true")<<",\"top1_different\":"<<different<<",\"max_abs\":"<<max_abs<<",\"mean_kl\":"<<kl_sum/a.count<<",\"max_kl\":"<<kl_max<<",\"verify_ms\":"<<ms(verify_start)<<"}\n";
                        // A non-zero result is preserved as evidence, never passed off as exact speculation.
                        if(different)parity_failed=true;
                        if(different)std::fprintf(stderr,"DSpark batch/serial TOP1 MISMATCH: %d/%d\n",different,a.count);
                    }
                } else {
                    check(engine.release_prefill_workspace(error),error);
                    if(!shared_draft){
                        shared_draft=std::make_unique<DSpark>();DSparkConfig dc;dc.backend=cfg.backend;dc.sidecar=a.sidecar;dc.target=t.model;dc.shared_head=engine.output_weight();
                        check(shared_draft->init(dc,error),error);
                    }
                    auto & draft=*shared_draft;
                    check(draft.append(seed.data(),seed_first,seed_rows,error),error);seed.clear();seed.shrink_to_fit();
                    int proposed=0,accepted=0,rounds=0;double draft_ms=0,verify_ms=0;const auto decode_start=Clock::now();
                    while(int(output.size())<a.count) {
                        const int64_t base=engine.position();const int remaining=a.count-int(output.size());
                        const int depth=std::min(a.depth,std::max(0,remaining-1));std::vector<int32_t> candidates;std::vector<float> confidence;
                        if(depth){check(draft.draft(next,depth,a.p_min,candidates,confidence,error),error);draft_ms+=draft.last_ms();}
                        std::vector<int32_t> batch{next};batch.insert(batch.end(),candidates.begin(),candidates.end());
                        const auto verify_start=Clock::now();
                        if(a.verifier=="n1")check(engine.verify(batch.data(),int(batch.size()),logits,error),error);
                        else {check(engine.forward(batch.data(),int(batch.size()),error),error);check(engine.logits(0,int(batch.size()),logits,error),error);}
                        verify_ms+=ms(verify_start);
                        int matched=0;for(;matched<int(candidates.size());++matched)if(argmax(logits.data()+size_t(matched)*vocab,vocab)!=candidates[size_t(matched)])break;
                        const int committed=matched+1;next=argmax(logits.data()+size_t(matched)*vocab,vocab);
                        int64_t features_pos;int features_rows;std::vector<float> features;check(engine.draft_features(features_pos,features_rows,features,error),error);
                        check(features_pos==base&&features_rows==int(batch.size()),"verify feature row geometry differs");
                        check(draft.append(features.data(),base,committed,error),error);
                        if(committed<int(batch.size()))check(engine.rewind(base+committed,error),error);
                        output.insert(output.end(),batch.begin(),batch.begin()+committed);proposed+=int(candidates.size());accepted+=matched;++rounds;
                        summary<<"{\"case\":"<<run::quote(case_name)<<",\"mode\":\"round\",\"position\":"<<base<<",\"proposed\":"<<candidates.size()<<",\"accepted\":"<<matched<<",\"committed\":"<<committed<<"}\n";summary.flush();
                    }
                    const double decode_ms=ms(decode_start);
                    summary<<"{\"case\":"<<run::quote(case_name)<<",\"mode\":\"spec\",\"verifier\":"<<run::quote(a.verifier)<<",\"depth\":"<<a.depth<<",\"p_min\":"<<a.p_min<<",\"tokens\":"<<output.size()<<",\"rounds\":"<<rounds<<",\"proposed\":"<<proposed<<",\"accepted\":"<<accepted<<",\"acceptance\":"<<(proposed?double(accepted)/proposed:0)<<",\"prefill_ms\":"<<prefill_ms<<",\"draft_ms\":"<<draft_ms<<",\"verify_ms\":"<<verify_ms<<",\"decode_ms\":"<<decode_ms<<",\"tps\":"<<1000*output.size()/decode_ms<<"}\n";
                }
                ids_file(run::fs::path(t.out)/(case_name+".generated.i32"),output);summary.flush();
            }
        }
        if(gpu)ggml_backend_free(gpu);return parity_failed?2:0;
    }catch(const std::exception & e){std::fprintf(stderr,"m31_dspark_generate: %s\n",e.what());if(gpu)ggml_backend_free(gpu);return 1;}
}
