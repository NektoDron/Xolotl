// Isolated acceptance dumper: real Engine.verify, partial commits and rewind.
// Correctness artifact only. It never acquires a card or changes a runtime profile.
#include "m31_engine.hpp"
#include "m31_run.hpp"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace strata::m31;
namespace {
void need(bool ok,const std::string & why){if(!ok)throw std::runtime_error(why);}
struct Args {run::Args target;int count=128;bool require_bitwise=true;};
Args parse(int argc,char ** argv){
    Args a;std::vector<char*> rest{argv[0]};
    for(int i=1;i<argc;++i){const std::string k=argv[i];
        if(k=="--generate"||k=="--require-bitwise"){
            need(i+1<argc,"missing value for "+k);const std::string v=argv[++i];size_t end=0;
            const int n=std::stoi(v,&end);need(end==v.size(),"invalid integer for "+k);
            if(k=="--generate")a.count=n;else {need(n==0||n==1,"require-bitwise must be0|1");a.require_bitwise=n==1;}
        }else rest.push_back(argv[i]);
    }
    a.target=run::parse(int(rest.size()),rest.data());
    need(a.count>=8&&a.count<=4096,"generate must be8..4096");
    need(a.target.taps=="none"&&!a.target.oracle_op_offload,"verify dump accepts no taps/oracle placement");
    need(a.target.gpu_experts_min_tokens>8,"verify short batches must use decode experts (threshold>8)");
    // The emitted dump explicitly defines its tail selection and head geometry.
    a.target.rows="last:"+std::to_string(a.count+1);a.target.head_tail=1;a.target.decode_tail=a.count;
    return a;
}
void binary(const run::fs::path & path,const std::vector<float> & values){
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(float)));need(bool(f),"cannot write "+path.string());
}
struct Difference {
    uint64_t values=0,different=0,nonfinite=0,rows=0,top1_different=0;double max_abs=0;
    void compare(const float * expected,const float * actual,size_t count,bool logits=false){
        ++rows;values+=count;
        if(logits)top1_different+=std::max_element(expected,expected+count)-expected!=std::max_element(actual,actual+count)-actual;
        for(size_t i=0;i<count;++i){
            different+=std::memcmp(expected+i,actual+i,sizeof(float))!=0;
            if(!std::isfinite(expected[i])||!std::isfinite(actual[i]))++nonfinite;
            else max_abs=std::max(max_abs,std::abs(double(expected[i])-double(actual[i])));
        }
    }
    bool exact()const{return !different&&!nonfinite;}
    void json(std::ostream & out)const{out<<"{\"rows\":"<<rows<<",\"values\":"<<values<<",\"different_f32_bits\":"<<different<<",\"nonfinite\":"<<nonfinite<<",\"max_abs\":"<<std::setprecision(17)<<max_abs<<",\"top1_different\":"<<top1_different<<'}';}
};
struct PrefixChunk {int position,n;int64_t nk;};
struct Call {int64_t position;int n,committed;};
std::vector<float> features(Engine & engine,int64_t first,int n,size_t width,std::string & error){
    int64_t got_first=-1;int got_n=0;std::vector<float> out;
    need(engine.draft_features(got_first,got_n,out,error),error);
    need(got_first==first&&got_n==n&&out.size()==size_t(n)*width,"captured features have wrong position/count/shape");return out;
}
std::vector<float> anchor_features(Engine & engine,int64_t position,size_t width,std::string & error){
    int64_t first=-1;int n=0;std::vector<float> out;need(engine.draft_features(first,n,out,error),error);
    need(n>0&&first+n==position+1&&out.size()==size_t(n)*width,"anchor features have wrong position/count/shape");
    return {out.end()-std::ptrdiff_t(width),out.end()};
}
}
int main(int argc,char ** argv){
    ggml_backend_t gpu=nullptr;bool any_mismatch=false;
    try{
        const auto a=parse(argc,argv);const auto & t=a.target;const auto cases=run::cases(t);
        need(run::fs::create_directory(t.out),"output directory exists; use a new path");
        if(t.backend=="cuda"){
#ifdef M31_HAVE_CUDA
            gpu=ggml_backend_cuda_init(0);need(gpu,"cannot initialize GPU0");
#else
            throw std::runtime_error("built without CUDA");
#endif
        }
        {
            Engine engine;Config c;c.backend=gpu;c.threads=t.threads;c.context=t.context;c.max_chunk=t.chunk;
            c.arena_gib=t.arena_gib;c.gpu_experts_gib=t.gpu_experts_gib;c.gpu_experts_min_tokens=t.gpu_experts_min_tokens;
            c.reuse_allocations=t.reuse_allocations;c.persistent_cpu_pool=t.persistent_cpu_pool;c.cpu_residual=t.cpu_residual;
            c.cpu_decode_cache=t.cpu_decode_cache;c.dense_decode_cache=t.dense_decode_cache;c.profile=t.profile;
            c.draft_feature_layers={4,18,32,46,60};c.draft_feature_max_rows=4096;
            std::string error;need(engine.init(t.model,c,error),error);const int vocab=int(engine.geometry().vocab);
            const size_t width=size_t(engine.geometry().embd)*c.draft_feature_layers.size();
            for(const auto & path:cases){
                const auto prompt=run::read_ids(path);const int p=int(prompt.size());
                need(p>=1&&int64_t(p)+a.count<=t.context,"prompt+tail exceeds context");
                std::vector<PrefixChunk> prefix;std::vector<Call> calls;std::vector<int32_t> combined=prompt;
                std::vector<float> serial,serial_features,logits,baseline_anchor_feature;
                serial.reserve(size_t(a.count+1)*vocab);serial_features.reserve(size_t(a.count+1)*width);
                engine.reset();
                for(int first=0;first<p;){
                    const int n=engine.prefill_chunk(std::min(t.chunk,p-first),error);need(n>0,error);
                    need(engine.forward(prompt.data()+first,n,error),error);
                    const int64_t pad=std::max<int64_t>(256,engine.geometry().block);
                    prefix.push_back({first,n,(int64_t(first)+n+pad-1)/pad*pad});first+=n;
                    if(first==p)need(engine.logits(n-1,1,logits,error),error);
                }
                need(logits.size()==size_t(vocab),"anchor logits shape");
                serial.insert(serial.end(),logits.begin(),logits.end());
                baseline_anchor_feature=anchor_features(engine,p-1,width,error);
                serial_features.insert(serial_features.end(),baseline_anchor_feature.begin(),baseline_anchor_feature.end());
                need(engine.release_prefill_workspace(error),error);
                for(int i=0;i<a.count;++i){
                    const int32_t token=int32_t(std::max_element(logits.begin(),logits.end())-logits.begin());combined.push_back(token);
                    need(engine.forward(&token,1,error),error);need(engine.logits(0,1,logits,error),error);
                    need(logits.size()==size_t(vocab),"ordinary tail logits shape");serial.insert(serial.end(),logits.begin(),logits.end());
                    const auto f=features(engine,p+i,1,width,error);serial_features.insert(serial_features.end(),f.begin(),f.end());
                }
                run::Dump dump(run::fs::path(t.out)/path.stem(),t,combined,vocab);
                binary(dump.dir/"serial-logits.f32",serial);binary(dump.dir/"serial-features.f32",serial_features);
                std::ofstream verified_features(dump.dir/"verified-features.f32",std::ios::binary);
                Difference anchor_logits,anchor_feature,tail_logits,tail_feature,committed_logits,committed_feature;
                engine.reset();
                // Replay the actual ordinary prefill boundaries, not a fresh planner decision.
                for(const auto & chunk:prefix)need(engine.forward(prompt.data()+chunk.position,chunk.n,error),error);
                need(engine.position()==p,"replay prefix position mismatch");
                need(engine.logits(prefix.back().n-1,1,logits,error),error);
                const auto replay_anchor=anchor_features(engine,p-1,width,error);
                anchor_logits.compare(serial.data(),logits.data(),size_t(vocab),true);
                anchor_feature.compare(serial_features.data(),replay_anchor.data(),width);
                dump.row(p-1,logits.data());verified_features.write(reinterpret_cast<const char*>(replay_anchor.data()),std::streamsize(width*sizeof(float)));
                need(engine.release_prefill_workspace(error),error);
                const int pattern[]={1,2,3,4,7,8};int first=0,round=0;
                while(first<a.count){
                    const int n=std::min(pattern[round%6],a.count-first);const int commit=round%3==0?1:round%3==1?n:std::max(1,n/2);
                    const int64_t start=p+first;need(engine.position()==start,"verify start position mismatch");
                    need(engine.verify(combined.data()+p+first,n,logits,error),error);
                    need(engine.position()==start+n&&logits.size()==size_t(n)*vocab,"verify output position/shape mismatch");
                    // Both arrays are caller-owned copies before rewind invalidates graph/capture state.
                    const auto captured=features(engine,start,n,width,error);
                    for(int r=0;r<n;++r){
                        const auto expected=serial.data()+size_t(first+r+1)*vocab;
                        const auto expected_feature=serial_features.data()+size_t(first+r+1)*width;
                        tail_logits.compare(expected,logits.data()+size_t(r)*vocab,size_t(vocab),true);
                        tail_feature.compare(expected_feature,captured.data()+size_t(r)*width,width);
                        if(r<commit){
                            dump.row(int(start)+r,logits.data()+size_t(r)*vocab);
                            committed_logits.compare(expected,logits.data()+size_t(r)*vocab,size_t(vocab),true);
                            committed_feature.compare(expected_feature,captured.data()+size_t(r)*width,width);
                            verified_features.write(reinterpret_cast<const char*>(captured.data()+size_t(r)*width),std::streamsize(width*sizeof(float)));
                        }
                    }
                    need(engine.rewind(start+commit,error),error);need(engine.position()==start+commit,"rollback position mismatch");
                    calls.push_back({start,n,commit});first+=commit;++round;
                }
                need(bool(verified_features),"cannot write verified features");verified_features.close();
                // actual_chunks remains EMPTY: overlapping speculative calls cannot describe a contiguous prefill.
                dump.finish("xolotl-m31-verify-tail-gate",int(combined.size()));
                const bool exact=anchor_logits.exact()&&anchor_feature.exact()&&tail_logits.exact()&&tail_feature.exact();any_mismatch|=!exact;
                std::ofstream meta(dump.dir/"verify-metadata.json");
                meta<<"{\"format\":\"m31-verify-tail-v1\",\"scope\":\"Actual Engine.verify teacher-forced on ordinary greedy IDs, partial commits and rewind; correctness artifact, not speed or sidecar acceptance\",\"anchor_position\":"<<p-1<<",\"tail_start\":"<<p<<",\"tail_rows\":"<<a.count<<",\"head_rows_per_query\":1,\"serial_logits_file\":\"serial-logits.f32\",\"serial_features_file\":\"serial-features.f32\",\"verified_features_file\":\"verified-features.f32\",\"feature_dim\":"<<width<<",\"feature_layers\":[4,18,32,46,60],\"feature_rows_include_anchor\":true,\"bitwise_parity_pass\":"<<(exact?"true":"false")<<",\"require_bitwise\":"<<(a.require_bitwise?"true":"false")<<",\"prefix_actual_chunks\":[";
                for(size_t i=0;i<prefix.size();++i){const auto & q=prefix[i];meta<<(i?",":"")<<"{\"position\":"<<q.position<<",\"tokens\":"<<q.n<<",\"nk\":"<<q.nk<<'}';}
                meta<<"],\"nk_source\":\"padding-derived: ceil((position+tokens)/max(256,geometry.block))*max(256,geometry.block); see m31_engine.cpp kv_length(n,g), runtime tap not exposed\",\"verify_calls\":[";
                for(size_t i=0;i<calls.size();++i){const auto & q=calls[i];meta<<(i?",":"")<<"{\"position\":"<<q.position<<",\"input_tokens\":"<<q.n<<",\"committed\":"<<q.committed<<",\"rewound_to\":"<<q.position+q.committed<<'}';}
                meta<<"],\"comparisons\":{\"anchor_logits\":";anchor_logits.json(meta);meta<<",\"anchor_features\":";anchor_feature.json(meta);meta<<",\"all_verified_logits\":";tail_logits.json(meta);meta<<",\"all_verified_features\":";tail_feature.json(meta);meta<<",\"committed_logits\":";committed_logits.json(meta);meta<<",\"committed_features\":";committed_feature.json(meta);meta<<"}}\n";need(bool(meta),"cannot write verify metadata");
                std::fprintf(stderr,"verify-tail case=%s prompt=%d committed=%d calls=%zu exact=%d (correctness only)\n",path.stem().string().c_str(),p,a.count,calls.size(),int(exact));
            }
        }
        if(gpu)ggml_backend_free(gpu);return a.require_bitwise&&any_mismatch?2:0;
    }catch(const std::exception & e){std::fprintf(stderr,"port7_verify_dump: %s\n",e.what());if(gpu)ggml_backend_free(gpu);return 1;}
}
