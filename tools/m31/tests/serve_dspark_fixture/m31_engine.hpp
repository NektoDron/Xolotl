#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "json.hpp"
using ggml_backend_t = void *;
struct ggml_tensor {};
inline int fixture_draft_residents = 0;
inline ggml_backend_t ggml_backend_cuda_init(int) { return reinterpret_cast<void *>(uintptr_t(1)); }
inline void ggml_backend_free(ggml_backend_t) {}
namespace strata::m31 {
struct Geometry { int64_t embd=2,vocab=64,block=128,layers=60; };
struct Config {
    ggml_backend_t backend=nullptr;
    int threads=2,max_chunk=16,gpu_experts_min_tokens=32,draft_feature_max_rows=4096;
    int64_t context=128;
    double arena_gib=0,gpu_experts_gib=0;
    bool reuse_allocations=false,persistent_cpu_pool=true,cpu_residual=false,
         cpu_decode_cache=false,dense_decode_cache=false,profile=false;
    std::vector<int> draft_feature_layers;
};
inline std::vector<float> fixture_row(int32_t token,int64_t position) {
    std::vector<float> row(64);
    for(int i=0;i<64;++i)row[size_t(i)]=-float(i)*0.01f;
    row[size_t((token+1+position%5)%64)]=2.0f;
    return row;
}
class Engine {
    Config cfg_;Geometry g_;int64_t pos_=0;int last_n_=0;
    std::vector<int32_t> fed_;std::vector<float> rows_,features_;
    uint64_t external_=0;int forward_calls_=0,verify_calls_=0,capture_observers_=0;
    std::vector<int> forward_sizes_,verify_sizes_;
    bool feed(const int32_t * tokens,int n,std::string & error) {
        if(!tokens||n<1||pos_+n>cfg_.context){error="fixture pass size";return false;}
        rows_.clear();features_.clear();
        for(int i=0;i<n;++i){
            if(tokens[i]<0||tokens[i]>=64){error="fixture token";return false;}
            fed_.push_back(tokens[i]);++pos_;
            const auto row=fixture_row(tokens[i],pos_);rows_.insert(rows_.end(),row.begin(),row.end());
            if(!cfg_.draft_feature_layers.empty())for(int f=0;f<10;++f)features_.push_back(float(pos_)+float(f)/100);
        }
        last_n_=n;return true;
    }
public:
    bool init(const std::string &,const Config & cfg,std::string &) {cfg_=cfg;return true;}
    const Geometry & geometry()const{return g_;}
    int64_t position()const{return pos_;}
    void reset(){pos_=0;last_n_=0;fed_.clear();rows_.clear();features_.clear();forward_calls_=verify_calls_=capture_observers_=0;forward_sizes_.clear();verify_sizes_.clear();}
    bool forward(const int32_t * t,int n,std::string & e){++forward_calls_;forward_sizes_.push_back(n);return feed(t,n,e);}
    bool verify(const int32_t * t,int n,std::vector<float> & logits,std::string & e){++verify_calls_;verify_sizes_.push_back(n);if(!feed(t,n,e))return false;logits=rows_;return true;}
    bool logits(int row,int count,std::vector<float> & out,std::string & error){if(row<0||count<1||row+count>last_n_){error="fixture logits";return false;}out.assign(rows_.begin()+row*64,rows_.begin()+(row+count)*64);return true;}
    bool rewind(int64_t p,std::string & error){if(p<0||p>pos_){error="fixture rewind";return false;}pos_=p;fed_.resize(size_t(p));last_n_=0;rows_.clear();features_.clear();return true;}
    int prefill_chunk(int requested,std::string & error)const{if(external_ || fixture_draft_residents){error="draft resident during cold prefill";return 0;}return int(std::min<int64_t>({requested,cfg_.max_chunk,cfg_.context-pos_}));}
    int prefill_chunk_at(int64_t p,int requested,std::string &)const{return int(std::min<int64_t>({requested,cfg_.max_chunk,cfg_.context-p}));}
    bool set_draft_feature_layers(const std::vector<int> & layers,int maximum,std::string & error){if(pos_||last_n_){error="fixture capture not at reset";return false;}cfg_.draft_feature_layers=layers;cfg_.draft_feature_max_rows=maximum;return true;}
    bool set_draft_feature_window(int64_t minimum,std::string & error){if(pos_||last_n_||minimum<0){error="fixture window not at reset";return false;}return true;}
    bool set_draft_feature_view_window(int64_t begin,int64_t end,std::string & error){if(pos_||last_n_||begin<0||end<=begin||end-begin>4096){error="fixture invalid view bounds";return false;}return true;}
    bool draft_features(int64_t & first,int & count,std::vector<float> & out,std::string & error){if(features_.empty()||!last_n_){error="fixture no features";return false;}++capture_observers_;first=pos_-last_n_;count=last_n_;out=features_;return true;}
    bool set_external_resident_bytes(uint64_t n,std::string &){external_=n;return true;}
    bool release_prefill_workspace(std::string &){last_n_=0;rows_.clear();features_.clear();return true;}
    ggml_tensor * output_weight()const{static ggml_tensor tensor;return &tensor;}
    nlohmann::json fixture_stats()const{return {{"forward_calls",forward_calls_},{"verify_calls",verify_calls_},{"feature_reads",capture_observers_},{"capture_enabled",!cfg_.draft_feature_layers.empty()},{"forward_sizes",forward_sizes_},{"verify_sizes",verify_sizes_},{"external_bytes",external_}};}
};
} // namespace strata::m31
