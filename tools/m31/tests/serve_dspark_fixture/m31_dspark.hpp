#pragma once
#include "m31_engine.hpp"
namespace strata::m31 {
struct DSparkConfig {ggml_backend_t backend=nullptr;std::string sidecar,target;ggml_tensor * shared_head=nullptr;int max_append=1024;};
class DSpark {
    int64_t pos_=0;
    bool allocated_=false;
public:
    static inline int inits=0;
    ~DSpark(){if(allocated_)--fixture_draft_residents;}
    bool init(const DSparkConfig &,std::string & error){++inits;allocated_=true;++fixture_draft_residents;if(std::getenv("DS_FIXTURE_DRAFT_FAIL_ONCE")&&inits==1){error="injected draft init error";return false;}return true;}
    bool append(const float *,int64_t first,int count,std::string & error){if(first<pos_||count<=0){error="fixture draft feature order";return false;}pos_=first+count;return true;}
    bool draft(int32_t anchor,int depth,float,std::vector<int32_t> & ids,std::vector<float> & confidence,std::string &){
        ids.clear();confidence.clear();int32_t token=anchor;
        const std::string mode=std::getenv("DS_FIXTURE_DRAFT_MODE")?std::getenv("DS_FIXTURE_DRAFT_MODE"):"all";
        for(int i=0;i<depth;++i){
            auto row=fixture_row(token,pos_+i+1);token=int32_t(std::max_element(row.begin(),row.end())-row.begin());
            if((mode=="none"&&i==0)||(mode=="middle"&&i==2))token=(token+13)%64;
            ids.push_back(token);confidence.push_back(.9f);
        }
        return true;
    }
    int64_t position()const{return pos_;}
    int feature_dim()const{return 10;}
    int block_size()const{return 32;}
    uint64_t weight_bytes()const{return 0;}
    uint64_t resident_bytes()const{return 0;}
    double last_ms()const{return 0;}
    const std::vector<int> & target_layers()const{static const std::vector<int> layers{4,18,32,46,60};return layers;}
    void reset(){pos_=0;}
};
} // namespace strata::m31
