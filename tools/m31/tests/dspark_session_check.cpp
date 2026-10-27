// Independent causal target fixture tests the HTTP controller's raw-row queue,
// rollback, lazy loading and stop/cancel/max semantics, without model/GPU math.
#include "m31_dspark_session.hpp"
#include <iostream>
#include <set>

using namespace strata::m31;
namespace {
constexpr int V=17;
int predict(const std::vector<int32_t> & tokens,int position){
    int value=position*3;
    for(int i=std::max(0,position-3);i<position;++i)value+=tokens[size_t(i)]*(i%7+2);
    return value%V;
}
std::vector<float> raw(const std::vector<int32_t> & tokens,int position){
    std::vector<float> result(V,-8);const int best=predict(tokens,position);
    for(int i=0;i<V;++i)result[size_t(i)]=float(-((i-best+V)%V));return result;
}
int choose(const std::vector<float> & row,const std::set<int> & excluded){
    int best=-1;for(int i=0;i<V;++i)if(!excluded.count(i)&&(best<0||row[size_t(i)]>row[size_t(best)]))best=i;return best;
}
struct FakeTarget {
    struct Geometry {int embd=1,vocab=V;} g;
    std::vector<int32_t> kv;
    std::vector<float> last,features;
    int64_t pos=0,first=0;
    int captured=0;
    uint64_t external=99;
    bool view_mode=false,view_ready=false;
    int64_t view_begin=0,view_end=0;
    std::shared_ptr<uint64_t> epoch=std::make_shared<uint64_t>(0);
    std::shared_ptr<float> window;
    void configure_view(int64_t begin,int64_t end) {
        view_mode=true;view_begin=begin;view_end=end;
        window=std::shared_ptr<float>(new float[4096*5](),std::default_delete<float[]>());
    }
    bool draft_feature_window_view(DraftFeatureView & out,std::string & error) {
        if(!view_mode||!view_ready){error="fake no window view";return false;}
        const int rows=int(std::max<int64_t>(0,pos-view_begin));
        out=DraftFeatureView(window,epoch,rows?view_begin:pos,rows,5);return true;
    }
    const Geometry & geometry()const{return g;}
    void reset(){++*epoch;pos=0;captured=0;features.clear();view_mode=false;view_ready=false;}
    int64_t position()const{return pos;}
    ggml_tensor * output_weight()const{return reinterpret_cast<ggml_tensor *>(uintptr_t(1));}
    bool set_external_resident_bytes(uint64_t bytes,std::string&){external=bytes;return true;}
    bool release_prefill_workspace(std::string&){captured=0;features.clear();view_ready=false;return true;}
    bool rewind(int64_t next,std::string & error){if(next<0||next>pos){error="fakeinvalidrewind";return false;}++*epoch;pos=next;captured=0;features.clear();view_ready=false;return true;}
    bool verify(const int32_t * input,int n,std::vector<float> & rows,std::string &){
        ++*epoch;view_ready=false;rows.clear();features.clear();first=pos;captured=n;
        for(int i=0;i<n;++i){if(kv.size()<=size_t(pos))kv.resize(size_t(pos)+1);kv[size_t(pos)]=input[i];++pos;
            last=raw(kv,int(pos));rows.insert(rows.end(),last.begin(),last.end());
            features.insert(features.end(),{float(input[i]),float(pos-1),float(input[i]),0,0});}
        return true;
    }
    bool forward(const int32_t * input,int n,std::string & error){
        if(window&&window.use_count()>1){error="fake live lease before forward";return false;}
        std::vector<float> rows;if(!verify(input,n,rows,error))return false;
        if(n>4096){const int skip=n-4096;features.erase(features.begin(),features.begin()+skip*5);first+=skip;captured=4096;}
        if(view_mode&&pos<=view_end){
            const auto begin=std::max(first,view_begin);
            if(pos>begin)std::copy(features.begin()+size_t(begin-first)*5,features.end(),window.get()+size_t(begin-view_begin)*5);
            view_ready=true;
        }
        return true;
    }
    bool draft_features(int64_t & begin,int & rows,std::vector<float> & data,std::string &){begin=first;rows=captured;data=features;return true;}
};
struct FakeDraft {
    inline static int mode=0,loads=0,destroys=0;
    inline static std::vector<std::pair<int64_t,int>> append_recipe;
    std::vector<int32_t> accepted;
    int64_t pos=0;int rounds=0;
    ~FakeDraft(){++destroys;}
    bool init(const DSparkConfig &,std::string&){++loads;append_recipe.clear();return true;}
    int feature_dim()const{return 5;}
    int64_t position()const{return pos;}
    double last_ms()const{return 0;}
    bool append(const float * features,int64_t first,int n,std::string & error){
        append_recipe.push_back({first,n});
        if(first<pos||n<=0){error="fakeoutoforderappend";return false;}
        if(accepted.size()<size_t(first+n))accepted.resize(size_t(first+n));
        for(int i=0;i<n;++i){if(features[i*5+1]!=float(first+i)){error="fakefeaturepositionmismatch";return false;}accepted[size_t(first+i)]=int(features[i*5]);}
        pos=first+n;return true;
    }
    bool draft(int32_t anchor,int depth,float p_min,std::vector<int32_t>&ids,std::vector<float>&confidence,std::string&){
        ids.clear();confidence.clear();auto history=accepted;if(history.size()<size_t(pos+depth+1))history.resize(size_t(pos+depth+1));history[size_t(pos)]=anchor;
        const int correct=mode==0?depth:mode==1?0:rounds%(depth+1);
        for(int i=0;i<depth;++i){const float conf=i<2?.8f:.3f;if(conf<p_min)break;int token=predict(history,int(pos+i+1));if(i>=correct)token=(token+1)%V;ids.push_back(token);confidence.push_back(conf);history[size_t(pos+i+1)]=token;}
        ++rounds;return true;
    }
};
void need(bool ok,const std::string & why){if(!ok)throw std::runtime_error(why);}
std::vector<int32_t> baseline(const std::vector<int32_t>&prompt,int maximum,int cancel,const std::set<int>&excluded){
    auto history=prompt;std::vector<int32_t> output;
    for(int i=0;i<maximum;++i){const auto row=raw(history,int(history.size()));const int token=choose(row,excluded);output.push_back(token);if(i==cancel)break;if(i+1<maximum)history.push_back(token);}
    return output;
}
}
int main(){
    try{
        size_t cases=0,rows=0;std::string error;
        std::vector<std::pair<int64_t,int>> legacy_recipe;
        for(int length:{5,255,4097})for(int depth:{1,3,7})for(int mode:{0,1,2})for(float p_min:{0.f,.5f,.9f})for(int maximum:{0,1,2,17,40})for(int cancelled:{-1,1,7})for(bool suppress:{false,true})for(bool skipped_window:{false,true})for(bool view_mode:{false,true}){
            std::vector<int32_t> prompt(size_t(length),0);for(int i=0;i<length;++i)prompt[size_t(i)]=(i*11+3)%V;
            const std::set<int> excluded=suppress?std::set<int>{1,2,3,5}:std::set<int>{};
            const auto expected=baseline(prompt,maximum,cancelled,excluded);
            setenv("M31_DSPARK_CAPTURE_VIEW",view_mode?"1":"0",1);
            DSparkSessionConfig cfg;cfg.depth=depth;cfg.p_min=p_min;cfg.context=10000;
            BasicDSparkGreedySession<FakeTarget,FakeDraft> session(cfg);FakeTarget target;FakeDraft::mode=mode;const int loads=FakeDraft::loads;
            need(session.begin_prefill(target,error),error);target.reset();
            // Two chunks exercise retained-segment trimming above4096.
            const int first=std::max(1,length/2);
            if(view_mode)target.configure_view(skipped_window?first:std::max(0,length-4096),length);
            need(target.forward(prompt.data(),first,error),error);
            // A capture window may skip complete early prefill chunks. The
            // getter's valid empty slice must neither fail nor seed a gap.
            if(skipped_window){target.captured=0;target.features.clear();target.first=target.position();}
            need(session.observe_prefill(target,error),error);
            if(first<length){need(target.forward(prompt.data()+first,length-first,error),error);need(session.observe_prefill(target,error),error);}
            auto logits=target.last;std::vector<int32_t> actual;
            for(int i=0;i<maximum;++i){const int token=choose(logits,excluded);actual.push_back(token);if(i==cancelled)break;if(i+1<maximum)need(session.advance(target,token,maximum-i-1,logits,error),error);}
            need(session.finish(target,error),error);
            need(actual==expected,"controller output differs from ordinary causal target");
            const int64_t fed=int64_t(prompt.size())+std::max<int64_t>(0,int64_t(actual.size())-1);
            need(target.position()==fed,"finish retained un-emitted or speculative KV rows");need(target.external==0,"draft budget leaked after finish");
            if(actual.size()<=1)need(FakeDraft::loads==loads,"max_tokens0/1 loaded the drafter");
            else {
                need(FakeDraft::loads==loads+1,"session loaded the drafter more than once");
                if(!view_mode)legacy_recipe=FakeDraft::append_recipe;
                else need(FakeDraft::append_recipe==legacy_recipe,"view changed draft append/GEMM segment geometry");
            }
            // Branch after finish. Stale future bytes must not enter the result.
            auto prefix=prompt;prefix.insert(prefix.end(),actual.begin(),actual.empty()?actual.end():actual.end()-1);
            const int32_t branch=9;prefix.push_back(branch);need(target.forward(&branch,1,error),error);need(target.last==raw(prefix,int(prefix.size())),"rejected speculative tail changed next branch");
            ++cases;rows+=actual.size();
        }
        std::cout<<"{\"pass\":true,\"cases\":"<<cases<<",\"generated_rows\":"<<rows<<",\"scope\":\"Controller fixture: feature-view vs legacy seed,  causal raw-row queue, all/none/partial acceptance, suppression, p_min, cancel/max, last4096seed, skipped-window empty capture slices, stale-tail branch and lazy single draft load; no GPU/model math\"}\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
