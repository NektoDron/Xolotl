#pragma once
// Greedy HTTP-session controller. The caller retains the ordinary serialized
// sample -> emit -> ACK -> advance order. No sampler is called ahead of emission.
#include "m31_engine.hpp"
#include "draft_feature_view.hpp"
#include "m31_dspark.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <stdexcept>

namespace strata::m31 {
struct DSparkSessionConfig {
    DSparkConfig draft;
    int depth=3;
    float p_min=0;
    int64_t context=40000;
};
struct DSparkSessionStats {
    uint64_t rounds=0,proposed=0,accepted=0,verified_rows=0,cached_rows=0;
    int seed_rows=0;
    double prepare_ms=0,draft_ms=0,verify_ms=0,append_ms=0;
};
// Template parameters permit a meaningful controller proof with a deterministic
// target/draft fixture; the production alias below uses the actual native engine.
template<class Target,class Drafter> class BasicDSparkGreedySession {
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
    struct Seed {
        int64_t first=0;
        int rows=0,skip=0;
        std::vector<float> data;
    };
    DSparkSessionConfig cfg_;
    std::unique_ptr<Drafter> draft_;
    std::deque<Seed> seed_;
    int seed_rows_=0;
    bool feature_view_=false;
    int64_t view_first_=0;
    int view_rows_=0;
    size_t width_=0;
    int vocab_=0;
    bool prepared_=false;
    int64_t logical_fed_=0,buffer_first_=0;
    size_t consumed_=0;
    std::vector<int32_t> inputs_;
    std::vector<float> rows_,features_;
    DSparkSessionStats stats_;
    static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
    void clear_buffer(){inputs_.clear();rows_.clear();features_.clear();consumed_=0;}
    bool flush_features(std::string & error) {
        if(consumed_){
            require(draft_&&draft_->position()==buffer_first_,"DSpark session draft/target feature positions differ");
            const auto started=Clock::now();
            if(!draft_->append(features_.data(),buffer_first_,int(consumed_),error))return false;
            stats_.append_ms+=elapsed(started);
        }
        clear_buffer();return true;
    }
public:
    explicit BasicDSparkGreedySession(const DSparkSessionConfig & cfg):cfg_(cfg){
        require(cfg.depth>=1&&cfg.depth<=7&&std::isfinite(cfg.p_min)&&cfg.p_min>=0&&cfg.p_min<=1&&cfg.context>0,"invalid DSpark session configuration");
        const char * view=std::getenv("M31_DSPARK_CAPTURE_VIEW");
        feature_view_=view&&std::string(view)=="1";
    }
    // Call before a cold eligible request, with a valid Engine. Worker then resets
    // Engine and enables capture through its reset-only feature-layer setter.
    bool begin_prefill(Target & target,std::string & error) {
        try {
            draft_.reset();seed_.clear();seed_rows_=0;width_=0;vocab_=0;
            prepared_=false;logical_fed_=0;view_first_=0;view_rows_=0;clear_buffer();stats_={};
            return target.set_external_resident_bytes(0,error);
        }catch(const std::exception & e){error=e.what();return false;}
    }
    bool observe_prefill(Target & target,std::string & error) {
        try {
            if (feature_view_) {
                if constexpr (requires(Target & t,DraftFeatureView & v,std::string & e){t.draft_feature_window_view(v,e);}) {
                    DraftFeatureView view;
                    if (!target.draft_feature_window_view(view,error)) return false;
                    require(view.valid()&&view.rows>=0&&view.rows<=4096&&view.width==size_t(target.geometry().embd)*5&&
                        view.first_position+view.rows==target.position(),"invalid DSpark accumulated feature window");
                    // Metadata only. Preserve the old per-chunk append sizes:
                    // changing seed GEMM batch geometry could change draft math.
                    // No lease survives this method/next forward.
                    if (view.rows) {
                        require(view_rows_==0||(view.first_position==view_first_&&view.rows>=view_rows_),
                            "DSpark accumulated feature window lost its prefix");
                        if (view.rows>view_rows_) seed_.push_back({view.first_position+view_rows_,view.rows-view_rows_,0,{}});
                        seed_rows_=view.rows;
                    }
                    view_first_=view.first_position;view_rows_=view.rows;
                    width_=view.width;vocab_=int(target.geometry().vocab);
                    return true;
                } else require(false,"target does not support feature view capture");
            }
            int64_t first=0;int count=0;std::vector<float> values;
            if(!target.draft_features(first,count,values,error))return false;
            if(count==0){
                require(values.empty()&&first==target.position(),"invalid empty DSpark prefill feature slice");
                return true;
            }
            const size_t width=size_t(target.geometry().embd)*5;
            require(count>0&&count<=4096&&values.size()==size_t(count)*width&&first+count==target.position(),"invalid DSpark prefill feature geometry");
            if(!seed_.empty()&&seed_.back().first+seed_.back().rows!=first){seed_.clear();seed_rows_=0;}
            width_=width;vocab_=int(target.geometry().vocab);
            seed_.push_back({first,count,0,std::move(values)});seed_rows_+=count;
            while(seed_rows_>4096){
                auto & front=seed_.front();const int drop=std::min(seed_rows_-4096,front.rows-front.skip);
                front.skip+=drop;seed_rows_-=drop;if(front.skip==front.rows)seed_.pop_front();
            }
            return true;
        }catch(const std::exception & e){error=e.what();return false;}
    }
    // Lazy: worker invokes it through the first advance AFTER token0 and its ACK.
    // max_tokens1, immediateEOS and cancellation before advance never load draft.
    bool prepare_after_first_sample(Target & target,std::string & error) {
        try {
            DraftFeatureView lease;
            if (feature_view_) {
                if constexpr (requires(Target & t,DraftFeatureView & v,std::string & e){t.draft_feature_window_view(v,e);}) {
                    if (!target.draft_feature_window_view(lease,error)) return false;
                    require(!prepared_&&lease.valid()&&lease.rows>0&&lease.rows==view_rows_&&lease.first_position==view_first_&&
                        lease.first_position+lease.rows==target.position()&&lease.width==width_,"DSpark final feature window changed");
                } else require(false,"target does not support feature view capture");
            } else {
                require(!prepared_&&!seed_.empty()&&seed_rows_>0,"DSpark session has no unconsumed prefill features");
                require(seed_.back().first+seed_.back().rows==target.position(),"DSpark seed does not end at current target position");
            }
            const auto started=Clock::now();logical_fed_=target.position();
            if(!target.release_prefill_workspace(error))return false;
            draft_=std::make_unique<Drafter>();auto dc=cfg_.draft;dc.shared_head=target.output_weight();
            if(!draft_->init(dc,error))return false;
            require(size_t(draft_->feature_dim())==width_,"DSpark feature width differs from target capture");
            if (feature_view_) {
                // release_prefill_workspace keeps this plain-host owner/epoch.
                // append completes its H2D/compute before the lease is dropped.
                require(lease.valid(),"DSpark feature lease invalidated during prepare");
                require(!seed_.empty()&&seed_rows_==lease.rows,"missing DSpark view segment recipe");
                for (const auto & segment:seed_) {
                    const auto offset=size_t(segment.first-lease.first_position)*width_;
                    if (!draft_->append(lease.data()+offset,segment.first,segment.rows,error)) return false;
                }
                stats_.seed_rows=lease.rows;lease.clear();
            } else {
                for(const auto & segment:seed_)if(!draft_->append(segment.data.data()+size_t(segment.skip)*width_,segment.first+segment.skip,segment.rows-segment.skip,error))return false;
                stats_.seed_rows=seed_rows_;
            }
            require(draft_->position()==logical_fed_,"DSpark seed position mismatch");
            stats_.prepare_ms=elapsed(started);
            seed_.clear();seed_rows_=0;prepared_=true;return true;
        }catch(const std::exception & e){error=e.what();return false;}
    }
    // previous_token was sampled and emitted by the worker, then ACKed. Returns
    // precisely the next raw logits row; caller runs its existing choose() once.
    // remaining_future_samples counts tokens still to sample after previous_token.
    bool advance(Target & target,int32_t previous_token,int remaining_future_samples,
                 std::vector<float> & logits,std::string & error) {
        try {
            require(remaining_future_samples>0,"DSpark advance requires a future sample");
            if(!prepared_&&!prepare_after_first_sample(target,error))return false;
            require(previous_token>=0&&previous_token<vocab_,"DSpark advance token outside vocabulary");
            const bool cached=consumed_<inputs_.size()&&inputs_[consumed_]==previous_token;
            if(!cached){
                if(!flush_features(error))return false;
                require(target.position()>=logical_fed_,"DSpark target rewound past logical fed prefix");
                if(target.position()!=logical_fed_&&!target.rewind(logical_fed_,error))return false;
                require(draft_->position()==logical_fed_,"DSpark draft ring lags accepted target prefix");
                const int64_t available=cfg_.context-logical_fed_;
                require(available>0,"DSpark verification exceeds context");
                const int depth=int(std::min<int64_t>({cfg_.depth,remaining_future_samples-1,available-1}));
                std::vector<int32_t> proposed;std::vector<float> confidence;
                if(depth){
                    if(!draft_->draft(previous_token,depth,cfg_.p_min,proposed,confidence,error))return false;
                    stats_.draft_ms+=draft_->last_ms();
                }
                inputs_.push_back(previous_token);inputs_.insert(inputs_.end(),proposed.begin(),proposed.end());buffer_first_=logical_fed_;
                const auto started=Clock::now();
                if(!target.verify(inputs_.data(),int(inputs_.size()),rows_,error))return false;
                stats_.verify_ms+=elapsed(started);++stats_.rounds;stats_.proposed+=proposed.size();stats_.verified_rows+=inputs_.size();
                int64_t first=0;int count=0;if(!target.draft_features(first,count,features_,error))return false;
                require(first==buffer_first_&&count==int(inputs_.size())&&features_.size()==inputs_.size()*width_&&rows_.size()==inputs_.size()*size_t(vocab_),"DSpark verified row/feature geometry differs");
            }else{++stats_.cached_rows;if(consumed_>0)++stats_.accepted;}
            require(consumed_<inputs_.size()&&inputs_[consumed_]==previous_token,"DSpark row input mismatch");
            const size_t first=consumed_*size_t(vocab_);
            logits.assign(rows_.begin()+first,rows_.begin()+first+size_t(vocab_));
            ++consumed_;++logical_fed_;return true;
        }catch(const std::exception & e){error=e.what();return false;}
    }
    // Stop/EOS/cancel/max_tokens: retain only the worker's actual fed prefix,
    // discard speculative tail and evict draft before another cold prefill.
    bool finish(Target & target,std::string & error) {
        try {
            if(prepared_){
                require(target.position()>=logical_fed_,"DSpark target lost its confirmed prefix");
                if(target.position()!=logical_fed_&&!target.rewind(logical_fed_,error))return false;
                if(!target.release_prefill_workspace(error))return false;
            }
            draft_.reset();prepared_=false;seed_.clear();seed_rows_=0;view_first_=0;view_rows_=0;clear_buffer();
            return target.set_external_resident_bytes(0,error);
        }catch(const std::exception & e){error=e.what();return false;}
    }
    const DSparkSessionStats & stats()const{return stats_;}
    bool prepared()const{return prepared_;}
    int64_t logical_fed_position()const{return logical_fed_;}
};
using DSparkGreedySession=BasicDSparkGreedySession<Engine,DSpark>;
} // namespace strata::m31
