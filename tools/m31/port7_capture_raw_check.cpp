// Exact residual capture comparison. Run both flag0/flag1 arms under aictl.
#include "m31_engine.hpp"
#include "json.hpp"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strata::m31;
using json=nlohmann::json;
int main(int argc,char ** argv){ggml_backend_t gpu=nullptr;try{
    if(argc!=7)throw std::runtime_error("usage: capture_check MODEL INPUT.i32 OUT_DIR MIN_POSITION CHUNK REPEAT");
    if(!std::getenv("M31_CARD_SESSION")||std::string(std::getenv("M31_CARD_SESSION"))!="1")throw std::runtime_error("aictl/flock session required");
    const int64_t minimum=std::stoll(argv[4]);const int chunk=std::stoi(argv[5]),repeat=std::stoi(argv[6]);
    std::ifstream input(argv[2],std::ios::binary|std::ios::ate);const auto bytes=input?std::streamoff(input.tellg()):-1;
    if(bytes<=0||bytes%4||bytes/4>32768||minimum<0||minimum>bytes/4||chunk<1||chunk>32768||repeat<1||repeat>3)throw std::runtime_error("invalid probe geometry");
    std::vector<int32_t> ids(size_t(bytes)/4);input.seekg(0);input.read(reinterpret_cast<char *>(ids.data()),bytes);if(!input)throw std::runtime_error("short ids");
    const auto out=std::filesystem::path(argv[3]);std::filesystem::create_directories(out);
    gpu=ggml_backend_cuda_init(0);if(!gpu)throw std::runtime_error("GPU init");
    const char * flag=std::getenv("M31_DSPARK_CAPTURE_REUSE");const bool reuse=flag&&std::string(flag)=="1";
    const char * view_flag=std::getenv("M31_DSPARK_CAPTURE_VIEW");const bool view_mode=view_flag&&std::string(view_flag)=="1";
    json runs=json::array();int64_t feature_width=0,vocab=0;
    {
        Config cfg;cfg.backend=gpu;cfg.context=40000;cfg.max_chunk=32768;cfg.threads=80;cfg.arena_gib=170;cfg.gpu_experts_gib=7;cfg.gpu_experts_min_tokens=32;
        cfg.reuse_allocations=true;cfg.persistent_cpu_pool=true;cfg.cpu_residual=true;cfg.cpu_decode_cache=true;cfg.dense_decode_cache=true;cfg.profile=true;
        Engine engine;std::string error;if(!engine.init(argv[1],cfg,error))throw std::runtime_error(error);
        feature_width=engine.geometry().embd*5;vocab=engine.geometry().vocab;
        for(int run=0;run<repeat;++run){
            engine.reset();if(!engine.set_draft_feature_layers({4,18,32,46,60},4096,error))throw std::runtime_error(error);
            if(view_mode){if(!engine.set_draft_feature_view_window(minimum,int64_t(ids.size()),error))throw std::runtime_error(error);}
            else if(reuse&&!engine.set_draft_feature_window(minimum,error))throw std::runtime_error(error);
            std::ofstream features(out/("features-"+std::to_string(run)+".f32"),std::ios::binary);
            json chunks=json::array();int64_t saved_rows=0;int last=0;
            const auto started=std::chrono::steady_clock::now();
            for(int64_t position=0;position<int64_t(ids.size());){
                const int requested=int(std::min<int64_t>(chunk,int64_t(ids.size())-position));last=engine.prefill_chunk(requested,error);if(last<=0)throw std::runtime_error(error);
                if(!engine.forward(ids.data()+position,last,error))throw std::runtime_error(error);
                int64_t first=0;int rows=0;std::vector<float> values;if(!engine.draft_features(first,rows,values,error))throw std::runtime_error(error);
                const int64_t width=engine.geometry().embd*5;
                if(first<position||first+rows!=position+last||int64_t(values.size())!=rows*width)throw std::runtime_error("invalid returned window");
                if(reuse&&first!=std::max(position+last-int64_t(4096),std::min(position+last,std::max(position,minimum))))throw std::runtime_error("capture minimum contract mismatch");
                const int64_t skip=std::min<int64_t>(rows,std::max<int64_t>(0,minimum-first));
                for(size_t i=size_t(skip*width);i<values.size();++i)if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite residual");
                if (rows>skip) features.write(reinterpret_cast<const char *>(values.data()+skip*width),std::streamsize((rows-skip)*width*sizeof(float)));
                saved_rows+=rows-skip;chunks.push_back({{"position",position},{"n",last},{"nk",((position+last+255)/256)*256},{"first",first},{"rows",rows},{"saved_rows",rows-skip}});position+=last;
            }
            std::vector<float> logits;if(!engine.logits(last-1,1,logits,error))throw std::runtime_error(error);
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            std::ofstream head(out/("head-"+std::to_string(run)+".f32"),std::ios::binary);head.write(reinterpret_cast<const char *>(logits.data()),std::streamsize(logits.size()*sizeof(float)));
            if(!head||!features)throw std::runtime_error("write failed");
            json lifetime=nullptr;
            if(view_mode){
                DraftFeatureView lease;
                if(!engine.draft_feature_window_view(lease,error)||!lease.valid()||lease.first_position!=minimum||lease.rows!=saved_rows||lease.width!=size_t(feature_width))
                    throw std::runtime_error("accumulated feature window does not cover complete requested prefix");
                auto write_view=[&](const std::string & name){std::ofstream file(out/(name+"-"+std::to_string(run)+".f32"),std::ios::binary);file.write(reinterpret_cast<const char *>(lease.data()),std::streamsize(lease.rows*lease.width*sizeof(float)));if(!file)throw std::runtime_error("window write failed");};
                write_view("window");
                if(!engine.release_prefill_workspace(error)||!lease.valid())throw std::runtime_error("workspace release invalidated plain-host lease");
                write_view("window-after-release");
                DraftFeatureView rejected;if(engine.draft_feature_window_view(rejected,error))throw std::runtime_error("released getter returned stale feature window");
                const int32_t token=0;const auto position=engine.position();
                if(engine.forward(&token,1,error)||engine.position()!=position||!lease.valid())throw std::runtime_error("live lease did not reject next forward preflight");
                if(!engine.rewind(position,error)||lease.valid())throw std::runtime_error("rewind did not invalidate feature lease epoch");
                bool invalidated=false;try{lease.data();}catch(const std::exception &){invalidated=true;}
                if(!invalidated)throw std::runtime_error("stale lease data exposed");lease.clear();
                lifetime={{"release_kept_bytes",true},{"released_getter_rejected",true},{"live_lease_forward_rejected",true},{"rewind_invalidated",true}};
            }
            runs.push_back({{"chunks",chunks},{"saved_rows",saved_rows},{"position",engine.position()},{"seconds_with_feature_export_and_IO",elapsed},{"view_lifetime",lifetime}});
        }
    }
    ggml_backend_free(gpu);gpu=nullptr;
    json result={{"pass",true},{"scope","raw F32 feature/head artifact producer; cross-arm byte comparison required, not performance"},{"feature_width",feature_width},{"vocab",vocab},{"view_mode",view_mode},{"reuse",reuse},{"minimum",minimum},{"prompt_tokens",ids.size()},{"runs",runs}};
    std::ofstream receipt(out/"result.json");receipt<<result.dump(2)<<'\n';std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception & error){std::cerr<<error.what()<<'\n';if(gpu)ggml_backend_free(gpu);return 1;}}
