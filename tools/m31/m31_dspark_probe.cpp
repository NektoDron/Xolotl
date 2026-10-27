// Draft-only probe over an exported target-feature fixture; no target FFN runs.
#include "m31_dspark.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using namespace strata::m31;
int main(int argc,char ** argv) {
    ggml_backend_t backend=nullptr;ggml_context * context=nullptr;ggml_backend_buffer_t buffer=nullptr;
    try {
        std::map<std::string,std::string> a;for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing argument value");a[argv[i]]=argv[i+1];}
        const bool cuda=a["--backend"]!="cpu";
        if(cuda) {
            if(!std::getenv("M31_CARD_SESSION")||std::string(std::getenv("M31_CARD_SESSION"))!="1")throw std::runtime_error("CUDA requires aictl/flock runner");
#ifdef M31_HAVE_CUDA
            backend=ggml_backend_cuda_init(0);
#else
            throw std::runtime_error("built without CUDA");
#endif
        }else backend=ggml_backend_cpu_init();
        if(!backend)throw std::runtime_error("cannot create backend");
        auto target=strata::GgufModel::open(a.at("--model"));size_t sh=0;auto ti=target.find("output.weight",&sh);
        if(!ti||ti->shape.size()!=2)throw std::runtime_error("target head missing");
        context=ggml_init({1024*1024,nullptr,true});auto head=ggml_new_tensor_2d(context,ggml_type(ti->type),int64_t(ti->shape[0]),int64_t(ti->shape[1]));
        buffer=ggml_backend_alloc_ctx_tensors_from_buft(context,ggml_backend_get_default_buffer_type(backend));
        if(!buffer)throw std::runtime_error("cannot allocate target head");ggml_backend_tensor_set(head,target.shard(sh).tensor_data(*ti),0,ggml_nbytes(head));
        {
            DSpark draft;DSparkConfig cfg;cfg.backend=backend;cfg.shared_head=head;cfg.target=a.at("--model");cfg.sidecar=a.at("--sidecar");
            std::string error;if(!draft.init(cfg,error))throw std::runtime_error(error);
            const int rows=std::stoi(a.at("--rows")),depth=std::stoi(a.at("--depth")),anchor=std::stoi(a.at("--anchor"));const int64_t first=std::stoll(a.at("--first"));
            if(rows<1||rows>4096)throw std::runtime_error("invalid rows");std::vector<float> features(size_t(rows)*draft.feature_dim());
            std::ifstream input(a.at("--features"),std::ios::binary);input.read(reinterpret_cast<char*>(features.data()),std::streamsize(features.size()*sizeof(float)));
            if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("feature file length mismatch");
            if(!draft.append(features.data(),first,rows,error))throw std::runtime_error(error);const double append_ms=draft.last_ms();
            for(int trial=0;trial<3;++trial){std::vector<int32_t> ids;std::vector<float> confidence;if(!draft.draft(anchor,depth,0,ids,confidence,error))throw std::runtime_error(error);
                std::cout<<"{\"trial\":"<<trial<<",\"weight_bytes\":"<<draft.weight_bytes()<<",\"append_ms\":"<<append_ms<<",\"draft_ms\":"<<draft.last_ms()<<",\"ids\":[";
                for(size_t i=0;i<ids.size();++i)std::cout<<(i?",":"")<<ids[i];std::cout<<"],\"confidence\":[";for(size_t i=0;i<confidence.size();++i)std::cout<<(i?",":"")<<confidence[i];std::cout<<"]}\n";std::cout.flush();}
        }
        ggml_backend_buffer_free(buffer);ggml_free(context);ggml_backend_free(backend);return 0;
    }catch(const std::exception & e){std::cerr<<"m31_dspark_probe: "<<e.what()<<"\n";if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);if(backend)ggml_backend_free(backend);return 1;}
}
