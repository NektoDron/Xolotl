// Dedicated acceptance probe: real GPU/Engine rows, no tokenizer or sampler.
// Invoke only through the aictl/flock runner with the accepted native ENV.
#include "m31_engine.hpp"
#include "server_prefix_cache.hpp"
#include "json.hpp"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strata::m31;
using json = nlohmann::json;
int main(int argc, char ** argv) {
    ggml_backend_t gpu = nullptr;
    try {
        if (argc != 4) throw std::runtime_error("usage: raw_check MODEL INPUT.i32 OUT_DIR");
        if (!std::getenv("M31_CARD_SESSION") || std::string(std::getenv("M31_CARD_SESSION")) != "1")
            throw std::runtime_error("aictl/flock session required");
        std::ifstream input(argv[2],std::ios::binary | std::ios::ate);
        const std::streamoff input_bytes = input ? std::streamoff(input.tellg()) : -1;
        if (input_bytes <= 0 || input_bytes % 4 != 0 || input_bytes / 4 > 32768) throw std::runtime_error("invalid input ids");
        std::vector<int32_t> ids(size_t(input_bytes)/4); input.seekg(0);
        input.read(reinterpret_cast<char *>(ids.data()),std::streamsize(ids.size()*4));
        if (!input) throw std::runtime_error("short input ids read");
        const std::filesystem::path out(argv[3]);std::filesystem::create_directories(out);
        gpu = ggml_backend_cuda_init(0);if(!gpu)throw std::runtime_error("GPU0 init");
        json result;
        {
            Config cfg;cfg.backend=gpu;cfg.context=40000;cfg.max_chunk=32768;cfg.threads=80;
            cfg.arena_gib=170;cfg.gpu_experts_gib=7;cfg.gpu_experts_min_tokens=32;
            cfg.reuse_allocations=true;cfg.persistent_cpu_pool=true;cfg.cpu_residual=true;
            cfg.cpu_decode_cache=true;cfg.dense_decode_cache=true;cfg.profile=true;
            Engine engine;std::string error;
            if(!engine.init(argv[1],cfg,error))throw std::runtime_error(error);
            auto append=[&](const int32_t *data,int n){if(!engine.forward(data,n,error))throw std::runtime_error(error);};
            auto head=[&](int row){std::vector<float> values;if(!engine.logits(row,1,values,error))throw std::runtime_error(error);return values;};
            auto write=[&](const std::string &name,const std::vector<float> &values){
                std::ofstream file(out/name,std::ios::binary);file.write(reinterpret_cast<const char *>(values.data()),std::streamsize(values.size()*sizeof(float)));
                if(!file)throw std::runtime_error("raw row write failed");
            };
            auto walk=[&](const std::string &name,std::vector<float> row){
                std::vector<float> rows;rows.reserve(row.size()*9);
                for(int step=0;step<9;++step){
                    rows.insert(rows.end(),row.begin(),row.end());
                    if(step<8){const int32_t token=int32_t(std::max_element(row.begin(),row.end())-row.begin());append(&token,1);row=head(0);}
                }
                write(name+".f32",rows);return rows;
            };
            const int n=int(ids.size());
            if(engine.prefill_chunk(n,error)!=n)throw std::runtime_error("probe requires one actual prefill chunk: "+error);
            append(ids.data(),n);auto raw=head(n-1);
            ExactPromptCache cache;
            const int64_t alignment=std::max<int64_t>(256,engine.geometry().block);
            if(!cache.store(ids,{{0,n,((n+alignment-1)/alignment)*alignment}},raw,int(engine.geometry().vocab)))throw std::runtime_error("cache store failed");
            const auto original=walk("cold-first",raw);
            const int planned=engine.prefill_chunk_at(0,n,error);
            if(!cache.matches(ids,planned,engine.position(),int(engine.geometry().vocab)))throw std::runtime_error("fresh geometry mismatched");
            if(!engine.rewind(n,error))throw std::runtime_error(error);
            const auto reused=walk("reuse",cache.raw_logits());
            engine.reset(); // This is the native cache_prompt=false operation.
            if(engine.prefill_chunk(n,error)!=n)throw std::runtime_error("cold replay changed chunk geometry");
            append(ids.data(),n);const auto cold_again=walk("cold-again",head(n-1));
            const auto equal=[&](const std::vector<float> &a,const std::vector<float> &b){
                return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
            };
            double maximum=0;
            for(size_t i=0;i<original.size();++i){
                if(!std::isfinite(original[i]) || !std::isfinite(reused[i]) || !std::isfinite(cold_again[i]))throw std::runtime_error("nonfinite raw logits");
                maximum=std::max({maximum,double(std::abs(original[i]-reused[i])),double(std::abs(original[i]-cold_again[i]))});
            }
            const bool pass=equal(original,reused)&&equal(original,cold_again);
            result={{"pass",pass},{"scope","real Engine + exact cache helper; prompt head and eight decode rows"},
                {"prompt_tokens",n},{"padded_keys",cache.chunk().padded_keys},{"rows_per_run",9},{"vocab",engine.geometry().vocab},
                {"rows",27},{"compared_bytes",2*original.size()*sizeof(float)},{"max_abs",maximum},
                {"reused_bytes_equal",equal(original,reused)},{"cold_replay_bytes_equal",equal(original,cold_again)},
                {"context",cfg.context},{"threads",cfg.threads},{"arena_gib",cfg.arena_gib},{"gpu_experts_gib",cfg.gpu_experts_gib}};
            std::ofstream receipt(out/"result.json");receipt<<result.dump(2)<<'\n';
            if(!pass)throw std::runtime_error("raw row mismatch: "+result.dump());
        }
        ggml_backend_free(gpu);std::cout<<result.dump()<<'\n';return 0;
    } catch(const std::exception &e){std::cerr<<"prefix raw check: "<<e.what()<<'\n';if(gpu)ggml_backend_free(gpu);return 1;}
}
