// CPU fixture runner: baseline versus persistent host vectors on one Engine.
#include "m31_engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::m31;
static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
struct Run {
    std::vector<std::vector<float>> logits;
    std::vector<std::vector<uint8_t>> taps;
};
static Run run(const std::string & model, bool enabled, bool reuse, bool taps, bool cpu_residual) {
    setenv("M31_REUSE_HOST_READS", enabled ? "1" : "0", 1);
    Engine engine;
    Config cfg;
    cfg.threads=2; cfg.context=4096; cfg.max_chunk=2048;
    cfg.reuse_allocations=reuse; cfg.persistent_cpu_pool=true;
    cfg.cpu_residual=cpu_residual; cfg.cpu_decode_cache=true;
    cfg.dense_decode_cache=reuse && cpu_residual;
    cfg.profile=true;
    Run result;
    if (taps) cfg.tap=[&](const std::string & name,int64_t,const ggml_tensor * t) {
        if (name != "l_out-1" && name != "ffn_moe_out-1") return;
        check(ggml_is_contiguous(t),"unexpected noncontiguous fixture tap");
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
        result.taps.push_back(std::move(bytes));
    };
    std::string error;
    check(engine.init(model,cfg,error),error.c_str());
    std::mt19937 random(54391);
    const std::vector<std::vector<int>> phases=taps || !reuse
        ? std::vector<std::vector<int>>{{128,7,1},{1,128,7}}
        : std::vector<std::vector<int>>{{128,2048,1,7},{1,128,2048,1}};
    for (const auto & phase : phases) {
        engine.reset();
        for (int n : phase) {
            std::vector<int32_t> ids(size_t(n),0);
            for (auto & id : ids) id=int32_t(random()%64);
            check(engine.forward(ids.data(),n,error),error.c_str());
            std::vector<float> values;
            check(engine.logits(0,n,values,error),error.c_str());
            result.logits.push_back(std::move(values));
        }
    }
    // A failure/reset must neither retain stale logical rows nor damage old owned outputs.
    const auto preserved=result.logits.front();
    const int32_t invalid=-1, valid=7;
    check(!engine.forward(&invalid,1,error),"invalid ID accepted");
    engine.reset();
    check(engine.forward(&valid,1,error),error.c_str());
    std::vector<float> recovered; check(engine.logits(0,1,recovered,error),error.c_str());
    result.logits.push_back(std::move(recovered));
    check(result.logits.front()==preserved,"caller-owned previous logits changed after reset");
    return result;
}
int main(int argc,char ** argv) {
    try {
        check(argc==2,"usage: cpu_host_read_check MODEL.gguf");
        for (const auto & mode : {std::vector<bool>{true,false,true}, // reuse, taps, CPU residual
                                  std::vector<bool>{true,false,false},
                                  std::vector<bool>{true,true,true},
                                  std::vector<bool>{false,false,true}}) {
            const auto base=run(argv[1],false,mode[0],mode[1],mode[2]);
            const auto opt=run(argv[1],true,mode[0],mode[1],mode[2]);
            check(base.logits.size()==opt.logits.size(),"different output count");
            for (size_t i=0;i<base.logits.size();++i) {
                check(base.logits[i].size()==opt.logits[i].size(),"different logits shape");
                check(std::memcmp(base.logits[i].data(),opt.logits[i].data(),base.logits[i].size()*sizeof(float))==0,"host read reuse changed logits bits");
            }
            check(base.taps==opt.taps,"tap fallback changed tensor bytes");
            std::cout << "PASS host reads: reuse=" << mode[0] << " taps=" << mode[1]
                      << " cpu_residual=" << mode[2] << " outputs=" << base.logits.size()
                      << " bit-identical across shape/reset/recovery; caller outputs owned\n";
        }
        return 0;
    } catch(const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
