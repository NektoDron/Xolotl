#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::m31 {
// GPU terms retain original rank slots; CPU terms contain only misses in original
// rank order. Each term has already been multiplied by that rank's route weight.
inline void merge_cached_decode_terms(size_t embd,const std::vector<uint8_t>&resident,
    const std::vector<float>&gpu_terms,const std::vector<float>&cpu_terms,std::vector<float>&output){
    size_t missing=0;for(auto hit:resident){if(hit>1)throw std::invalid_argument("invalid cached decode resident mask");missing+=!hit;}
    if(!embd||resident.empty()||gpu_terms.size()!=embd*resident.size()||cpu_terms.size()!=embd*missing)
        throw std::invalid_argument("cached decode contribution geometry differs");
    output.resize(embd);
    for(size_t row=0;row<embd;++row){
        float sum=0;size_t miss=0;
        for(size_t rank=0;rank<resident.size();++rank){
            const float term=resident[rank]?gpu_terms[rank*embd+row]:cpu_terms[miss++*embd+row];
            sum=rank?sum+term:term;
        }
        output[row]=sum;
    }
}
}
