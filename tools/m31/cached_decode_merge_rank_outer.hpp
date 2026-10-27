#pragma once
// Independent opt-in proposal; production merge_cached_decode_terms is unchanged.
#include "cached_decode_merge.hpp"
#include <array>
#include <cstring>
namespace strata::m31 {
inline void merge_cached_decode_terms_rank_outer(size_t embd,const std::vector<uint8_t>&resident,
    const std::vector<float>&gpu_terms,const std::vector<float>&cpu_terms,std::vector<float>&output){
    // Keep the previous generic API for exotic sizes; M31 uses four ranks.
    if(resident.size()>128){merge_cached_decode_terms(embd,resident,gpu_terms,cpu_terms,output);return;}
    size_t missing=0;for(auto hit:resident){if(hit>1)throw std::invalid_argument("invalid cached decode resident mask");missing+=!hit;}
    if(!embd||resident.empty()||gpu_terms.size()!=embd*resident.size()||cpu_terms.size()!=embd*missing||&output==&gpu_terms||&output==&cpu_terms)
        throw std::invalid_argument("cached decode contribution geometry/alias differs");
    std::array<const float*,128>terms;size_t miss=0;
    for(size_t rank=0;rank<resident.size();++rank)terms[rank]=resident[rank]?gpu_terms.data()+rank*embd:cpu_terms.data()+miss++*embd;
    output.resize(embd);
    std::memcpy(output.data(),terms[0],embd*sizeof(float)); // Preserve rank0 bits, including -0.
    for(size_t rank=1;rank<resident.size();++rank){
        float * __restrict__ destination=output.data();
        const float * __restrict__ source=terms[rank];
        for(size_t row=0;row<embd;++row)destination[row]=destination[row]+source[row];
    }
}
} // namespace strata::m31
