#include "shortprefill_cuda_nvfp4_cpu.hpp"
#include <cassert>
#include <iostream>
#include <random>
#ifdef M31_TEST_CUDA_HEADERS
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif
using namespace strata::m31::cuda_nvfp4_cpu;
int main(){
    for(int code=0;code<127;++code)assert(fp8_code(2.f*fp8_half(uint8_t(code)))==code);
    const float ties[]={.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f};
    const int even[]={0,2,2,4,4,6,6};
    for(int i=0;i<7;++i){assert(fp4_code(ties[i])==even[i]);assert(fp4_code(-ties[i])==(even[i]|8));}
    assert(fp8_code(1e20f)==126&&fp8_code(0)==0&&fp4_code(1e20f)==7);
    Activation a;float input[256]{};quantize(input,256,a);assert(a.row_scale==0);
    for(auto code:a.scales)assert(code==0);for(auto value:a.values)assert(value==0);
    uint8_t weights[4*36]{};assert(dot(weights,a,256)==0);
    // Constant rows are represented exactly after the row and subblock scales.
    std::fill(input,input+256,1.f);quantize(input,256,a);
    for(auto &byte:weights)byte=0;
    for(int block=0;block<4;++block){auto *w=weights+36*block;for(int sub=0;sub<4;++sub){w[sub]=fp8_code(1.f);std::fill(w+4+sub*8,w+4+(sub+1)*8,uint8_t(0x22));}}
    const float result=dot(weights,a,256);assert(std::fabs(result-256.f)<1e-3f);
    std::mt19937 rng(42);std::uniform_int_distribution<int> bytes(0,255),values(-12,12);
    for(int trial=0;trial<10000;++trial){uint8_t w[8];int8_t x[16];for(auto &v:w)v=uint8_t(bytes(rng));for(auto &v:x)v=int8_t(values(rng));
#if defined(__x86_64__) || defined(__i386__)
        if(__builtin_cpu_supports("ssse3"))assert(integer_dot_simd(w,x)==integer_dot_scalar(w,x));
#endif
    }
#ifdef M31_TEST_CUDA_HEADERS
    std::uniform_real_distribution<float> float_values(-1000.f,1000.f);
    for(int trial=0;trial<200000;++trial){const float value=float_values(rng);
        assert(fp4_code(value)==__nv_fp4_e2m1(value).__x);
        assert(fp8_code(std::fabs(value))==__nv_fp8_e4m3(std::fabs(value)).__x);}
    for(float tie:ties){assert(fp4_code(tie)==__nv_fp4_e2m1(tie).__x);assert(fp4_code(-tie)==__nv_fp4_e2m1(-tie).__x);}
#endif
    bool threw=false;try{quantize(input,63,a);}catch(const std::invalid_argument&){threw=true;}assert(threw);
    std::cout<<"shortprefill CUDA NVFP4 CPU quant and integer-dot PASS (GPU parity still required)\n";
}
