#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::m31 {
namespace libm_expf_detail {
// Independently expressed evaluation order of this machine's glibc 2.43
// expf FMA implementation. Constants are the numeric table/polynomial values
// read from Ubuntu libm.so.6, SHA256
// 1c488db6f4fe8cecb72456bc018de000b7a9b539c2994864b0416b5e8371b3d8.
// Resolved expf ELF offset 0xa0e40; table 0xf33a0, coefficients 0xf34c0.
// Intended contract: M31 OAI finite exp arguments >= -1.702f*7.f, normal RN.
// The CPU's errno/floating-point exception side effects are not reproduced.
static __device__ __constant__ uint64_t table[32]={
    0x3ff0000000000000ULL,0x3fefd9b0d3158574ULL,0x3fefb5586cf9890fULL,0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL,0x3fef54873168b9aaULL,0x3fef387a6e756238ULL,0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL,0x3feef1a7373aa9cbULL,0x3feedea64c123422ULL,0x3feece086061892dULL,
    0x3feebfdad5362a27ULL,0x3feeb42b569d4f82ULL,0x3feeab07dd485429ULL,0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL,0x3fee9f75e8ec5f74ULL,0x3feea11473eb0187ULL,0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL,0x3feeb737b0cdc5e5ULL,0x3feec49182a3f090ULL,0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL,0x3feeff76f2fb5e47ULL,0x3fef199bdd85529cULL,0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL,0x3fef7c97337b9b5fULL,0x3fefa4afa2a490daULL,0x3fefd0765b6e4540ULL};
}
__device__ __forceinline__ float gpu_libm_expf(float value) {
    const uint32_t bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u)return __fadd_rn(value,value);
    if(value>0x1.62e42ep+6f)return __int_as_float(0x7f800000);
    if(value<-0x1.9fe368p+6f)return 0.f;
    // This interval cannot occur in OAI because its gate is capped at +7.
    if(value<-0x1.9d1d9ep+6f)return __double2float_rn(exp(double(value)));
    constexpr double invln2_scaled=0x1.71547652b82fep+5;
    constexpr double shift=0x1.8p+52;
    const double x=double(value);
    const double rounded=__fma_rn(invln2_scaled,x,shift);
    const uint64_t index=uint64_t(__double_as_longlong(rounded));
    const double integral=__dsub_rn(rounded,shift);
    const double remainder=__fma_rn(invln2_scaled,x,-integral);
    const uint64_t scale_bits=libm_expf_detail::table[index&31]+(index<<47);
    const double scale=__longlong_as_double(static_cast<long long>(scale_bits));
    const double squared=__dmul_rn(remainder,remainder);
    const double quadratic=__fma_rn(0x1.c6af84b912394p-20,remainder,0x1.ebfce50fac4f3p-13);
    const double linear=__fma_rn(remainder,0x1.62e42ff0c52d6p-6,1.0);
    const double polynomial=__fma_rn(quadratic,squared,linear);
    return __double2float_rn(__dmul_rn(polynomial,scale));
}
} // namespace strata::m31
