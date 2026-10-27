#pragma once
// Adapted from ggml x86 quants.c / ggml-common.h at
// 6050b467c07a95c7b87debd4ddac2f8ba8d9b807. Original nrc=1 dot arithmetic
// is retained independently for two RHS; only weight unpack is shared.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "ggml.h"
#include <cstdint>
#include <cstring>
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
extern "C" float ggml_table_f32_f16[1 << 16];
#endif
namespace strata::m31::cpu_pair_detail {
struct Q8K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct IQ3 { uint16_t d; uint8_t qs[96]; };
struct IQ4 { uint16_t d,scales_h;uint8_t scales_l[4],qs[128]; };
static_assert(sizeof(Q8K)==292 && sizeof(IQ3)==98 && sizeof(IQ4)==136);
inline constexpr int8_t iq4_values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
alignas(32) inline constexpr int8_t iq3_signs[1024]={
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
alignas(32) inline constexpr uint32_t iq3_grid[256]={
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,

};
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define M31_PAIR_TARGET __attribute__((target("avx2,fma"),noinline))
__attribute__((target("avx2,fma"))) inline float hsum8(__m256 x) {
    __m128 res=_mm256_extractf128_ps(x,1);
    res=_mm_add_ps(res,_mm256_castps256_ps128(x));
    res=_mm_add_ps(res,_mm_movehl_ps(res,res));
    res=_mm_add_ss(res,_mm_movehdup_ps(res));return _mm_cvtss_f32(res);
}
__attribute__((target("avx2,fma"))) inline __m256i mul_add(__m256i x,__m256i y) {
    const __m256i ax=_mm256_sign_epi8(x,x),sy=_mm256_sign_epi8(y,x);
    return _mm256_maddubs_epi16(ax,sy);
}
inline int64_t signs64(unsigned index){int64_t result;std::memcpy(&result,iq3_signs+index*8,8);return result;}
M31_PAIR_TARGET inline void iq4_pair(int n,const void * weights,const void * rhs0,const void * rhs1,float & out0,float & out1) {
    const auto * x=static_cast<const IQ4 *>(weights);const Q8K * y[2]={static_cast<const Q8K *>(rhs0),static_cast<const Q8K *>(rhs1)};
    const __m128i values=_mm_loadu_si128(reinterpret_cast<const __m128i *>(iq4_values));const __m128i mask=_mm_set1_epi8(15);
    __m256 accum0=_mm256_setzero_ps(),accum1=_mm256_setzero_ps();
    for(int block=0;block<n/256;++block) {
        const uint8_t * qs=x[block].qs;uint16_t sh=x[block].scales_h;
        __m256i s10=_mm256_setzero_si256(),s20=_mm256_setzero_si256(),s11=_mm256_setzero_si256(),s21=_mm256_setzero_si256();
        for(int ib=0;ib<8;ib+=2) {
            const __m128i bits1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));qs+=16;
            const __m128i bits2=_mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));qs+=16;
            const __m256i q1=_mm256_set_m128i(_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(bits1,4),mask)),_mm_shuffle_epi8(values,_mm_and_si128(bits1,mask)));
            const __m256i q2=_mm256_set_m128i(_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(bits2,4),mask)),_mm_shuffle_epi8(values,_mm_and_si128(bits2,mask)));
            const int16_t ls1=((x[block].scales_l[ib/2]&15)|((sh<<4)&48))-32;
            const int16_t ls2=((x[block].scales_l[ib/2]>>4)|((sh<<2)&48))-32;sh>>=4;
            const __m256i scale1=_mm256_set1_epi16(ls1),scale2=_mm256_set1_epi16(ls2);
            const __m256i r10=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[0][block].qs+ib*32));
            const __m256i r20=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[0][block].qs+(ib+1)*32));
            const __m256i r11=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[1][block].qs+ib*32));
            const __m256i r21=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[1][block].qs+(ib+1)*32));
            s10=_mm256_add_epi32(_mm256_madd_epi16(mul_add(q1,r10),scale1),s10);
            s20=_mm256_add_epi32(_mm256_madd_epi16(mul_add(q2,r20),scale2),s20);
            s11=_mm256_add_epi32(_mm256_madd_epi16(mul_add(q1,r11),scale1),s11);
            s21=_mm256_add_epi32(_mm256_madd_epi16(mul_add(q2,r21),scale2),s21);
        }
        const float dx=ggml_table_f32_f16[x[block].d];
        accum0=_mm256_fmadd_ps(_mm256_set1_ps(dx*y[0][block].d),_mm256_cvtepi32_ps(_mm256_add_epi32(s10,s20)),accum0);
        accum1=_mm256_fmadd_ps(_mm256_set1_ps(dx*y[1][block].d),_mm256_cvtepi32_ps(_mm256_add_epi32(s11,s21)),accum1);
    }
    out0=hsum8(accum0);out1=hsum8(accum1);
}
M31_PAIR_TARGET inline void iq3_pair(int n,const void * weights,const void * rhs0,const void * rhs1,float & out0,float & out1) {
    const auto * x=static_cast<const IQ3 *>(weights);const Q8K * y[2]={static_cast<const Q8K *>(rhs0),static_cast<const Q8K *>(rhs1)};
    __m256 accum0=_mm256_setzero_ps(),accum1=_mm256_setzero_ps();
    for(int block=0;block<n/256;++block) {
        const uint8_t * q3=x[block].qs,* gas=x[block].qs+64;
        __m256i s10=_mm256_setzero_si256(),s20=_mm256_setzero_si256(),s11=_mm256_setzero_si256(),s21=_mm256_setzero_si256();
        for(int ib=0;ib<8;ib+=2) {
            const __m256i q1=_mm256_set_epi32(iq3_grid[q3[7]],iq3_grid[q3[6]],iq3_grid[q3[5]],iq3_grid[q3[4]],iq3_grid[q3[3]],iq3_grid[q3[2]],iq3_grid[q3[1]],iq3_grid[q3[0]]);q3+=8;
            const __m256i q2=_mm256_set_epi32(iq3_grid[q3[7]],iq3_grid[q3[6]],iq3_grid[q3[5]],iq3_grid[q3[4]],iq3_grid[q3[3]],iq3_grid[q3[2]],iq3_grid[q3[1]],iq3_grid[q3[0]]);q3+=8;
            uint32_t aux[2];std::memcpy(aux,gas,8);gas+=8;
            const __m256i sign1=_mm256_set_epi64x(signs64((aux[0]>>21)&127),signs64((aux[0]>>14)&127),signs64((aux[0]>>7)&127),signs64(aux[0]&127));
            const __m256i sign2=_mm256_set_epi64x(signs64((aux[1]>>21)&127),signs64((aux[1]>>14)&127),signs64((aux[1]>>7)&127),signs64(aux[1]&127));
            const __m256i scale1=_mm256_set1_epi16(2*(aux[0]>>28)+1),scale2=_mm256_set1_epi16(2*(aux[1]>>28)+1);
            const __m256i r10=_mm256_sign_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[0][block].qs+ib*32)),sign1);
            const __m256i r20=_mm256_sign_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[0][block].qs+(ib+1)*32)),sign2);
            const __m256i r11=_mm256_sign_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[1][block].qs+ib*32)),sign1);
            const __m256i r21=_mm256_sign_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[1][block].qs+(ib+1)*32)),sign2);
            s10=_mm256_add_epi32(s10,_mm256_madd_epi16(_mm256_maddubs_epi16(q1,r10),scale1));
            s20=_mm256_add_epi32(s20,_mm256_madd_epi16(_mm256_maddubs_epi16(q2,r20),scale2));
            s11=_mm256_add_epi32(s11,_mm256_madd_epi16(_mm256_maddubs_epi16(q1,r11),scale1));
            s21=_mm256_add_epi32(s21,_mm256_madd_epi16(_mm256_maddubs_epi16(q2,r21),scale2));
        }
        const float dx=ggml_table_f32_f16[x[block].d];
        accum0=_mm256_fmadd_ps(_mm256_set1_ps(dx*y[0][block].d),_mm256_cvtepi32_ps(_mm256_add_epi32(s10,s20)),accum0);
        accum1=_mm256_fmadd_ps(_mm256_set1_ps(dx*y[1][block].d),_mm256_cvtepi32_ps(_mm256_add_epi32(s11,s21)),accum1);
    }
    out0=.25f*hsum8(accum0);out1=.25f*hsum8(accum1);
}
#undef M31_PAIR_TARGET
inline bool available(){static const bool yes=__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");return yes;}
#else
inline bool available(){return false;}
#endif
inline bool dot_pair(ggml_type type,int n,const void * weights,const void * rhs0,const void * rhs1,float & out0,float & out1,unsigned mode=2) {
    if(n<=0||n%256||!weights||!rhs0||!rhs1||!available())return false;
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    if(type==GGML_TYPE_IQ4_XS && mode>=1){iq4_pair(n,weights,rhs0,rhs1,out0,out1);return true;}
    if(type==GGML_TYPE_IQ3_XXS && mode>=2){iq3_pair(n,weights,rhs0,rhs1,out0,out1);return true;}
#endif
    return false;
}
} // namespace strata::m31::cpu_pair_detail
