#pragma once
// Isolated HOST component reference. Never include in Engine/vendor/CUDA TU.
// ggml tables/types are the original vendored ggml-common.h (MIT, ggml authors).
// Compile without fast-math/reassociation; compare integer lanes BEFORE F32.
#if defined(__CUDACC__)
#error "IQ3 lane reference belongs in a separate host C++ translation unit"
#endif
#if defined(__FAST_MATH__)
#error "IQ3 lane reference requires original F32/FMA order, no fast-math"
#endif
#ifndef GGML_COMMON_DECL
#define GGML_COMMON_DECL_C
#define PORT7_IQ3_OWN_DECL
#endif
#ifndef GGML_COMMON_IMPL
#define GGML_COMMON_IMPL_C
#define PORT7_IQ3_OWN_IMPL
#endif
#include "ggml-common.h"
#ifdef PORT7_IQ3_OWN_DECL
#undef GGML_COMMON_DECL_C
#undef PORT7_IQ3_OWN_DECL
#endif
#ifdef PORT7_IQ3_OWN_IMPL
#undef GGML_COMMON_IMPL_C
#undef PORT7_IQ3_OWN_IMPL
#endif
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::m31::iq3_lane_reference {

struct BlockLanes {
    // Stage1 diagnostics: unscaled[sub32][CPUlane], then scaled[sub32][lane].
    std::array<std::array<int32_t, 8>, 8> unscaled{};
    std::array<std::array<int32_t, 8>, 8> scaled{};
    std::array<int32_t, 8> scale{};
    std::array<int32_t, 8> sum{};
};

// Matches AVX2 nrc1 and original GPU short path on valid Q8K activations.
// -128 is explicitly rejected: AVX2 psignb wraps neg(-128), unlike the old
// scalar GPU signedweight*q8 expression. quantize_row_q8_K_ref uses -127/max.
inline BlockLanes block(const block_iq3_xxs &weight, const block_q8_K &rhs) {
    BlockLanes result;
    for (int sub = 0; sub != 8; ++sub) {
        const auto *gas = weight.qs + 64 + sub * 4;
        const uint32_t packed = uint32_t(gas[0]) | (uint32_t(gas[1]) << 8)
            | (uint32_t(gas[2]) << 16) | (uint32_t(gas[3]) << 24);
        const int32_t scale = 2 * int32_t(packed >> 28) + 1;
        result.scale[sub] = scale;
        for (int lane = 0; lane != 8; ++lane) {
            const uint8_t signs = ksigns_iq2xs[(packed >> (7 * (lane / 2))) & 127];
            const uint32_t grid = iq3xxs_grid[weight.qs[sub * 8 + lane]];
            int32_t partial = 0;
            for (int j = 0; j != 4; ++j) {
                const int pos = lane * 4 + j;
                const int32_t q8 = rhs.qs[sub * 32 + pos];
                if (q8 == -128) throw std::domain_error("IQ3 valid-activation reference rejects raw q8=-128");
                const int32_t magnitude = (grid >> (8 * j)) & 255;
                partial += magnitude * q8 * ((signs & (1 << (pos % 8))) ? -1 : 1);
            }
            result.unscaled[sub][lane] = partial;
            result.scaled[sub][lane] = partial * scale;
            result.sum[lane] += result.scaled[sub][lane];
        }
    }
    return result;
}

// Fragment D mapping for mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32:
// thread holds d[0..3] = (row0,col0),(row0,col1),(row0+8,col0),(row0+8,col1).
struct FragmentPosition { int row, cpu_lane; };
inline FragmentPosition fragment_position(int warp_lane, int reg) {
    if (warp_lane < 0 || warp_lane >= 32 || reg < 0 || reg >= 4)
        throw std::out_of_range("IMMA fragment lane/register");
    return {warp_lane / 4 + (reg / 2) * 8, (warp_lane % 4) * 2 + reg % 2};
}

// d_per_block must already be original RN(F32(F16weightD)*Q8K.d).
// isums layout [K256block][CPUlane]. No reduction across K before FMA!
inline float finish(const int32_t *isums, const float *d_per_block, int blocks) {
    if (!isums || !d_per_block || blocks <= 0) throw std::invalid_argument("IQ3 finish shape");
    std::array<float, 8> acc{};
    for (int b = 0; b < blocks; ++b)
        for (int lane = 0; lane < 8; ++lane)
            acc[lane] = std::fma(d_per_block[b], float(isums[b * 8 + lane]), acc[lane]);
    // Same tree as hsum_float_8: upper+lower, pairs0+2/1+3, then0+1.
    const float p0 = acc[4] + acc[0], p1 = acc[5] + acc[1];
    const float p2 = acc[6] + acc[2], p3 = acc[7] + acc[3];
    return ((p0 + p2) + (p1 + p3)) * 0.25f;
}

} // namespace strata::m31::iq3_lane_reference
