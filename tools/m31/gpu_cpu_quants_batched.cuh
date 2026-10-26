// Weight and activation reuse across a 16x16 output tile. Integer dot products
// retain the original 256-value CPU reduction and ordered float block epilogue.
// Included inside the CUDA implementation's anonymous namespace.
template<int Type>
__global__ void multiply_batched_kernel(const uint8_t * pool, size_t stride,
        int columns, int rows, int used, int n, const uint8_t * quant,
        size_t quant_row_bytes, bool broadcast, const int32_t * ids,
        const int32_t * routes, const int32_t * tiles, float * output) {
    constexpr int tile = 16;
    __shared__ int weights[tile][65];
    __shared__ int activations[tile][65];
    __shared__ int scales[tile][9];
    __shared__ float wd[tile], ad[tile];
    __shared__ int original[tile];
    const int thread = threadIdx.x;
    const int r = thread % tile, t = thread / tile;
    const int row = blockIdx.x*tile+r;
    const int slot = tiles[blockIdx.y*3];
    const int begin = tiles[blockIdx.y*3+1];
    const int count = tiles[blockIdx.y*3+2];
    if (thread < tile) original[thread] = thread < count ? routes[begin+thread] : -1;
    __syncthreads();
    const int route = original[t];
    const bool tiled = Type != GGML_TYPE_NVFP4 && route >= 0 && ids[size_t(used)*n+route];
    const auto * w = pool+size_t(slot)*stride;
    float acc = 0;
    float lane_acc[8] = {};
    for (int b = 0; b < columns/256; ++b) {
        // Every compressed weight is expanded once for all 16 activation rows.
        for (int index = thread; index < tile*64; index += 256) {
            const int wr = int(blockIdx.x)*tile+index/64;
            const int sub = (index%64)/8, lane = index%8;
            int packed = 0;
            if (wr < rows) {
                if constexpr (Type == GGML_TYPE_IQ3_XXS) {
                    const auto & x = reinterpret_cast<const block_iq3_xxs *>(w)[size_t(wr)*(columns/256)+b];
                    const auto * gas = x.qs+64+sub*4;
                    const uint32_t bits = uint32_t(gas[0]) | (uint32_t(gas[1])<<8) | (uint32_t(gas[2])<<16) | (uint32_t(gas[3])<<24);
                    const uint8_t signs = ksigns_iq2xs[(bits>>(7*(lane/2)))&127];
                    const uint32_t grid = iq3xxs_grid[x.qs[sub*8+lane]];
                    for (int j = 0; j < 4; ++j) {
                        const int value = int((grid>>(8*j))&255)*((signs&(1<<((lane*4+j)%8))) ? -1 : 1);
                        packed |= int(uint32_t(uint8_t(value))<<(j*8));
                    }
                    if (!lane) scales[index/64][sub] = 2*(bits>>28)+1;
                    if (!(index%64)) wd[index/64] = __fmul_rn(__half2float(x.d),0.25f);
                } else {
                    const auto & x = reinterpret_cast<const block_iq4_xs *>(w)[size_t(wr)*(columns/256)+b];
                    for (int j = 0; j < 4; ++j) {
                        const int pos = lane*4+j;
                        const uint8_t q = x.qs[sub*16+pos%16];
                        const int value = kvalues_iq4nl[pos<16 ? q&15 : q>>4];
                        packed |= int(uint32_t(uint8_t(value))<<(j*8));
                    }
                    if (!lane) scales[index/64][sub] = (((x.scales_l[sub/2]>>(4*(sub%2)))&15) | (((x.scales_h>>(2*sub))&3)<<4))-32;
                    if (!(index%64)) wd[index/64] = __half2float(x.d);
                }
            }
            weights[index/64][index%64] = packed;
            const int ar = original[index/64];
            const auto * y = ar < 0 ? nullptr : reinterpret_cast<const block_q8_K *>(quant+size_t(broadcast ? ar/used : ar)*quant_row_bytes)+b;
            activations[index/64][index%64] = y ? reinterpret_cast<const int *>(y->qs)[index%64] : 0;
            if (!(index%64)) ad[index/64] = y ? y->d : 0;
        }
        __syncthreads();
        if (row < rows && route >= 0) {
            if (tiled) {
                int total = 0;
                for (int sub = 0; sub < 8; ++sub) {
                    int sum = 0;
                    #pragma unroll
                    for (int lane = 0; lane < 8; ++lane)
                        sum = __dp4a(weights[r][sub*8+lane],activations[t][sub*8+lane],sum);
                    total += sum*scales[r][sub];
                }
                const float product = __fmul_rn(__int2float_rn(total),wd[r]);
                acc = fmaf(product,ad[t],acc);
            } else {
                // Preserve eight independent AVX2 accumulators for short experts.
                const float d = Type == GGML_TYPE_IQ3_XXS ?
                    __fmul_rn(__fmul_rn(wd[r],4.f),ad[t]) : __fmul_rn(wd[r],ad[t]);
                #pragma unroll
                for (int lane = 0; lane < 8; ++lane) {
                    int sum = 0;
                    for (int sub = 0; sub < 8; ++sub)
                        sum += __dp4a(weights[r][sub*8+lane],activations[t][sub*8+lane],0)*scales[r][sub];
                    lane_acc[lane] = fmaf(d,__int2float_rn(sum),lane_acc[lane]);
                }
            }
        }
        __syncthreads();
    }
    if (row < rows && route >= 0) {
        if (!tiled) {
            // Same xor4, xor2, xor1 tree as the scalar eight-lane kernel.
            const float a0 = __fadd_rn(lane_acc[0],lane_acc[4]);
            const float a1 = __fadd_rn(lane_acc[1],lane_acc[5]);
            const float a2 = __fadd_rn(lane_acc[2],lane_acc[6]);
            const float a3 = __fadd_rn(lane_acc[3],lane_acc[7]);
            acc = __fadd_rn(__fadd_rn(a0,a2),__fadd_rn(a1,a3));
            if constexpr (Type == GGML_TYPE_IQ3_XXS) acc = __fmul_rn(acc,0.25f);
        }
        output[size_t(route)*rows+row] = acc;
    }
}
