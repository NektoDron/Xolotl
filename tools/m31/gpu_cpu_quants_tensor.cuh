// Tiled CPU contract only: exact dot32 through two signed-int8 MMA operations,
// integer scale application and dot256, then the original ordered F32 FMA.
// Each CTA reuses 16 decoded weight rows across 64 routed activation rows.
template<int Type>
__global__ void multiply_tensor_kernel(const uint8_t * pool,size_t stride,
        int columns,int rows,int used,int n,const uint8_t * quant,
        size_t quant_row_bytes,bool broadcast,const int32_t * routes,
        const int32_t * tiles,float * output) {
    constexpr int m = 16, nt = 64, ld = 32;
    // Each K16 half starts at a 32-byte aligned base, as required by WMMA.
    // Stage one K32 sub-block at a time instead of padding the full K256 block.
    __shared__ __align__(32) int8_t weights[2][m][ld];
    __shared__ __align__(32) int8_t activations[2][nt][ld];
    __shared__ __align__(32) int dots[4][m*16];
    __shared__ int scales[m], original[nt];
    __shared__ float wd[m], ad[nt];
    const int thread = threadIdx.x, warp = thread/32, lane = thread%32;
    const int slot = tiles[blockIdx.y*3], begin = tiles[blockIdx.y*3+1], count = tiles[blockIdx.y*3+2];
    if (thread < nt) original[thread] = thread < count ? routes[begin+thread] : -1;
    __syncthreads();
    const auto * w = pool+size_t(slot)*stride;
    float acc[8] = {};
    for (int b = 0; b < columns/256; ++b) {
      int total[8] = {};
      for (int sub = 0; sub < 8; ++sub) {
        {
            const int wr = int(blockIdx.x)*m+thread/8;
            const int group = thread%8;
            uint32_t packed = 0;
            if (wr < rows) {
                if constexpr (Type == GGML_TYPE_IQ3_XXS) {
                    const auto & x = reinterpret_cast<const block_iq3_xxs *>(w)[size_t(wr)*(columns/256)+b];
                    const auto * gas = x.qs+64+sub*4;
                    const uint32_t bits = uint32_t(gas[0]) | (uint32_t(gas[1])<<8) | (uint32_t(gas[2])<<16) | (uint32_t(gas[3])<<24);
                    const uint8_t signs = ksigns_iq2xs[(bits>>(7*(group/2)))&127];
                    const uint32_t grid = iq3xxs_grid[x.qs[sub*8+group]];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int value = int((grid>>(8*j))&255)*((signs&(1<<((group*4+j)%8))) ? -1 : 1);
                        packed |= uint32_t(uint8_t(value))<<(j*8);
                    }
                    if (!group) scales[thread/8] = 2*(bits>>28)+1;
                    if (!group) wd[thread/8] = __fmul_rn(__half2float(x.d),0.25f);
                } else {
                    const auto & x = reinterpret_cast<const block_iq4_xs *>(w)[size_t(wr)*(columns/256)+b];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int pos = group*4+j;
                        const uint8_t q = x.qs[sub*16+pos%16];
                        packed |= uint32_t(uint8_t(kvalues_iq4nl[pos<16 ? q&15 : q>>4]))<<(j*8);
                    }
                    if (!group) scales[thread/8] = (((x.scales_l[sub/2]>>(4*(sub%2)))&15) | (((x.scales_h>>(2*sub))&3)<<4))-32;
                    if (!group) wd[thread/8] = __half2float(x.d);
                }
            }
            reinterpret_cast<uint32_t *>(weights[group/4][thread/8])[group%4] = packed;
        }
        for (int index = thread; index < nt*8; index += 128) {
            const int route = original[index/8], group = index%8;
            const auto * y = route < 0 ? nullptr : reinterpret_cast<const block_q8_K *>(quant+size_t(broadcast ? route/used : route)*quant_row_bytes)+b;
            reinterpret_cast<uint32_t *>(activations[group/4][index/8])[group%4] = y ? reinterpret_cast<const uint32_t *>(y->qs)[sub*8+group] : 0;
            if (!group) ad[index/8] = y ? y->d : 0;
        }
        __syncthreads();
        {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,signed char,nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,signed char,nvcuda::wmma::col_major> v;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,int> dot;
            nvcuda::wmma::fill_fragment(dot,0);
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                nvcuda::wmma::load_matrix_sync(a,weights[half][0],ld);
                nvcuda::wmma::load_matrix_sync(v,activations[half][warp*16],ld);
                nvcuda::wmma::mma_sync(dot,a,v,dot);
            }
            nvcuda::wmma::store_matrix_sync(dots[warp],dot,16,nvcuda::wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int index = lane+i*32, r = index/16;
                if (int(blockIdx.x)*m+r < rows) total[i] += dots[warp][index]*scales[r];
            }
            __syncwarp();
        }
        __syncthreads();
      }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int index = lane+i*32, r = index/16, t = warp*16+index%16;
            if (int(blockIdx.x)*m+r < rows && original[t] >= 0)
                acc[i] = fmaf(__fmul_rn(__int2float_rn(total[i]),wd[r]),ad[t],acc[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int index = lane+i*32, row = int(blockIdx.x)*m+index/16, route = original[warp*16+index%16];
        if (row < rows && route >= 0) output[size_t(route)*rows+row] = acc[i];
    }
}
