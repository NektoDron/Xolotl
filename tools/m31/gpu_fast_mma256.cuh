// Opt-in whole-K256 integer MMA. IQ scale_sub is folded into signed16 weights,
// then split exactly as W=128*hi+lo, lo in[0,127], hi in[-32,31]. Two signed8
// dot256 accumulators replace eight scaled dot32 stages. Their integer sum and
// the original wd/ad/ordered FMA epilogue reproduce the CPU tiled contract.
// Largest original integer bound: 4064*127*256 < 2^31.
#pragma once
__device__ __forceinline__ void m31_mma_s8_k32_accumulate(int *dot,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

template<int Type>
__global__ void multiply_tensor_mma256_kernel(const uint8_t * pool,size_t stride,
        int columns,int rows,int used,int n,const uint8_t * quant,
        size_t quant_row_bytes,bool broadcast,const int32_t * routes,
        const int32_t * tiles,float * output) {
    static_assert(Type==GGML_TYPE_IQ3_XXS||Type==GGML_TYPE_IQ4_XS,"MMA256 supports Q8_K IQ experts");
    constexpr int m=16,nt=64;
    __shared__ __align__(32) uint32_t weight_lo[m][65],weight_hi[m][65];
    __shared__ __align__(32) uint32_t activations[nt][65];
    __shared__ int original[nt];
    __shared__ float wd[m],ad[nt];
    const int thread=threadIdx.x,warp=thread/32,lane=thread%32;
    const int slot=tiles[blockIdx.y*3],begin=tiles[blockIdx.y*3+1],count=tiles[blockIdx.y*3+2];
    if(thread<nt)original[thread]=thread<count?routes[begin+thread]:-1;
    __syncthreads();
    const auto *w=pool+size_t(slot)*stride;
    float acc[8]={};
    for(int block=0;block<columns/256;++block) {
        for(int index=thread;index<m*64;index+=128) {
            const int wr=int(blockIdx.x)*m+index/64,sub=(index%64)/8,group=index%8;
            uint32_t lo=0,hi=0;
            if(wr<rows) {
                if constexpr(Type==GGML_TYPE_IQ3_XXS) {
                    const auto &x=reinterpret_cast<const block_iq3_xxs *>(w)[size_t(wr)*(columns/256)+block];
                    const auto *gas=x.qs+64+sub*4;
                    const uint32_t bits=uint32_t(gas[0])|(uint32_t(gas[1])<<8)|(uint32_t(gas[2])<<16)|(uint32_t(gas[3])<<24);
                    const int scale=2*int(bits>>28)+1;
                    const uint8_t signs=ksigns_iq2xs[(bits>>(7*(group/2)))&127];
                    const uint32_t grid=iq3xxs_grid[x.qs[sub*8+group]];
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const int value=scale*int((grid>>(8*j))&255)*((signs&(1<<((group*4+j)%8)))?-1:1);
                        lo|=uint32_t(uint8_t(value&127))<<(8*j);
                        hi|=uint32_t(uint8_t(value>>7))<<(8*j);
                    }
                    if(!(index%64))wd[index/64]=__fmul_rn(__half2float(x.d),0.25f);
                } else {
                    const auto &x=reinterpret_cast<const block_iq4_xs *>(w)[size_t(wr)*(columns/256)+block];
                    const int scale=(((x.scales_l[sub/2]>>(4*(sub%2)))&15)|(((x.scales_h>>(2*sub))&3)<<4))-32;
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const int pos=group*4+j;
                        const uint8_t q=x.qs[sub*16+pos%16];
                        const int value=scale*int(kvalues_iq4nl[pos<16?q&15:q>>4]);
                        lo|=uint32_t(uint8_t(value&127))<<(8*j);
                        hi|=uint32_t(uint8_t(value>>7))<<(8*j);
                    }
                    if(!(index%64))wd[index/64]=__half2float(x.d);
                }
            } else if(!(index%64))wd[index/64]=0;
            weight_lo[index/64][index%64]=lo;weight_hi[index/64][index%64]=hi;
        }
        for(int index=thread;index<nt*64;index+=128) {
            const int ar=original[index/64];
            const auto *y=ar<0?nullptr:reinterpret_cast<const block_q8_K *>(quant+size_t(broadcast?ar/used:ar)*quant_row_bytes)+block;
            activations[index/64][index%64]=y?reinterpret_cast<const uint32_t *>(y->qs)[index%64]:0;
            if(!(index%64))ad[index/64]=y?y->d:0;
        }
        __syncthreads();
        int dot_lo[8]={},dot_hi[8]={};
        #pragma unroll
        for(int sub=0;sub<8;++sub) {
            const int ar=lane/4,ac=sub*8+lane%4;
            const uint32_t al0=weight_lo[ar][ac],al1=weight_lo[ar+8][ac],al2=weight_lo[ar][ac+4],al3=weight_lo[ar+8][ac+4];
            const uint32_t ah0=weight_hi[ar][ac],ah1=weight_hi[ar+8][ac],ah2=weight_hi[ar][ac+4],ah3=weight_hi[ar+8][ac+4];
            #pragma unroll
            for(int half_n=0;half_n<2;++half_n) {
                const int token=warp*16+half_n*8+lane/4;
                const uint32_t b0=activations[token][sub*8+lane%4],b1=activations[token][sub*8+lane%4+4];
                m31_mma_s8_k32_accumulate(dot_lo+half_n*4,al0,al1,al2,al3,b0,b1);
                m31_mma_s8_k32_accumulate(dot_hi+half_n*4,ah0,ah1,ah2,ah3,b0,b1);
            }
        }
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int part=i%4,r=(part/2)*8+lane/4,t=warp*16+(i/4)*8+(lane%4)*2+part%2;
            if(int(blockIdx.x)*m+r<rows&&original[t]>=0) {
                const int total=dot_lo[i]+128*dot_hi[i];
                acc[i]=fmaf(__fmul_rn(__int2float_rn(total),wd[r]),ad[t],acc[i]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int part=i%4,row=int(blockIdx.x)*m+(part/2)*8+lane/4;
        const int route=original[warp*16+(i/4)*8+(lane%4)*2+part%2];
        if(row<rows&&route>=0)output[size_t(route)*rows+row]=acc[i];
    }
}
