// Optional CPU-compatible IQ3_XXS/IQ4_XS integer dot kernel using explicit
// m16n8k32 MMA fragments. Included after ggml CUDA quant declarations inside
// the implementation's anonymous namespace. Original dot256 scale/ordered FMA
// epilogue is retained; only integer dot32 scheduling and register layout change.
#pragma once
__device__ __forceinline__ void m31_mma_s8_k32(int (&dot)[4],
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    dot[0]=dot[1]=dot[2]=dot[3]=0;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

template<int Type>
__global__ void multiply_tensor_mma32_kernel(const uint8_t * pool,size_t stride,
        int columns,int rows,int used,int n,const uint8_t * quant,
        size_t quant_row_bytes,bool broadcast,const int32_t * routes,
        const int32_t * tiles,float * output) {
    static_assert(Type==GGML_TYPE_IQ3_XXS||Type==GGML_TYPE_IQ4_XS,"MMA32 supports Q8_K-compatible IQ experts");
    constexpr int m=16,nt=64;
    __shared__ __align__(32) uint32_t weights[m][8];
    __shared__ __align__(32) uint32_t activations[nt][8];
    __shared__ int scales[m],original[nt];
    __shared__ float wd[m],ad[nt];
    const int thread=threadIdx.x,warp=thread/32,lane=thread%32;
    const int slot=tiles[blockIdx.y*3],begin=tiles[blockIdx.y*3+1],count=tiles[blockIdx.y*3+2];
    if(thread<nt)original[thread]=thread<count?routes[begin+thread]:-1;
    __syncthreads();
    const auto * w=pool+size_t(slot)*stride;
    float acc[8]={};
    for(int block=0;block<columns/256;++block) {
        int total[8]={};
        for(int sub=0;sub<8;++sub) {
            const int local_row=thread/8,group=thread%8,wr=int(blockIdx.x)*m+local_row;
            uint32_t packed=0;
            if(wr<rows) {
                if constexpr(Type==GGML_TYPE_IQ3_XXS) {
                    const auto & x=reinterpret_cast<const block_iq3_xxs *>(w)[size_t(wr)*(columns/256)+block];
                    const auto * gas=x.qs+64+sub*4;
                    const uint32_t bits=uint32_t(gas[0])|(uint32_t(gas[1])<<8)|(uint32_t(gas[2])<<16)|(uint32_t(gas[3])<<24);
                    const uint8_t signs=ksigns_iq2xs[(bits>>(7*(group/2)))&127];
                    const uint32_t grid=iq3xxs_grid[x.qs[sub*8+group]];
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const int value=int((grid>>(8*j))&255)*((signs&(1<<((group*4+j)%8)))?-1:1);
                        packed|=uint32_t(uint8_t(value))<<(8*j);
                    }
                    if(!group)scales[local_row]=2*int(bits>>28)+1;
                    if(!group)wd[local_row]=__fmul_rn(__half2float(x.d),0.25f);
                } else {
                    const auto & x=reinterpret_cast<const block_iq4_xs *>(w)[size_t(wr)*(columns/256)+block];
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const int pos=group*4+j;
                        const uint8_t q=x.qs[sub*16+pos%16];
                        packed|=uint32_t(uint8_t(kvalues_iq4nl[pos<16?q&15:q>>4]))<<(8*j);
                    }
                    if(!group)scales[local_row]=(((x.scales_l[sub/2]>>(4*(sub%2)))&15)|(((x.scales_h>>(2*sub))&3)<<4))-32;
                    if(!group)wd[local_row]=__half2float(x.d);
                }
            } else if(!group) {
                scales[local_row]=0;wd[local_row]=0;
            }
            weights[local_row][group]=packed;
            for(int index=thread;index<nt*8;index+=128) {
                const int ar=original[index/8],g=index%8;
                const auto * y=ar<0?nullptr:reinterpret_cast<const block_q8_K *>(quant+size_t(broadcast?ar/used:ar)*quant_row_bytes)+block;
                activations[index/8][g]=y?reinterpret_cast<const uint32_t *>(y->qs)[sub*8+g]:0;
                if(!g)ad[index/8]=y?y->d:0;
            }
            __syncthreads();
            // PTX A: rows lane/4 and lane/4+8 at K[0..15], then the
            // same rows at K[16..31]. PTX B packs the two K16 halves.
            const int ar=lane/4,ac=lane%4;
            const uint32_t a0=weights[ar][ac],a1=weights[ar+8][ac];
            const uint32_t a2=weights[ar][ac+4],a3=weights[ar+8][ac+4];
            #pragma unroll
            for(int half_n=0;half_n<2;++half_n) {
                const int token=warp*16+half_n*8+lane/4;
                int dot[4];
                m31_mma_s8_k32(dot,a0,a1,a2,a3,
                    activations[token][lane%4],activations[token][lane%4+4]);
                #pragma unroll
                for(int i=0;i<4;++i)total[half_n*4+i]+=dot[i]*scales[(i/2)*8+lane/4];
            }
            __syncthreads();
        }
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int part=i%4,r=(part/2)*8+lane/4,t=warp*16+(i/4)*8+(lane%4)*2+part%2;
            if(int(blockIdx.x)*m+r<rows&&original[t]>=0)
                acc[i]=fmaf(__fmul_rn(__int2float_rn(total[i]),wd[r]),ad[t],acc[i]);
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
