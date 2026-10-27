#include "gpu_fast_ffn.hpp"
#include "prefill_route_worklist.hpp"
#include "ggml.h"
#include "ggml-backend-impl.h"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <limits>
#include <filesystem>
#include <fstream>
#include "gpu_fast_quant.cuh"
#include "gpu_libm_expf.cuh"
#ifdef M31_CPU_QUANTS_CUDA
#include "gpu_resident_dot.hpp"
#endif

namespace strata::m31 {
namespace {
void fast_require(bool ok,const char * msg) {
    if (!ok) throw std::runtime_error(std::string("M31 fast FFN: ")+msg);
}
void fast_cuda(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string("M31 fast FFN: ")+cudaGetErrorString(status));
}
void fast_blas(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("M31 fast FFN cuBLAS status "+std::to_string(int(status)));
}
__device__ float fast_ue4m3(uint8_t x) {
    if(x==0 || x==127) return 0;
    const int e=(x>>3)&15,m=x&7;
    return e ? ldexpf(1.f+float(m)/8.f,e-8) : ldexpf(float(m),-10);
}
template<int Type>
__global__ void fast_dequant_weights(const uint8_t * packed,half * high,half * low,size_t elements,int columns) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=elements)return;
    float value=0;
    if constexpr(Type==GGML_TYPE_IQ3_XXS) {
        const auto & x=reinterpret_cast<const block_iq3_xxs *>(packed)[index/256];
        const int pos=int(index%256),sub=pos/32,p=pos%32;
        const auto * gs=x.qs+64+sub*4;
        const uint32_t bits=uint32_t(gs[0])|(uint32_t(gs[1])<<8)|(uint32_t(gs[2])<<16)|(uint32_t(gs[3])<<24);
        const uint8_t signs=ksigns_iq2xs[(bits>>(7*(p/8)))&127];
        const uint32_t grid=iq3xxs_grid[x.qs[sub*8+p/4]];
        const int q=int((grid>>(8*(p%4)))&255)*((signs&(1<<(p%8)))?-1:1);
        value=__fmul_rn(__fmul_rn(__half2float(x.d),0.25f),float((2*int(bits>>28)+1)*q));
    } else if constexpr(Type==GGML_TYPE_IQ4_XS) {
        const auto & x=reinterpret_cast<const block_iq4_xs *>(packed)[index/256];
        const int pos=int(index%256),sub=pos/32,p=pos%32;
        const int scale=(((x.scales_l[sub/2]>>(4*(sub%2)))&15)|(((x.scales_h>>(2*sub))&3)<<4))-32;
        const uint8_t q=x.qs[sub*16+p%16];
        value=__fmul_rn(__half2float(x.d),float(scale*int(kvalues_iq4nl[p<16?q&15:q>>4])));
    } else {
        const auto & x=reinterpret_cast<const block_nvfp4 *>(packed)[index/64];
        const int pos=int(index%64),sub=pos/16,p=pos%16;
        const uint8_t q=x.qs[sub*8+p%8];
        value=__fmul_rn(fast_ue4m3(x.d[sub]),float(kvalues_fp4[p<8?q&15:q>>4]));
    }
    const half h=__float2half_rn(value);
    high[index]=h;
    if(low)low[index]=__float2half_rn(__fsub_rn(value,__half2float(h)));
}
__global__ void fast_gather_float(const float * input,float * output,int cols,size_t count,const int32_t *routes,int used) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i]=input[size_t(routes[i/cols]/used)*cols+i%cols];
}
__global__ void fast_oai(float * gate,const float * up,size_t count,float gs,float us,bool precise_exp,bool libm_exp) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float gv=__fmul_rn(gate[i],gs),uv=__fmul_rn(up[i],us);
    const float a=gv>7.f?7.f:gv;
    const float b=uv < -7.f ? -7.f : (uv>7.f?7.f:uv);
    const float exponent=__fmul_rn(-1.702f,a);
    const float exponential=libm_exp?gpu_libm_expf(exponent):precise_exp?float(exp(double(exponent))):expf(exponent);
    gate[i]=__fmul_rn(__fdiv_rn(a,__fadd_rn(1.f,exponential)),__fadd_rn(b,1.f));
}
__global__ void fast_scatter(const float * local,float * routed,const int32_t * routes,size_t count,int rows,float scale) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)routed[size_t(routes[i/rows])*rows+i%rows]=__fmul_rn(local[i],scale);
}
// One stream orders all experts/parts; prepare rejects duplicate expert IDs in
// one token. Every thread in this kernel owns a unique (token,row), so plain
// F32 addition is race-free. The summation order becomes expert-ID order.
__global__ void fast_scatter_reduce(const float * local,float * output,const float * weights,
        const int32_t * routes,size_t count,int rows,int used,float scale) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const int32_t route=routes[i/rows];
    const size_t output_index=size_t(route/used)*rows+i%rows;
    const float term=__fmul_rn(__fmul_rn(local[i],scale),weights[route]);
    output[output_index]=__fadd_rn(output[output_index],term);
}
// Every route in this invocation has the same rank. Global rank phases are
// serialized on the private stream. Assign rank0, then add ranks1..used-1 in
// the exact original CPU F32 order without storing the full routed tensor.
__global__ void fast_scatter_rank_phase(const float *local,float *output,const float *weights,
        const int32_t *routes,size_t count,int rows,int used,float scale,bool first) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const int32_t route=routes[i/rows];
    const size_t oi=size_t(route/used)*rows+i%rows;
    const float term=__fmul_rn(__fmul_rn(local[i],scale),weights[route]);
    output[oi]=first?term:__fadd_rn(output[oi],term);
}
__global__ void fast_hidden_sum(const float *moe,const float *shared,const float *residual,
                                float *destination,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)destination[i]=__fadd_rn(__fadd_rn(moe[i],shared[i]),residual[i]);
}
__global__ void fast_reduce(const float * routed,const float * weights,float * output,int rows,int used,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const size_t token=i/rows,row=i%rows;
    float sum=__fmul_rn(routed[(token*used)*rows+row],weights[token*used]);
    for(int rank=1;rank<used;++rank)
        sum=__fadd_rn(sum,__fmul_rn(routed[(token*used+rank)*rows+row],weights[token*used+rank]));
    output[i]=sum;
}
}
struct GpuFastFfn::Impl {
    struct Buffer {
        void * pointer=nullptr;
        size_t capacity=0;
        void reserve(size_t bytes) {
            if(capacity>=bytes)return;
            fast_cuda(cudaFree(pointer));pointer=nullptr;capacity=0;
            fast_cuda(cudaMalloc(&pointer,bytes));capacity=bytes;
        }
        template<class T>T * as(){return static_cast<T *>(pointer);}
        ~Buffer(){cudaFree(pointer);}
    };
    Buffer input,route,weights,routed,output,wh,wl,xh,xl,hh,hl,gate,up,down,exact_input;
#ifdef M31_CPU_QUANTS_CUDA
    std::unique_ptr<GpuDeviceDots> device_dots;
#endif
    cudaStream_t stream=nullptr;
    const float *active_input=nullptr;
    cublasHandle_t blas=nullptr;
    bool active=false,compensated=true,weight_has_low=true,fused_reduce=false,rank_phase=false,cpu_dots=false,tiled_enabled=true,tiled_forced=false,precise_exp=false,libm_exp=true;
    int expert_batch=1024;
    int gt=0,ut=0,dt=0,columns=0,hidden=0,used=0,n=0;
    std::vector<int32_t> order,histogram,offset,assigned_slots,rank_histogram,rank_offset;
    std::vector<uint8_t> submitted;
    size_t calls=0,expert_calls=0,host_upload_bytes=0,host_read_bytes=0,device_input_calls=0,device_output_calls=0;
    std::string debug_directory;
    std::vector<float> debug_gate,debug_up,debug_activation,debug_down;
    double wall_seconds=0;
    std::chrono::steady_clock::time_point start;
    Impl() {
        const char *debug=std::getenv("M31_GPU_FAST_FFN_DEBUG_DIR");if(debug&&*debug)debug_directory=debug;
        const char * env=std::getenv("M31_GPU_FAST_FFN_COMPENSATED");
        compensated=!env || std::atoi(env)!=0;
        const char *reduce=std::getenv("M31_GPU_FAST_FFN_FUSED_REDUCE");
        fused_reduce=reduce&&std::atoi(reduce)!=0;
        const char *phases=std::getenv("M31_GPU_FAST_FFN_RANK_PHASES");rank_phase=phases&&std::atoi(phases)!=0;
        const char *precise=std::getenv("M31_GPU_FAST_FFN_EXP_DOUBLE");precise_exp=precise&&std::atoi(precise)!=0;
        const char *libm=std::getenv("M31_GPU_FAST_FFN_LIBM_EXPF");libm_exp=!libm||std::atoi(libm)!=0;
        const char *batch=std::getenv("M31_GPU_FAST_FFN_EXPERT_BATCH");
        if(batch)expert_batch=std::stoi(batch);
        fast_require(expert_batch>=32&&expert_batch<=8192&&expert_batch%32==0,"EXPERT_BATCH must be 32..8192 and divisible by32");
        const char *dot_mode=std::getenv("M31_GPU_FAST_FFN_DOTS");
        fast_require(!dot_mode||std::string(dot_mode)=="cublas"||std::string(dot_mode)=="cpu","DOTS must be cublas or cpu");
        #ifdef M31_CPU_QUANTS_CUDA
        cpu_dots=!dot_mode||std::string(dot_mode)=="cpu";
#else
        fast_require(dot_mode&&std::string(dot_mode)=="cublas","fast resident default requires STRATA_M31_CPU_QUANTS_CUDA; experimental cublas requires explicit DOTS=cublas");
        cpu_dots=false;
#endif
        const char *tile=std::getenv("GGML_CPU_TILED_MM"),*force=std::getenv("GGML_CPU_TILED_MM_FORCE");
        tiled_enabled=!tile||std::atoi(tile)!=0;tiled_forced=force&&std::atoi(force)==1;
#ifndef M31_CPU_QUANTS_CUDA
        fast_require(!cpu_dots,"cpu resident dots require STRATA_M31_CPU_QUANTS_CUDA build");
#endif
        fast_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        try {
            #ifdef M31_CPU_QUANTS_CUDA
            if(cpu_dots)device_dots=std::make_unique<GpuDeviceDots>(stream);
            else
#endif
            {
            fast_blas(cublasCreate(&blas));
            fast_blas(cublasSetStream(blas,stream));
            fast_blas(cublasSetMathMode(blas,cublasMath_t(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
            }
        }catch(...){if(blas)cublasDestroy(blas);cudaStreamDestroy(stream);throw;}
        std::fprintf(stderr,"M31 fast FFN: GPU Q8_K/Q8_0 + %s, resident OAI/down/reduction\n",cpu_dots?"original CPU-compatible IMMA/short dots":compensated?"compensated half Tensor Core GEMM":"single half Tensor Core GEMM");
    }
    void drain()noexcept {if(active && stream)cudaStreamSynchronize(stream);active=false;active_input=nullptr;}
    ~Impl(){drain();
#ifdef M31_CPU_QUANTS_CUDA
        device_dots.reset();
#endif
        if(blas)cublasDestroy(blas);if(stream)cudaStreamDestroy(stream);
    }
    size_t scratch()const {
        size_t bytes=0;
        for(const auto *b:{&input,&route,&weights,&routed,&output,&wh,&wl,&xh,&xl,&hh,&hl,&gate,&up,&down,&exact_input})bytes+=b->capacity;
#ifdef M31_CPU_QUANTS_CUDA
        if(device_dots)bytes+=device_dots->scratch_bytes();
#endif
        return bytes;
    }
    void capture(std::vector<float> &destination,const float *local,int rows,int count,int begin,float scale=1.f) {
        if(debug_directory.empty()||calls)return;
        std::vector<float> host(size_t(rows)*count);
        fast_cuda(cudaMemcpyAsync(host.data(),local,host.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
        fast_cuda(cudaStreamSynchronize(stream));
        for(int i=0;i<count;++i)for(int row=0;row<rows;++row)
            destination[size_t(order[size_t(begin)+i])*rows+row]=host[size_t(i)*rows+row]*scale;
    }
    void save_debug(const std::vector<float> &result) {
        if(debug_directory.empty()||calls)return;
        std::filesystem::create_directories(debug_directory);
        const auto write=[&](const char *name,const auto& values) {
            std::ofstream out(std::filesystem::path(debug_directory)/(std::string("call0_")+name+".bin"),std::ios::binary);
            out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));fast_require(bool(out),"debug write failed");
        };
        write("gate",debug_gate);write("up",debug_up);write("activation",debug_activation);write("down",debug_down);write("result",result);write("scheduled_routes",order);
    }
    void wait_event(ggml_backend_event_t ready) {
        if(!ready)return;
        int device=-1;fast_cuda(cudaGetDevice(&device));
        const auto registry=ggml_backend_dev_backend_reg(ready->device);
        fast_require(registry&&std::string(ggml_backend_reg_name(registry))=="CUDA"&&std::string(ggml_backend_dev_name(ready->device))=="CUDA"+std::to_string(device)&&ready->context,"event belongs to another backend/device");
        fast_cuda(cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(ready->context),0));
    }
    void dequant(int type,const void * packed,int cols,int rows) {
        const size_t count=size_t(cols)*rows;
        const unsigned blocks=unsigned((count+255)/256);
        weight_has_low=type!=GGML_TYPE_NVFP4;
        half *low=compensated&&weight_has_low?wl.as<half>():nullptr;
        switch(type) {
            case GGML_TYPE_IQ3_XXS:fast_dequant_weights<GGML_TYPE_IQ3_XXS><<<blocks,256,0,stream>>>(static_cast<const uint8_t *>(packed),wh.as<half>(),low,count,cols);break;
            case GGML_TYPE_IQ4_XS:fast_dequant_weights<GGML_TYPE_IQ4_XS><<<blocks,256,0,stream>>>(static_cast<const uint8_t *>(packed),wh.as<half>(),low,count,cols);break;
            case GGML_TYPE_NVFP4:fast_dequant_weights<GGML_TYPE_NVFP4><<<blocks,256,0,stream>>>(static_cast<const uint8_t *>(packed),wh.as<half>(),low,count,cols);break;
            default:throw std::runtime_error("unsupported fast FFN quant");
        }
        fast_cuda(cudaGetLastError());
    }
    void gemm(int cols,int rows,int count,half *high,half *low,float *out) {
        const float alpha=1.f,zero=0.f,one=1.f;
        const auto multiply=[&](half *w,half *x,const float *beta) {
            fast_blas(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,count,cols,&alpha,
                w,CUDA_R_16F,cols,x,CUDA_R_16F,cols,beta,out,CUDA_R_32F,rows,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        multiply(wh.as<half>(),high,&zero);
        if(compensated) {if(weight_has_low)multiply(wl.as<half>(),high,&one);multiply(wh.as<half>(),low,&one);}
    }
};
GpuFastFfn::GpuFastFfn():p_(std::make_unique<Impl>()){}
GpuFastFfn::~GpuFastFfn()=default;
bool GpuFastFfn::supported(int gate,int up,int down) {
    const auto iq=[](int type){return type==GGML_TYPE_IQ3_XXS||type==GGML_TYPE_IQ4_XS;};
    return iq(gate)&&iq(up)&&(iq(down)||down==GGML_TYPE_NVFP4);
}
size_t GpuFastFfn::scratch_bytes()const{return p_->scratch();}
int GpuFastFfn::expert_batch_rows()const{return p_->expert_batch;}
bool GpuFastFfn::fused_reduction()const{return p_->fused_reduce;}
bool GpuFastFfn::rank_phases()const{return p_->rank_phase;}
void GpuFastFfn::abort()noexcept{p_->drain();}
void GpuFastFfn::prepare(int gt,int ut,int dt,int cols,int hidden,int used,int n,
        const std::vector<float>& input,const std::vector<int32_t>& ids,const std::vector<float>& weights) {
    fast_require(cols>0&&n>0&&input.size()==size_t(cols)*size_t(n),"input geometry mismatch");
    prepare_impl(gt,ut,dt,cols,hidden,used,n,input.data(),nullptr,ids,weights,nullptr);
}
void GpuFastFfn::prepare_device(int gt,int ut,int dt,int cols,int hidden,int used,int n,
        const float *input,const std::vector<int32_t>& ids,const std::vector<float>& weights,
        ggml_backend_event_t ready) {
    fast_require(input!=nullptr,"missing device input");
    prepare_impl(gt,ut,dt,cols,hidden,used,n,nullptr,input,ids,weights,ready);
}
void GpuFastFfn::prepare_impl(int gt,int ut,int dt,int cols,int hidden,int used,int n,
        const float *host_input,const float *device_input,const std::vector<int32_t>& ids,
        const std::vector<float>& weights,ggml_backend_event_t input_ready) {
    fast_require(!p_->active,"previous batch needs read/abort");
    fast_require(supported(gt,ut,dt)&&cols>0&&hidden>0&&cols%256==0&&hidden%256==0&&used>0&&used<=32&&n>0&&n<=65535,"unsupported shape/type");
    fast_require((host_input||device_input)&&ids.size()==size_t(used)*n&&weights.size()==ids.size(),"input geometry mismatch");
    const size_t input_count=size_t(cols)*n;
    int32_t maximum=-1;
    for(size_t i=0;i<ids.size();++i){
        fast_require(ids[i]>=0&&ids[i]<1048576&&std::isfinite(weights[i]),"invalid route");
        for(size_t prior=i-i%size_t(used);prior<i;++prior)fast_require(ids[prior]!=ids[i],"duplicate expert within a token");
        maximum=std::max(maximum,ids[i]);
    }
    const char *validate=std::getenv("M31_GPU_FAST_FFN_VALIDATE_INPUT");
    if(host_input&&validate&&std::atoi(validate))for(size_t i=0;i<input_count;++i)fast_require(std::isfinite(host_input[i]),"nonfinite input");
    const auto work=prefill_route_worklist(ids,size_t(maximum)+1,used);
    p_->order=work.route;p_->histogram=work.histogram;p_->submitted.assign(work.histogram.size(),0);p_->assigned_slots.clear();
    p_->offset.resize(work.histogram.size());
    int32_t begin=0;int largest=0;
    for(size_t i=0;i<work.histogram.size();++i){p_->offset[i]=begin;begin+=work.histogram[i];largest=std::max(largest,work.histogram[i]);}
    if(p_->rank_phase) {
        const size_t experts=p_->histogram.size();
        p_->rank_histogram.assign(size_t(used)*experts,0);p_->rank_offset.resize(size_t(used)*experts);
        for(size_t r=0;r<ids.size();++r)++p_->rank_histogram[(r%size_t(used))*experts+size_t(ids[r])];
        std::vector<int32_t> cursor(p_->rank_histogram.size());int32_t rank_begin=0;
        for(size_t i=0;i<cursor.size();++i){p_->rank_offset[i]=cursor[i]=rank_begin;rank_begin+=p_->rank_histogram[i];}
        p_->order.resize(ids.size());
        for(size_t r=0;r<ids.size();++r){const size_t bucket=(r%size_t(used))*experts+size_t(ids[r]);p_->order[size_t(cursor[bucket]++)]=int32_t(r);}
    }
    largest=std::min(p_->expert_batch,std::min(n,((largest+255)/256)*256));
    p_->gt=gt;p_->ut=ut;p_->dt=dt;p_->columns=cols;p_->hidden=hidden;p_->used=used;p_->n=n;
    if(!p_->debug_directory.empty()&&!p_->calls) {
        fast_require(n<=512,"debug stage capture limited to512tokens");
        p_->debug_gate.resize(size_t(hidden)*ids.size());p_->debug_up.resize(size_t(hidden)*ids.size());
        p_->debug_activation.resize(size_t(hidden)*ids.size());p_->debug_down.resize(size_t(cols)*ids.size());
    }
    const size_t weight_bytes=size_t(cols)*hidden*sizeof(half);
    if(host_input)p_->input.reserve(input_count*sizeof(float));
    p_->route.reserve(ids.size()*sizeof(int32_t));p_->weights.reserve(weights.size()*sizeof(float));
    if(!p_->fused_reduce&&!p_->rank_phase)p_->routed.reserve(size_t(cols)*ids.size()*sizeof(float));
    p_->output.reserve(input_count*sizeof(float));
    if(p_->cpu_dots) {
        p_->exact_input.reserve(size_t(cols)*largest*sizeof(float));
#ifdef M31_CPU_QUANTS_CUDA
        p_->device_dots->reserve(cols,hidden,largest);
#endif
    }
    else {
        p_->wh.reserve(weight_bytes);if(p_->compensated)p_->wl.reserve(weight_bytes);
        p_->xh.reserve(size_t(cols)*largest*sizeof(half));if(p_->compensated)p_->xl.reserve(size_t(cols)*largest*sizeof(half));
        p_->hh.reserve(size_t(hidden)*largest*sizeof(half));if(p_->compensated)p_->hl.reserve(size_t(hidden)*largest*sizeof(half));
    }
    p_->gate.reserve(size_t(hidden)*largest*sizeof(float));p_->up.reserve(size_t(hidden)*largest*sizeof(float));p_->down.reserve(size_t(cols)*largest*sizeof(float));
    p_->active=true;p_->start=std::chrono::steady_clock::now();
    if(p_->fused_reduce&&!p_->rank_phase)fast_cuda(cudaMemsetAsync(p_->output.pointer,0,input_count*sizeof(float),p_->stream));
    if(host_input) {
        fast_cuda(cudaMemcpyAsync(p_->input.pointer,host_input,input_count*sizeof(float),cudaMemcpyHostToDevice,p_->stream));
        p_->active_input=p_->input.as<float>();p_->host_upload_bytes+=input_count*sizeof(float);
    } else {
        p_->wait_event(input_ready);p_->active_input=device_input;++p_->device_input_calls;
    }
    fast_cuda(cudaMemcpyAsync(p_->route.pointer,p_->order.data(),p_->order.size()*sizeof(int32_t),cudaMemcpyHostToDevice,p_->stream));
    fast_cuda(cudaMemcpyAsync(p_->weights.pointer,weights.data(),weights.size()*sizeof(float),cudaMemcpyHostToDevice,p_->stream));
    p_->host_upload_bytes+=p_->order.size()*sizeof(int32_t)+weights.size()*sizeof(float);
}
void GpuFastFfn::enqueue(const void *gp,const void *up,const void *dp,size_t stride,
        const std::vector<int32_t>& logical,const std::vector<int32_t>& slots,
        const std::vector<float>& gs,const std::vector<float>& us,const std::vector<float>& ds,ggml_backend_event_t ready) {
    fast_require(p_->active&&gp&&up&&dp&&!logical.empty()&&logical.size()==slots.size(),"invalid enqueue");
    fast_require(stride>=std::max({ggml_row_size(ggml_type(p_->gt),p_->columns)*size_t(p_->hidden),
        ggml_row_size(ggml_type(p_->ut),p_->columns)*size_t(p_->hidden),ggml_row_size(ggml_type(p_->dt),p_->hidden)*size_t(p_->columns)}),"short weight stride");
    for(size_t i=0;i<logical.size();++i) {
        const int id=logical[i];
        fast_require(id>=0&&size_t(id)<p_->histogram.size()&&p_->histogram[id]>0&&!p_->submitted[id]&&slots[i]>=0,"invalid/duplicate expert");
        fast_require((gs.empty()||gs.size()>size_t(id))&&(us.empty()||us.size()>size_t(id))&&(ds.empty()||ds.size()>size_t(id)),"short scales");
        for(const auto *values:{&gs,&us,&ds})fast_require(values->empty()||std::isfinite((*values)[id]),"nonfinite role scale");
        for(size_t j=0;j<i;++j)fast_require(logical[j]!=id&&slots[j]!=slots[i],"duplicate logical expert/aliased slots");
        for(int previous:p_->assigned_slots)fast_require(previous!=slots[i],"physical slot reused across groups");
    }
    if(p_->rank_phase) {
        const size_t active=std::count_if(p_->histogram.begin(),p_->histogram.end(),[](int32_t count){return count>0;});
        fast_require(p_->assigned_slots.empty()&&logical.size()==active,"RANK_PHASES requires one enqueue with all active experts");
    }
    p_->wait_event(ready);
    const int phases=p_->rank_phase?p_->used:1;
    for(int phase=0;phase<phases;++phase)for(size_t i=0;i<logical.size();++i) {
        const int id=logical[i],total=p_->histogram[id],slot=slots[i];
        const size_t bucket=size_t(phase)*p_->histogram.size()+size_t(id);
        const int begin=p_->rank_phase?p_->rank_offset[bucket]:p_->offset[id];
        const int scheduled=p_->rank_phase?p_->rank_histogram[bucket]:total;
        for(int part=0;part<scheduled;part+=p_->expert_batch) {
        const int count=std::min(p_->expert_batch,scheduled-part);
        const auto *routes=p_->route.as<int32_t>()+begin+part;
        #ifdef M31_CPU_QUANTS_CUDA
        const bool original_tiled=p_->tiled_enabled&&(p_->tiled_forced||total>=8);
        if(p_->device_dots) {
            const size_t input_count=size_t(p_->columns)*count;
            fast_gather_float<<<unsigned((input_count+255)/256),256,0,p_->stream>>>(p_->active_input,p_->exact_input.as<float>(),p_->columns,input_count,routes,p_->used);
            fast_cuda(cudaGetLastError());
            p_->device_dots->multiply(p_->gt,static_cast<const uint8_t*>(gp)+size_t(slot)*stride,p_->columns,p_->hidden,count,p_->exact_input.as<float>(),p_->gate.as<float>(),original_tiled);
            p_->device_dots->multiply(p_->ut,static_cast<const uint8_t*>(up)+size_t(slot)*stride,p_->columns,p_->hidden,count,p_->exact_input.as<float>(),p_->up.as<float>(),original_tiled);
        } else
#endif
        {
        fast_quant_gather_half(p_->active_input,p_->xh.as<half>(),p_->columns,count,routes,p_->used,p_->stream,false,p_->compensated?p_->xl.as<half>():nullptr);
        p_->dequant(p_->gt,static_cast<const uint8_t*>(gp)+size_t(slot)*stride,p_->columns,p_->hidden);
        p_->gemm(p_->columns,p_->hidden,count,p_->xh.as<half>(),p_->compensated?p_->xl.as<half>():nullptr,p_->gate.as<float>());
        p_->dequant(p_->ut,static_cast<const uint8_t*>(up)+size_t(slot)*stride,p_->columns,p_->hidden);
        p_->gemm(p_->columns,p_->hidden,count,p_->xh.as<half>(),p_->compensated?p_->xl.as<half>():nullptr,p_->up.as<float>());
        }
        p_->capture(p_->debug_gate,p_->gate.as<float>(),p_->hidden,count,begin+part,gs.empty()?1.f:gs[id]);
        p_->capture(p_->debug_up,p_->up.as<float>(),p_->hidden,count,begin+part,us.empty()?1.f:us[id]);
        const size_t hidden_count=size_t(p_->hidden)*count;
        fast_oai<<<unsigned((hidden_count+255)/256),256,0,p_->stream>>>(p_->gate.as<float>(),p_->up.as<float>(),hidden_count,gs.empty()?1.f:gs[id],us.empty()?1.f:us[id],p_->precise_exp,p_->libm_exp);
        fast_cuda(cudaGetLastError());
        p_->capture(p_->debug_activation,p_->gate.as<float>(),p_->hidden,count,begin+part);
        #ifdef M31_CPU_QUANTS_CUDA
        if(p_->device_dots)
            p_->device_dots->multiply(p_->dt,static_cast<const uint8_t*>(dp)+size_t(slot)*stride,p_->hidden,p_->columns,count,p_->gate.as<float>(),p_->down.as<float>(),original_tiled);
        else
#endif
        {
        fast_quant_gather_half(p_->gate.as<float>(),p_->hh.as<half>(),p_->hidden,count,nullptr,1,p_->stream,p_->dt==GGML_TYPE_NVFP4,p_->compensated?p_->hl.as<half>():nullptr);
        p_->dequant(p_->dt,static_cast<const uint8_t*>(dp)+size_t(slot)*stride,p_->hidden,p_->columns);
        p_->gemm(p_->hidden,p_->columns,count,p_->hh.as<half>(),p_->compensated?p_->hl.as<half>():nullptr,p_->down.as<float>());
        }
        p_->capture(p_->debug_down,p_->down.as<float>(),p_->columns,count,begin+part,ds.empty()?1.f:ds[id]);
        const size_t out_count=size_t(p_->columns)*count;
        if(p_->rank_phase)
            fast_scatter_rank_phase<<<unsigned((out_count+255)/256),256,0,p_->stream>>>(p_->down.as<float>(),p_->output.as<float>(),p_->weights.as<float>(),routes,out_count,p_->columns,p_->used,ds.empty()?1.f:ds[id],phase==0);
        else if(p_->fused_reduce)
            fast_scatter_reduce<<<unsigned((out_count+255)/256),256,0,p_->stream>>>(p_->down.as<float>(),p_->output.as<float>(),p_->weights.as<float>(),routes,out_count,p_->columns,p_->used,ds.empty()?1.f:ds[id]);
        else
            fast_scatter<<<unsigned((out_count+255)/256),256,0,p_->stream>>>(p_->down.as<float>(),p_->routed.as<float>(),routes,out_count,p_->columns,ds.empty()?1.f:ds[id]);
        fast_cuda(cudaGetLastError());
        }
        if(scheduled)++p_->expert_calls;
    }
    for(size_t i=0;i<logical.size();++i){p_->submitted[logical[i]]=1;p_->assigned_slots.push_back(slots[i]);}
}
void GpuFastFfn::read_into(std::vector<float>& output) {
    finish_impl(&output,nullptr,nullptr,nullptr);
}
void GpuFastFfn::finish_device(float *destination,const float *shared,const float *residual) {
    fast_require(destination&&((shared&&residual)||(!shared&&!residual)),"invalid device destination/addends");
    finish_impl(nullptr,destination,shared,residual);
}
void GpuFastFfn::finish_impl(std::vector<float> *host_output,float *destination,const float *shared,const float *residual) {
    fast_require(p_->active,"read outside batch");
    for(size_t id=0;id<p_->histogram.size();++id)fast_require(!p_->histogram[id]||p_->submitted[id],"not all experts submitted");
    const size_t count=size_t(p_->columns)*p_->n;
    if(!p_->fused_reduce&&!p_->rank_phase)fast_reduce<<<unsigned((count+255)/256),256,0,p_->stream>>>(p_->routed.as<float>(),p_->weights.as<float>(),p_->output.as<float>(),p_->columns,p_->used,count);
    fast_cuda(cudaGetLastError());
    std::vector<float> debug_result;
    if(host_output) {
        host_output->resize(count);
        fast_cuda(cudaMemcpyAsync(host_output->data(),p_->output.pointer,count*sizeof(float),cudaMemcpyDeviceToHost,p_->stream));
        p_->host_read_bytes+=count*sizeof(float);
    } else {
        if(!p_->debug_directory.empty()&&!p_->calls) {
            debug_result.resize(count);
            fast_cuda(cudaMemcpyAsync(debug_result.data(),p_->output.pointer,count*sizeof(float),cudaMemcpyDeviceToHost,p_->stream));
            p_->host_read_bytes+=count*sizeof(float);
        }
        if(shared) {
            fast_hidden_sum<<<unsigned((count+255)/256),256,0,p_->stream>>>(p_->output.as<float>(),shared,residual,destination,count);
            fast_cuda(cudaGetLastError());
        } else if(destination!=p_->output.pointer)
            fast_cuda(cudaMemcpyAsync(destination,p_->output.pointer,count*sizeof(float),cudaMemcpyDeviceToDevice,p_->stream));
        ++p_->device_output_calls;
    }
    fast_cuda(cudaStreamSynchronize(p_->stream));p_->active=false;p_->active_input=nullptr;
    if(host_output)p_->save_debug(*host_output);else p_->save_debug(debug_result);
    ++p_->calls;
    p_->wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-p_->start).count();
}
void GpuFastFfn::print_profile(const char *label)const {
    std::fprintf(stderr,"M31 fast FFN profile label=%s calls=%zu experts=%zu compensated=%d expert_batch=%d fused_reduce=%d cpu_dots=%d exp_double=%d libm_expf=%d rank_phases=%d device_inputs=%zu device_outputs=%zu wall_ms=%.3f H2D_bytes=%zu D2H_bytes=%zu scratch_bytes=%zu\n",label?label:"runtime",p_->calls,p_->expert_calls,int(p_->compensated),p_->expert_batch,int(p_->fused_reduce),int(p_->cpu_dots),int(p_->precise_exp),int(p_->libm_exp),int(p_->rank_phase),p_->device_input_calls,p_->device_output_calls,p_->wall_seconds*1000,p_->host_upload_bytes,p_->host_read_bytes,p_->scratch());
}
}
