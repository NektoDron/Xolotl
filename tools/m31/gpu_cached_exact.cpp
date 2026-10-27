#include "gpu_cached_exact.hpp"
#if defined(M31_ENGINE_CUDA) && defined(M31_CPU_QUANTS_CUDA)
#include "gpu_resident_dot.hpp"
#include "ggml.h"
#include "cpu_threadpool_config.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cerrno>
#include <cstdio>
#if defined(__linux__)
#include <sched.h>
#endif
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace strata::m31 {
namespace {
void exact_check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(std::string("M31 cached exact: ")+cudaGetErrorString(rc));}
}
struct GpuCachedExact::Impl {
    using Clock=std::chrono::steady_clock;
    int embd,ff,maximum_used,maximum_tokens,token_count=0,device=0;
    cudaStream_t stream=nullptr;
    std::unique_ptr<GpuDeviceDots> dots;
    float *input=nullptr,*gate_up=nullptr,*activation=nullptr,*down=nullptr;
    float *host_input=nullptr,*host_gate_up=nullptr,*host_activation=nullptr,*host_down=nullptr;
    std::mutex mutex;std::condition_variable wake,completed;std::thread worker;
    bool active=false,pending=false,stopping=false,done=false;
    bool grouped_mode=false,grouped_result=false,joined_gu_mode=false,iq3_imma_mode=false;
    std::exception_ptr failure;
    std::array<int,3> types{};std::array<const void*,3> pools{};
    size_t stride=0;std::vector<int32_t> slots,tokens;std::vector<float> weights;
    std::array<std::vector<float>,3> scales;
    fenv_t environment{};
#if defined(__SSE__)
    unsigned mxcsr=0;
#endif
#if defined(__linux__)
    cpu_set_t worker_affinity{};
    bool explicit_affinity=false,affinity_applied=false;
#endif
    Stats stat;
    static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
    static void allocate(float*&p,size_t count){exact_check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(float)));}
    static void pin(float*&p,size_t count){exact_check(cudaHostAlloc(reinterpret_cast<void**>(&p),count*sizeof(float),cudaHostAllocDefault));}
    Impl(int e,int f,int used,int n):embd(e),ff(f),maximum_used(used),maximum_tokens(n){
        const char*session=std::getenv("M31_CARD_SESSION");
        if(!session||std::string(session)!="1"||e<=0||f<=0||used<=0||used>128||n<=0||n>8||e%256||f%256)
            throw std::invalid_argument("M31 cached exact: authorized session and aligned geometry required");
        try {
            const char*group=std::getenv("M31_GPU_CACHED_GROUPED");grouped_mode=group&&std::string(group)=="1";
            const char*joined=std::getenv("M31_GPU_CACHED_GATE_UP");joined_gu_mode=joined&&std::string(joined)=="1";
            const char*imma=std::getenv("M31_GPU_CACHED_IQ3_IMMA");iq3_imma_mode=imma&&std::string(imma)=="1";
            if(grouped_mode&&e>2*f)throw std::invalid_argument("M31 grouped cached exact requires embd<=2ff for scratch reuse");
#if defined(__linux__)
            if(std::getenv("M31_CPU_AFFINITY")){
                auto params=ggml_threadpool_params_default(1);apply_cpu_pool_environment(params);
                CPU_ZERO(&worker_affinity);size_t count=0;
                for(unsigned cpu=0;cpu<GGML_MAX_N_THREADS;++cpu)if(params.cpumask[cpu]){CPU_SET(cpu,&worker_affinity);++count;}
                explicit_affinity=true;
                std::fprintf(stderr,"M31 cached exact worker: explicit_affinity_cpus=%zu (restored in worker after main-thread pinning)\n",count);
            }
#endif
            exact_check(cudaGetDevice(&device));exact_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            allocate(input,size_t(e)*n);allocate(gate_up,2*size_t(f)*used*n);allocate(activation,size_t(f)*used*n);allocate(down,size_t(e)*used*n);
            pin(host_input,size_t(e)*n);pin(host_gate_up,2*size_t(f)*used*n);pin(host_activation,size_t(f)*used*n);pin(host_down,size_t(e)*used*n);
            dots=std::make_unique<GpuDeviceDots>(stream);dots->reserve(e,f,grouped_mode?n:1);
            if(iq3_imma_mode)dots->reserve_iq3_n1(std::max(e,f),std::max(e,f<=std::numeric_limits<int>::max()/2?2*f:f));
            stat.device_bytes=(size_t(e)+3*size_t(f)*used+size_t(e)*used)*size_t(n)*sizeof(float)+dots->scratch_bytes();
            stat.pinned_bytes=(size_t(e)+3*size_t(f)*used+size_t(e)*used)*size_t(n)*sizeof(float);
            worker=std::thread([this]{loop();});
        } catch (...) {release();throw;}
    }
    ~Impl(){
        drain();{std::lock_guard lock(mutex);stopping=true;}wake.notify_one();if(worker.joinable())worker.join();release();
    }
    void release() noexcept {
        if(stream)cudaStreamSynchronize(stream);
        dots.reset();
        for(auto**p:{&input,&gate_up,&activation,&down})if(*p){cudaFree(*p);*p=nullptr;}
        for(auto**p:{&host_input,&host_gate_up,&host_activation,&host_down})if(*p){cudaFreeHost(*p);*p=nullptr;}
        if(stream){cudaStreamDestroy(stream);stream=nullptr;}
    }
    void drain() noexcept {
        std::unique_lock lock(mutex);if(active){completed.wait(lock,[this]{return done;});active=false;}
    }
    void loop() noexcept {
        for(;;){
            {std::unique_lock lock(mutex);wake.wait(lock,[this]{return pending||stopping;});if(stopping)return;pending=false;}
            try {
#if defined(__linux__)
                if(explicit_affinity&&!affinity_applied){
                    if(sched_setaffinity(0,sizeof(worker_affinity),&worker_affinity))
                        throw std::runtime_error(std::string("M31 cached exact worker affinity: ")+std::strerror(errno));
                    cpu_set_t actual{};
                    if(sched_getaffinity(0,sizeof(actual),&actual))
                        throw std::runtime_error("M31 cached exact worker: cannot read restored affinity");
                    std::fprintf(stderr,"M31 cached exact worker: restored_affinity_cpus=%d\n",CPU_COUNT(&actual));
                    affinity_applied=true;
                }
#endif
                exact_check(cudaSetDevice(device));run();
            }catch(...){if(stream)cudaStreamSynchronize(stream);failure=std::current_exception();}
            {std::lock_guard lock(mutex);done=true;}completed.notify_one();
        }
    }
    void multiply(int type,const void *weights,int columns,int rows,int count,
                  const float *input,float *output,bool original_tiled){
        if(iq3_imma_mode&&type==GGML_TYPE_IQ3_XXS&&count==1&&!original_tiled){
            dots->multiply_iq3_n1(weights,columns,rows,input,output);
            ++stat.iq3_imma_calls;stat.iq3_imma_rows+=uint64_t(rows);
        }else dots->multiply(type,weights,columns,rows,count,input,output,original_tiled);
    }
    void run_baseline(){
        if(can_join_gate_up()){
            std::vector<size_t>order(slots.size());std::iota(order.begin(),order.end(),size_t(0));
            run_joined_gate_up(order,false);return;
        }
        const auto start=Clock::now();const size_t hit=slots.size();
        if(fesetenv(&environment))throw std::runtime_error("M31 cached exact: cannot apply FP environment");
#if defined(__SSE__)
        _mm_setcsr(mxcsr);
#endif
        const auto gate_start=Clock::now();
        exact_check(cudaMemcpyAsync(input,host_input,size_t(embd)*size_t(token_count)*sizeof(float),cudaMemcpyHostToDevice,stream));
        for(size_t rank=0;rank<hit;++rank){
            const size_t offset=size_t(slots[rank])*stride;
            multiply(types[0],static_cast<const uint8_t*>(pools[0])+offset,embd,ff,1,input+size_t(tokens[rank])*size_t(embd),gate_up+rank*size_t(ff),false);
            multiply(types[1],static_cast<const uint8_t*>(pools[1])+offset,embd,ff,1,input+size_t(tokens[rank])*size_t(embd),gate_up+(hit+rank)*size_t(ff),false);
        }
        // Gate and up outputs are adjacent and finish in one D2H join.
        const size_t gate_bytes=2*hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_gate_up,gate_up,gate_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));
        stat.gate_up_seconds+=elapsed(gate_start);++stat.joins;
        const auto oai_start=Clock::now();
        for(size_t rank=0;rank<hit;++rank)for(int row=0;row<ff;++row){
            const size_t i=rank*size_t(ff)+size_t(row);
            const float gate=host_gate_up[i]*scales[0][rank];
            const float up=host_gate_up[hit*size_t(ff)+i]*scales[1][rank];
            const float a=std::min(gate,7.f),b=std::clamp(up,-7.f,7.f);
            const float glu=a/(1.f+expf(1.702f*(-a)));
            host_activation[i]=glu*(b+1.f);
        }
        stat.oai_seconds+=elapsed(oai_start);
        const auto down_start=Clock::now();const size_t activation_bytes=hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(activation,host_activation,activation_bytes,cudaMemcpyHostToDevice,stream));
        for(size_t rank=0;rank<hit;++rank){
            const auto*weights_base=static_cast<const uint8_t*>(pools[2])+size_t(slots[rank])*stride;
            multiply(types[2],weights_base,ff,embd,1,activation+rank*size_t(ff),down+rank*size_t(embd),false);
        }
        const size_t down_bytes=hit*size_t(embd)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_down,down,down_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));++stat.joins;
        for(size_t rank=0;rank<hit;++rank)for(int row=0;row<embd;++row){
            const size_t i=rank*size_t(embd)+size_t(row);
            const float scaled=host_down[i]*scales[2][rank];
            host_down[i]=scaled*weights[rank];
        }
        stat.down_seconds+=elapsed(down_start);stat.worker_seconds+=elapsed(start);
        stat.multiply_calls+=3*hit;
        ++stat.jobs;stat.experts+=hit;stat.h2d_bytes+=size_t(embd)*size_t(token_count)*sizeof(float)+activation_bytes;stat.d2h_bytes+=gate_bytes+down_bytes;
    }
    void run_grouped(const std::vector<size_t>&order){
        const auto start=Clock::now();const size_t hit=slots.size();
        if(fesetenv(&environment))throw std::runtime_error("M31 cached exact: cannot apply FP environment");
#if defined(__SSE__)
        _mm_setcsr(mxcsr);
#endif
        // host_down/down are unused before the down stage. Gather input by
        // expert here; no new device or pinned allocation is needed.
        const auto gate_start=Clock::now();
        for(size_t p=0;p<hit;++p)std::memcpy(host_down+p*size_t(embd),host_input+size_t(tokens[order[p]])*size_t(embd),size_t(embd)*sizeof(float));
        const size_t gathered_bytes=hit*size_t(embd)*sizeof(float);
        exact_check(cudaMemcpyAsync(down,host_down,gathered_bytes,cudaMemcpyHostToDevice,stream));
        size_t groups=0;
        for(size_t first=0;first<hit;){
            size_t end=first+1;while(end<hit&&slots[order[end]]==slots[order[first]])++end;
            const int count=int(end-first);const size_t offset=size_t(slots[order[first]])*stride;
            const auto*x=down+first*size_t(embd);
            // false preserves original n1 IQ3/IQ4 accumulation at every count;
            // NVFP4 always takes its scalar lane8 contract, never MMA.
            multiply(types[0],static_cast<const uint8_t*>(pools[0])+offset,embd,ff,count,x,gate_up+first*size_t(ff),false);
            multiply(types[1],static_cast<const uint8_t*>(pools[1])+offset,embd,ff,count,x,gate_up+(hit+first)*size_t(ff),false);
            ++groups;first=end;
        }
        const size_t gate_bytes=2*hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_gate_up,gate_up,gate_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));++stat.joins;stat.gate_up_seconds+=elapsed(gate_start);
        const auto oai_start=Clock::now();
        for(size_t p=0;p<hit;++p)for(int row=0;row<ff;++row){
            const size_t i=p*size_t(ff)+size_t(row),original=order[p];
            const float gate=host_gate_up[i]*scales[0][original];
            const float up=host_gate_up[hit*size_t(ff)+i]*scales[1][original];
            const float a=std::min(gate,7.f),b=std::clamp(up,-7.f,7.f);
            const float glu=a/(1.f+expf(1.702f*(-a)));
            host_activation[i]=glu*(b+1.f);
        }
        stat.oai_seconds+=elapsed(oai_start);
        const auto down_start=Clock::now();const size_t activation_bytes=hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(activation,host_activation,activation_bytes,cudaMemcpyHostToDevice,stream));
        for(size_t first=0;first<hit;){
            size_t end=first+1;while(end<hit&&slots[order[end]]==slots[order[first]])++end;
            const auto*w=static_cast<const uint8_t*>(pools[2])+size_t(slots[order[first]])*stride;
            multiply(types[2],w,ff,embd,int(end-first),activation+first*size_t(ff),down+first*size_t(embd),false);
            first=end;
        }
        const size_t down_bytes=hit*size_t(embd)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_down,down,down_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));++stat.joins;
        // GU storage is dead after OAI; restore the original assignment/rank
        // order there, preserving two separate scale/weight multiplies.
        for(size_t p=0;p<hit;++p)for(int row=0;row<embd;++row){
            const size_t original=order[p];
            const float scaled=host_down[p*size_t(embd)+size_t(row)]*scales[2][original];
            host_gate_up[original*size_t(embd)+size_t(row)]=scaled*weights[original];
        }
        grouped_result=true;stat.down_seconds+=elapsed(down_start);stat.worker_seconds+=elapsed(start);
        ++stat.jobs;stat.experts+=hit;++stat.grouped_jobs;stat.grouped_assignments+=hit;stat.grouped_unique_experts+=groups;stat.multiply_calls+=3*groups;
        stat.h2d_bytes+=gathered_bytes+activation_bytes;stat.d2h_bytes+=gate_bytes+down_bytes;
    }
    bool can_join_gate_up() const {
        if(grouped_mode||!joined_gu_mode||types[0]!=types[1]||!pools[0]||!pools[1]||ff>std::numeric_limits<int>::max()/2)return false;
        if(types[0]!=GGML_TYPE_IQ3_XXS&&types[0]!=GGML_TYPE_IQ4_XS&&types[0]!=GGML_TYPE_NVFP4)return false;
        const size_t row_bytes=ggml_row_size(ggml_type(types[0]),embd);
        if(!row_bytes||size_t(ff)>std::numeric_limits<size_t>::max()/row_bytes)return false;
        const size_t raw=row_bytes*size_t(ff);
        if(raw>std::numeric_limits<size_t>::max()/2||2*raw>stride)return false;
        const uintptr_t gate=reinterpret_cast<uintptr_t>(pools[0]),up=reinterpret_cast<uintptr_t>(pools[1]);
        return gate<=std::numeric_limits<uintptr_t>::max()-raw&&up==gate+raw;
    }
    void run_joined_gate_up(const std::vector<size_t>&order,bool grouped){
        const auto start=Clock::now();const size_t hit=slots.size();
        if(fesetenv(&environment))throw std::runtime_error("M31 cached exact: cannot apply FP environment");
#if defined(__SSE__)
        _mm_setcsr(mxcsr);
#endif
        const auto gate_start=Clock::now();size_t input_bytes=0;
        if(grouped){
            for(size_t p=0;p<hit;++p)std::memcpy(host_down+p*size_t(embd),host_input+size_t(tokens[order[p]])*size_t(embd),size_t(embd)*sizeof(float));
            input_bytes=hit*size_t(embd)*sizeof(float);
            exact_check(cudaMemcpyAsync(down,host_down,input_bytes,cudaMemcpyHostToDevice,stream));
        }else{
            input_bytes=size_t(embd)*size_t(token_count)*sizeof(float);
            exact_check(cudaMemcpyAsync(input,host_input,input_bytes,cudaMemcpyHostToDevice,stream));
        }
        size_t groups=0;
        for(size_t first=0;first<hit;){
            size_t end=first+1;if(grouped)while(end<hit&&slots[order[end]]==slots[order[first]])++end;
            const auto*w=static_cast<const uint8_t*>(pools[0])+size_t(slots[order[first]])*stride;
            const auto*x=grouped?down+first*size_t(embd):input+size_t(tokens[order[first]])*size_t(embd);
            // Two adjacent matrices become rows[gate;up]. Each row still uses
            // the original n1-compatible tree and the same once-quantized X.
            multiply(types[0],w,embd,2*ff,int(end-first),x,gate_up+2*first*size_t(ff),false);
            ++groups;first=end;
        }
        const size_t gate_bytes=2*hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_gate_up,gate_up,gate_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));++stat.joins;stat.gate_up_seconds+=elapsed(gate_start);
        const auto oai_start=Clock::now();
        for(size_t p=0;p<hit;++p)for(int row=0;row<ff;++row){
            const size_t original=order[p],i=p*size_t(ff)+size_t(row);
            const float gate=host_gate_up[2*p*size_t(ff)+size_t(row)]*scales[0][original];
            const float up=host_gate_up[(2*p+1)*size_t(ff)+size_t(row)]*scales[1][original];
            const float a=std::min(gate,7.f),b=std::clamp(up,-7.f,7.f);
            const float glu=a/(1.f+expf(1.702f*(-a)));
            host_activation[i]=glu*(b+1.f);
        }
        stat.oai_seconds+=elapsed(oai_start);
        const auto down_start=Clock::now();const size_t activation_bytes=hit*size_t(ff)*sizeof(float);
        exact_check(cudaMemcpyAsync(activation,host_activation,activation_bytes,cudaMemcpyHostToDevice,stream));
        for(size_t first=0;first<hit;){
            size_t end=first+1;if(grouped)while(end<hit&&slots[order[end]]==slots[order[first]])++end;
            const auto*w=static_cast<const uint8_t*>(pools[2])+size_t(slots[order[first]])*stride;
            multiply(types[2],w,ff,embd,int(end-first),activation+first*size_t(ff),down+first*size_t(embd),false);
            first=end;
        }
        const size_t down_bytes=hit*size_t(embd)*sizeof(float);
        exact_check(cudaMemcpyAsync(host_down,down,down_bytes,cudaMemcpyDeviceToHost,stream));
        exact_check(cudaStreamSynchronize(stream));++stat.joins;
        for(size_t p=0;p<hit;++p)for(int row=0;row<embd;++row){
            const size_t original=order[p],i=p*size_t(embd)+size_t(row);
            const float scaled=host_down[i]*scales[2][original];
            if(grouped)host_gate_up[original*size_t(embd)+size_t(row)]=scaled*weights[original];
            else host_down[i]=scaled*weights[original];
        }
        grouped_result=grouped;stat.down_seconds+=elapsed(down_start);stat.worker_seconds+=elapsed(start);
        ++stat.jobs;stat.experts+=hit;stat.multiply_calls+=2*groups;
        if(grouped){++stat.grouped_jobs;stat.grouped_assignments+=hit;stat.grouped_unique_experts+=groups;}
        ++stat.joined_gu_jobs;stat.joined_gu_assignments+=hit;stat.joined_gu_groups+=groups;
        stat.h2d_bytes+=input_bytes+activation_bytes;stat.d2h_bytes+=gate_bytes+down_bytes;
    }

    void run(){
        grouped_result=false;
        if(grouped_mode){
            const auto prep_start=Clock::now();
            std::vector<size_t>order(slots.size());std::iota(order.begin(),order.end(),size_t(0));
            std::stable_sort(order.begin(),order.end(),[this](size_t a,size_t b){return slots[a]<slots[b];});
            bool duplicate=false;for(size_t i=1;i<order.size();++i)duplicate|=slots[order[i-1]]==slots[order[i]];
            const double prep_seconds=elapsed(prep_start);
            if(duplicate){run_grouped(order);stat.worker_seconds+=prep_seconds;return;}
            run_baseline();stat.worker_seconds+=prep_seconds;return;
        }
        run_baseline();
    }

};
GpuCachedExact::GpuCachedExact(int e,int f,int u):GpuCachedExact(e,f,u,1){}
GpuCachedExact::GpuCachedExact(int e,int f,int u,int n):p_(std::make_unique<Impl>(e,f,u,n)){}
GpuCachedExact::~GpuCachedExact()=default;
void GpuCachedExact::begin(const std::array<int,3>&types,const std::array<const void*,3>&pools,
    size_t stride,const std::vector<float>&input,const std::vector<int32_t>&slots,
    const std::vector<float>&weights,const std::array<std::vector<float>,3>&scales){
    std::vector<Assignment> assignments;assignments.reserve(slots.size());
    for(size_t rank=0;rank<slots.size();++rank)assignments.push_back({0,int(rank),slots[rank]});
    begin_batched(types,pools,stride,input,1,assignments,weights,scales);
}
void GpuCachedExact::begin_batched(const std::array<int,3>&types,const std::array<const void*,3>&pools,
    size_t stride,const std::vector<float>&input,int n,const std::vector<Assignment>&assignments,
    const std::vector<float>&weights,const std::array<std::vector<float>,3>&scales){
    if(n<=0||n>p_->maximum_tokens||input.size()!=size_t(p_->embd)*size_t(n)||assignments.empty()||
        assignments.size()>size_t(p_->maximum_used)*size_t(n)||weights.size()!=assignments.size()||!stride)
        throw std::invalid_argument("M31 cached exact batch: invalid input/hit geometry");
    for(size_t role=0;role<3;++role){
        if((types[role]!=GGML_TYPE_IQ3_XXS&&types[role]!=GGML_TYPE_IQ4_XS&&types[role]!=GGML_TYPE_NVFP4)||!pools[role]||scales[role].size()!=assignments.size())throw std::invalid_argument("M31 cached exact batch: invalid role");
        for(float scale:scales[role])if(!std::isfinite(scale))throw std::invalid_argument("M31 cached exact batch: nonfinite scale");
    }
    int previous=-1;
    for(size_t i=0;i<assignments.size();++i){
        const auto&a=assignments[i];
        if(a.token<0||a.token>=n||a.rank<0||a.rank>=p_->maximum_used||a.slot<0||!std::isfinite(weights[i]))throw std::invalid_argument("M31 cached exact batch: invalid assignment");
        const int key=a.token*p_->maximum_used+a.rank;
        if(key<=previous)throw std::invalid_argument("M31 cached exact batch: assignments must be token/rank ordered and unique");
        for(size_t j=0;j<i;++j)if(assignments[j].token==a.token&&assignments[j].slot==a.slot)throw std::invalid_argument("M31 cached exact batch: duplicate expert slot within token");
        previous=key;
    }
    std::lock_guard lock(p_->mutex);if(p_->active)throw std::runtime_error("M31 cached exact: overlapping submissions");
    if(fegetenv(&p_->environment))throw std::runtime_error("M31 cached exact: cannot read FP environment");
#if defined(__SSE__)
    p_->mxcsr=_mm_getcsr();
#endif
    std::memcpy(p_->host_input,input.data(),input.size()*sizeof(float));
    p_->slots.clear();p_->tokens.clear();
    for(const auto&a:assignments){p_->slots.push_back(a.slot);p_->tokens.push_back(a.token);}
    p_->token_count=n;p_->types=types;p_->pools=pools;p_->stride=stride;p_->weights=weights;p_->scales=scales;
    p_->failure=nullptr;p_->done=false;p_->active=true;p_->pending=true;p_->wake.notify_one();
}
void GpuCachedExact::finish(std::vector<float>&terms){
    std::unique_lock lock(p_->mutex);if(!p_->active)throw std::runtime_error("M31 cached exact: no pending submission");
    p_->completed.wait(lock,[this]{return p_->done;});p_->active=false;if(p_->failure)std::rethrow_exception(p_->failure);
    const float*result=p_->grouped_result?p_->host_gate_up:p_->host_down;
    terms.assign(result,result+size_t(p_->embd)*p_->slots.size());
}
void GpuCachedExact::drain()noexcept{p_->drain();}
GpuCachedExact::Stats GpuCachedExact::stats()const{std::lock_guard lock(p_->mutex);if(p_->active)throw std::runtime_error("M31 cached exact: profile requested during submission");return p_->stat;}
}
#else
#include <stdexcept>
namespace strata::m31 {
struct GpuCachedExact::Impl{};
GpuCachedExact::GpuCachedExact(int,int,int){throw std::runtime_error("M31 cached exact requires CUDA CPU-compatible dots");}
GpuCachedExact::GpuCachedExact(int,int,int,int){throw std::runtime_error("M31 cached exact requires CUDA CPU-compatible dots");}
GpuCachedExact::~GpuCachedExact()=default;
void GpuCachedExact::begin_batched(const std::array<int,3>&,const std::array<const void*,3>&,size_t,const std::vector<float>&,int,const std::vector<Assignment>&,const std::vector<float>&,const std::array<std::vector<float>,3>&){throw std::runtime_error("M31 cached exact unavailable");}
void GpuCachedExact::begin(const std::array<int,3>&,const std::array<const void*,3>&,size_t,const std::vector<float>&,const std::vector<int32_t>&,const std::vector<float>&,const std::array<std::vector<float>,3>&){throw std::runtime_error("M31 cached exact unavailable");}
void GpuCachedExact::finish(std::vector<float>&){throw std::runtime_error("M31 cached exact unavailable");}
void GpuCachedExact::drain()noexcept{}
GpuCachedExact::Stats GpuCachedExact::stats()const{return {};}
}
#endif
