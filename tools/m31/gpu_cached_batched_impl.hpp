// Included at namespace scope. Only resident weights participate: CPU computes
// misses while the exact private stream handles cached original n1 assignments.
bool GpuExperts::forward_cached_batched(int layer,int n,const std::vector<float>&x,
    const std::vector<int32_t>&ids,const std::vector<float>&weights,
    const MaskedDecode&cpu_misses,std::vector<float>&output){
#ifndef M31_CPU_QUANTS_CUDA
    (void)layer;(void)n;(void)x;(void)ids;(void)weights;(void)cpu_misses;(void)output;
    throw std::runtime_error("M31 cached batched requires STRATA_M31_CPU_QUANTS_CUDA");
#else
    check(p_->ready&&p_->g.routed(layer),"uninitialized cached verifier or invalid routed layer");
    check(p_->cached_verify_enabled&&p_->cached_batch_worker&&native_gpu_math()&&!p_->layer_banks_requested,
          "cached verifier requires M31_DSPARK_GPU_CACHED=1 and native LRU prefill");
    const auto&g=p_->g;const size_t routes=size_t(n>0?n:0)*size_t(g.used);
    check(n>0&&n<=8&&x.size()==size_t(g.embd)*size_t(n)&&ids.size()==routes&&weights.size()==routes,
          "cached verifier input dimensions differ");
    check(&output != &x && &output != &weights&&bool(cpu_misses),"cached verifier requires separate output/CPU callback");
    // Validate every assignment before seed DMA, metadata updates or GPU work.
    for(size_t i=0;i<routes;++i){
        check(ids[i]>=0&&ids[i]<g.experts&&std::isfinite(weights[i]),"invalid cached verifier route");
        for(size_t j=i-i%size_t(g.used);j<i;++j)check(ids[j]!=ids[i],"duplicate cached verifier top-k expert");
    }
    check(p_->active_read_keys.empty(),"overlapping cached expert consumers are unsupported");
    const auto wall_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    const auto seed_start=wall_start;
    // Cached verification is a decode transition, not streaming prefill: keep
    // full decode-only pins across rounds and speculative rewind.
    p_->prime_decode_seed();
    if(p_->profile)p_->stat.cached_batch_seed_seconds+=Impl::elapsed(seed_start);
    check(p_->pending_prefetch.empty(),"cached verifier has pending prefill weights");
    std::vector<uint8_t>cpu_mask(routes,1);
    std::vector<int32_t>gpu_index(routes,-1);
    std::vector<GpuCachedExact::Assignment>assignments;
    std::vector<float>hit_weights;
    std::array<std::vector<float>,3>hit_scales;
    std::set<uint64_t>readers;
    const auto&info=p_->layers[size_t(layer)];
    for(size_t i=0;i<routes;++i){
        const auto key=p_->key(layer,ids[i]);const auto found=p_->entries.find(key);
        if(found==p_->entries.end())continue;
        gpu_index[i]=int32_t(assignments.size());cpu_mask[i]=0;
        assignments.push_back({int(i/size_t(g.used)),int(i%size_t(g.used)),int(found->second.slot)});
        hit_weights.push_back(weights[i]);
        for(size_t role=0;role<3;++role)hit_scales[role].push_back(info.scales[role][size_t(ids[i])]);
        readers.insert(key);
    }
    ++p_->stat.cached_batch_calls;p_->stat.cached_batch_tokens+=uint64_t(n);
    p_->stat.cached_batch_routes+=routes;p_->stat.cached_batch_hits+=assignments.size();
    if(assignments.empty()){
        ++p_->stat.cached_batch_allmiss;
        if(p_->profile)p_->stat.cached_batch_wall_seconds+=Impl::elapsed(wall_start);
        return false;
    }
    p_->active_read_keys=readers;
    struct Lease {
        Impl*owner;
        ~Lease(){owner->cached_batch_worker->drain();owner->active_read_keys.clear();}
    }lease{p_.get()};
    const auto submit_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    const auto*base=static_cast<const uint8_t*>(ggml_backend_buffer_get_base(p_->pool));
    const std::array<int,3>types{int(info.types[0]),int(info.types[1]),int(info.types[2])};
    const std::array<const void*,3>pools{base+info.offsets[0],base+info.offsets[1],base+info.offsets[2]};
    p_->cached_batch_worker->begin_batched(types,pools,p_->stat.slot_stride,x,n,assignments,hit_weights,hit_scales);
    if(p_->profile)p_->stat.cached_batch_submit_seconds+=Impl::elapsed(submit_start);
    const auto cpu_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    std::vector<float>cpu_terms;
    if(assignments.size()!=routes){
        cpu_misses(cpu_mask,cpu_terms);
        check(cpu_terms.size()==routes*size_t(g.embd),"masked CPU callback returned invalid full-rank terms");
    }
    if(p_->profile)p_->stat.cached_batch_cpu_seconds+=Impl::elapsed(cpu_start);
    const auto join_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    std::vector<float>gpu_terms;
    p_->cached_batch_worker->finish(gpu_terms);
    check(gpu_terms.size()==assignments.size()*size_t(g.embd),"cached verifier GPU contribution geometry differs");
    output.resize(size_t(g.embd)*size_t(n));
    for(int token=0;token<n;++token)for(size_t row=0;row<size_t(g.embd);++row){
        float sum=0;
        for(int rank=0;rank<g.used;++rank){
            const size_t route=size_t(token)*size_t(g.used)+size_t(rank);
            const int32_t hit=gpu_index[route];
            const float term=hit>=0?gpu_terms[size_t(hit)*size_t(g.embd)+row]:cpu_terms[route*size_t(g.embd)+row];
            sum=rank?sum+term:term;
        }
        output[size_t(token)*size_t(g.embd)+row]=sum;
    }
    for(auto key:readers){p_->entries.at(key).last_used=++p_->clock;++p_->stat.hits;}
    ++p_->stat.cached_batch_forwards;++p_->stat.forwards;
    if(p_->profile){p_->stat.cached_batch_join_seconds+=Impl::elapsed(join_start);p_->stat.cached_batch_wall_seconds+=Impl::elapsed(wall_start);}
    return true;
#endif
}
