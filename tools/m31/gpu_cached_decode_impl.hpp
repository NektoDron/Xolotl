// Included at namespace scope by m31_gpu_experts.cpp. Native grouped n1 decode;
// retains original [used,1] dispatch and top-k accumulation across CPU/GPU.
bool GpuExperts::forward_cached_hybrid(int layer,const std::vector<float>&x,
    const std::vector<int32_t>&ids,const std::vector<float>&weights,
    const MissingDecode&cpu_misses,std::vector<float>&output){
    check(p_->ready&&p_->g.routed(layer),"uninitialized hybrid decode or invalid layer");
    check(native_gpu_math()&&!p_->layer_banks_requested,"hybrid decode requires native LRU experts");
    const auto &g=p_->g;
    check(x.size()==size_t(g.embd)&&ids.size()==size_t(g.used)&&weights.size()==ids.size(),"hybrid decode dimensions differ");
    check(&output != &x && &output != &weights&&bool(cpu_misses),"hybrid decode needs separate output and CPU callback");
    // Validate all route data before priming any optional decode-only hot set.
    for(size_t rank=0;rank<ids.size();++rank){
        check(ids[rank]>=0&&ids[rank]<g.experts&&std::isfinite(weights[rank]),"invalid hybrid route");
        for(size_t j=0;j<rank;++j)check(ids[j]!=ids[rank],"duplicate hybrid top-k expert");
    }
    p_->prime_decode_seed();
    std::vector<int32_t> mapped(ids.size(),-1);size_t hits=0;
    std::set<uint64_t> readers;
    // Validate all IDs before updating counters, creating graphs, or submitting work.
    for(size_t rank=0;rank<ids.size();++rank){
        check(ids[rank]>=0&&ids[rank]<g.experts&&std::isfinite(weights[rank]),"invalid hybrid route");
        for(size_t j=0;j<rank;++j)check(ids[j]!=ids[rank],"duplicate hybrid top-k expert");
        const auto found=p_->entries.find(p_->key(layer,ids[rank]));
        if(found!=p_->entries.end()){mapped[rank]=int32_t(found->second.slot);++hits;readers.insert(found->first);}
    }
    p_->stat.decode_route_lookups+=ids.size();p_->stat.decode_resident_routes+=hits;
    if(!hits)return false;
    check(p_->pending_prefetch.empty(),"hybrid decode has pending prefill weights");
    check(p_->active_read_keys.empty(),"overlapping hybrid expert consumers are unsupported");
    p_->active_read_keys=readers;
    // The callback may throw while native GPU work is in flight. Always drain the
    // actual borrowed backend before releasing slot leases or graph/input owners.
    struct Drain {Impl*owner;~Drain(){
#ifdef M31_CPU_QUANTS_CUDA
        if(owner->cached_exact_worker)owner->cached_exact_worker->drain();
#endif
        ggml_backend_synchronize(owner->backend);owner->active_read_keys.clear();
    }}drain{p_.get()};
    const auto submit_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
#ifdef M31_CPU_QUANTS_CUDA
    if(p_->cached_decode_exact){
        if(!p_->cached_exact_worker){
            p_->cached_exact_worker=std::make_unique<GpuCachedExact>(int(g.embd),int(g.expert_ff),int(g.used));
            p_->stat.decode_scratch_bytes+=p_->cached_exact_worker->stats().device_bytes;
        }
        const auto&info=p_->layers[size_t(layer)];
        const auto*base=static_cast<const uint8_t*>(ggml_backend_buffer_get_base(p_->pool));
        const std::array<int,3>types{int(info.types[0]),int(info.types[1]),int(info.types[2])};
        const std::array<const void*,3>pools{base+info.offsets[0],base+info.offsets[1],base+info.offsets[2]};
        std::vector<int32_t>hit_slots,missing_ids;
        std::vector<float>hit_weights,missing_weights;
        std::array<std::vector<float>,3>hit_scales;
        std::vector<uint8_t>resident(ids.size());
        for(size_t rank=0;rank<ids.size();++rank){
            resident[rank]=mapped[rank]>=0;
            if(resident[rank]){
                hit_slots.push_back(mapped[rank]);hit_weights.push_back(weights[rank]);
                for(size_t role=0;role<3;++role)hit_scales[role].push_back(info.scales[role][size_t(ids[rank])]);
            }else{missing_ids.push_back(ids[rank]);missing_weights.push_back(weights[rank]);}
        }
        // Weight slots remain leased until the worker's raw stream completes,
        // including callback exceptions. The worker has its own CPU OAI thread.
        p_->cached_exact_worker->begin(types,pools,p_->stat.slot_stride,x,hit_slots,hit_weights,hit_scales);
        if(p_->profile)p_->stat.hybrid_submit_seconds+=Impl::elapsed(submit_start);
        const auto cpu_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
        std::vector<float>cpu_terms;
        if(!missing_ids.empty())cpu_misses(missing_ids,missing_weights,cpu_terms);
        check(cpu_terms.size()==size_t(g.embd)*missing_ids.size(),"exact hybrid CPU contribution geometry differs");
        if(p_->profile)p_->stat.hybrid_cpu_seconds+=Impl::elapsed(cpu_start);
        const auto join_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
        std::vector<float>hit_terms,gpu_terms(size_t(g.embd)*size_t(g.used));
        p_->cached_exact_worker->finish(hit_terms);
        size_t hit=0;
        for(size_t rank=0;rank<ids.size();++rank)if(resident[rank]){
            std::copy_n(hit_terms.data()+hit*size_t(g.embd),size_t(g.embd),gpu_terms.data()+rank*size_t(g.embd));++hit;
        }
        const char*rank_outer=std::getenv("M31_GPU_CACHED_RANK_OUTER");
        if(rank_outer&&std::string(rank_outer)=="1"){
            merge_cached_decode_terms_rank_outer(size_t(g.embd),resident,gpu_terms,cpu_terms,output);
            ++p_->cached_rank_outer_calls;
        }
        else merge_cached_decode_terms(size_t(g.embd),resident,gpu_terms,cpu_terms,output);
        for(auto key:readers){p_->entries.at(key).last_used=++p_->clock;++p_->stat.hits;}
        if(p_->profile)p_->stat.hybrid_join_seconds+=Impl::elapsed(join_start);
        ++p_->stat.hybrid_forwards;++p_->stat.forwards;
        return true;
    }
#endif
    auto &owner=p_->cached_decode_graphs[size_t(layer)];
    if(!owner){
        auto d=std::make_unique<Impl::CachedDecodeGraph>();
        d->context=std::make_unique<Context>(1024*1024);auto *c=d->context->value;
        d->graph=ggml_new_graph_custom(c,512,false);
        d->input=ggml_new_tensor_3d(c,GGML_TYPE_F32,g.embd,1,1);
        d->ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,g.used,1);
        d->weights=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,g.used,1);
        for(auto*t:{d->input,d->ids,d->weights})ggml_set_input(t);
        const auto&info=p_->layers[size_t(layer)];
        auto mm=[&](int role,ggml_tensor*activation){
            auto*weight=p_->weight_view(c,layer,role);auto*out=ggml_mul_mat_id(c,weight,activation,d->ids);
            if(weight->type==GGML_TYPE_NVFP4)ggml_prec_set_acc(out,GGML_PREC_BF16);
            if(p_->stat.q8_activations)ggml_prec_set_src(out,GGML_PREC_Q8,1);
            if(info.has_scale[size_t(role)]){
                auto*scale=ggml_new_tensor_1d(c,GGML_TYPE_F32,int64_t(p_->stat.slot_count));
                ggml_set_input(scale);d->scale_inputs[size_t(role)]=scale;
                auto*s=ggml_reshape_3d(c,scale,1,int64_t(p_->stat.slot_count),1);
                out=ggml_mul(c,out,ggml_get_rows(c,s,d->ids));
            }
            return out;
        };
        // Exact same graph geometry/precisions as native forward n1. Missing
        // ranks alias a resident expert and are discarded after completion.
        auto*gate=mm(0,d->input);auto*up=mm(1,d->input);
        auto*act=ggml_swiglu_oai(c,gate,up,1.702f,7.f);
        auto*down=mm(2,act);d->output=ggml_cont(c,ggml_mul(c,down,d->weights));
        ggml_set_output(d->output);ggml_build_forward_expand(d->graph,d->output);
        for(int i=0;i<ggml_graph_n_nodes(d->graph);++i){auto*node=ggml_graph_node(d->graph,i);check(ggml_backend_dev_supports_op(p_->device,node),std::string("hybrid decode unsupported ")+ggml_op_name(node->op));}
        d->allocator=ggml_gallocr_new(p_->buffer_type);
        check(d->allocator&&ggml_gallocr_alloc_graph(d->allocator,d->graph),"hybrid decode graph allocation failed");
        d->scratch=ggml_gallocr_get_buffer_size(d->allocator,0);
        check(ggml_is_contiguous(d->output)&&ggml_nelements(d->output)==g.embd*g.used,"hybrid decode output layout");
        for(auto&scales:d->scales)scales.assign(p_->stat.slot_count,1.f);
        d->gpu_terms.resize(size_t(g.embd)*size_t(g.used));d->resident.resize(size_t(g.used));
        p_->stat.decode_scratch_bytes+=d->scratch;
        owner=std::move(d);
    }
    auto&d=*owner;d.mapped=std::move(mapped);d.missing_ids.clear();d.missing_weights.clear();d.cpu_terms.clear();
    const int32_t placeholder=*std::find_if(d.mapped.begin(),d.mapped.end(),[](int32_t x){return x>=0;});
    for(size_t rank=0;rank<ids.size();++rank){
        d.resident[rank]=d.mapped[rank]>=0;
        if(d.resident[rank]){
            for(size_t role=0;role<3;++role)d.scales[role][size_t(d.mapped[rank])]=p_->layers[size_t(layer)].scales[role][size_t(ids[rank])];
        }else{d.mapped[rank]=placeholder;d.missing_ids.push_back(ids[rank]);d.missing_weights.push_back(weights[rank]);}
    }
    // Inputs remain alive until Drain runs. No cache load, host packing or weight
    // upload is performed here; misses go exclusively through the CPU callback.
    ggml_backend_tensor_set_async(p_->backend,d.input,x.data(),0,x.size()*sizeof(float));
    ggml_backend_tensor_set_async(p_->backend,d.ids,d.mapped.data(),0,d.mapped.size()*sizeof(int32_t));
    ggml_backend_tensor_set_async(p_->backend,d.weights,weights.data(),0,weights.size()*sizeof(float));
    for(size_t role=0;role<3;++role)if(d.scale_inputs[role])ggml_backend_tensor_set_async(p_->backend,d.scale_inputs[role],d.scales[role].data(),0,p_->stat.slot_count*sizeof(float));
    check(ggml_backend_graph_compute_async(p_->backend,d.graph)==GGML_STATUS_SUCCESS,"hybrid decode graph submission failed");
    if(p_->profile)p_->stat.hybrid_submit_seconds+=Impl::elapsed(submit_start);
    const auto cpu_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    if(!d.missing_ids.empty())cpu_misses(d.missing_ids,d.missing_weights,d.cpu_terms);
    check(d.cpu_terms.size()==size_t(g.embd)*d.missing_ids.size(),"hybrid CPU callback contribution geometry differs");
    if(p_->profile)p_->stat.hybrid_cpu_seconds+=Impl::elapsed(cpu_start);
    const auto join_start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
    ggml_backend_synchronize(p_->backend);
    ggml_backend_tensor_get(d.output,d.gpu_terms.data(),0,d.gpu_terms.size()*sizeof(float));
    merge_cached_decode_terms(size_t(g.embd),d.resident,d.gpu_terms,d.cpu_terms,output);
    for(auto key:readers){p_->entries.at(key).last_used=++p_->clock;++p_->stat.hits;}
    if(p_->profile)p_->stat.hybrid_join_seconds+=Impl::elapsed(join_start);
    ++p_->stat.hybrid_forwards;++p_->stat.forwards;++p_->stat.graph_batches;
    return true;
}
