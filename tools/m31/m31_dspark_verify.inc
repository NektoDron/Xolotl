// Included inside strata::m31, after Engine::Impl is defined.
// Experimental exact-shape target verification: each dense query retains n1
// projections, MSA sorted gathers and flash-attention geometry. Only the CPU
// routed FFN processes several tokens together, using nrc=1 scalar dot calls.
bool Engine::verify(const int32_t * tokens,int n,std::vector<float> & logits,std::string & error) {
    try {
        auto & p=*p_;const auto & g=p.g;
        require(p.ready&&!p.poisoned,"verify requires a valid initialized engine");
        require(tokens&&n>0&&n<=8&&p.pos+n<=p.cfg.context,"invalid small verification batch");
        require(p.cpu_pool&&p.cfg.persistent_cpu_pool&&!p.cfg.tap,"verify requires persistent CPU pool and no tap");
        for(int i=0;i<n;++i)require(tokens[i]>=0&&tokens[i]<g.vocab,"verify token outside vocabulary");
        const auto dense_batch_value=std::getenv("M31_DSPARK_BATCH_DENSE");
        const bool batch_dense=dense_batch_value&&std::string(dense_batch_value)=="1";
#if defined(M31_ENGINE_CUDA) && defined(GGML_CUDA_HAS_EXACT_MMV_MARKER)
        require(!batch_dense||ggml_backend_is_cuda(p.backend),"batched dense verification requires CUDA");
#else
        require(!batch_dense,"batched dense verification requires the exact MMV marker API");
#endif
        const auto shared_input_value=std::getenv("M31_DSPARK_SHARED_INPUTS");
        const bool shared_inputs=shared_input_value&&std::string(shared_input_value)=="1";
        const auto batch_ops_value=std::getenv("M31_DSPARK_BATCH_OPS");
        const bool batch_ops=batch_ops_value&&std::string(batch_ops_value)=="1";
        require(!batch_ops||(batch_dense&&shared_inputs),"batched ops require batched dense and shared inputs");
        const auto overlap_value=std::getenv("M31_DSPARK_SHARED_OVERLAP");
        const bool shared_overlap=overlap_value&&std::string(overlap_value)=="1";
        require(!shared_overlap||batch_dense,"shared overlap requires batched dense verification");
        const auto view_value=std::getenv("M31_DSPARK_VIEW_INPUTS");
        const bool view_inputs=view_value&&std::string(view_value)=="1";
        require(!view_inputs||batch_dense,"view inputs require batched dense verification");
        const auto coalesce_value=std::getenv("M31_DSPARK_COALESCE_INPUTS");
        const bool coalesce_inputs=coalesce_value&&std::string(coalesce_value)=="1";
        require(!coalesce_inputs||view_inputs,"coalesced inputs require view inputs");
        const auto packed_value=std::getenv("M31_DSPARK_PACKED_ROUTED");
        const bool packed_routed=packed_value&&std::string(packed_value)=="1";
        require(!packed_routed||batch_dense,"packed routed metadata requires batched dense verification");
        const auto msa_value=std::getenv("M31_DSPARK_BATCH_MSA");
        const bool batch_msa=msa_value&&std::string(msa_value)=="1";
        require(!batch_msa||(batch_dense&&batch_ops&&g.index_heads==4),"batched MSA requires batched dense/ops and four index heads");
        const auto graph_opt=std::getenv("GGML_CUDA_GRAPH_OPT");
        require(!batch_msa||!graph_opt||std::atoi(graph_opt)!=1,"batched MSA requires GGML_CUDA_GRAPH_OPT=0 to preserve explicit KV write ordering");
        const auto gpu_cached_value=std::getenv("M31_DSPARK_GPU_CACHED");
        const bool gpu_cached=gpu_cached_value&&std::string(gpu_cached_value)=="1";
        require(!gpu_cached||p.gpu_experts!=nullptr,"cached verification requires a GPU expert cache");
        const int64_t first=p.pos;
        // Once graph ownership or KV changes, unexpected errors require reset.
        p.poisoned=true;p.set_dense_fusion(true);p.verified_logits.clear();p.clear_dense_decode_cache(true);
        p.decode_chain_active=false;p.prefill_chain_active=false;
        auto embedding=p.host.get("token_embd.weight");
        auto traits=ggml_get_type_traits(embedding->type);
        require(embedding->type==GGML_TYPE_F32||traits->to_float,"no embedding decoder");
        std::vector<float> hidden(size_t(g.embd)*size_t(n));
        for(int row=0;row<n;++row){
            const auto source=static_cast<const uint8_t *>(embedding->data)+size_t(tokens[row])*embedding->nb[1];
            auto destination=hidden.data()+size_t(row)*size_t(g.embd);
            if(embedding->type==GGML_TYPE_F32)std::memcpy(destination,source,size_t(g.embd)*sizeof(float));
            else traits->to_float(source,destination,g.embd);
        }
        p.invalidate_capture(); // verification writes the legacy owned row vector
        p.captured_rows=p.cfg.draft_feature_layers.empty()?0:std::min(n,p.cfg.draft_feature_max_rows);
        p.captured_first=first+n-p.captured_rows;
        const size_t feature_width=p.cfg.draft_feature_layers.size()*size_t(g.embd);
        p.captured_features.resize(size_t(p.captured_rows)*feature_width);
        const auto persistent_value=std::getenv("M31_DSPARK_PERSISTENT_VERIFY");
        const bool persistent=persistent_value&&std::string(persistent_value)=="1";
        std::unique_ptr<DSparkVerifyState> transient;
        if(persistent){if(!p.verify_state)p.verify_state=std::make_unique<DSparkVerifyState>();}
        else {p.verify_state.reset();transient=std::make_unique<DSparkVerifyState>();}
        auto & state=persistent?*p.verify_state:*transient;
        const auto verify_started=Impl::Clock::now();
        const auto reconfigure_started=Impl::Clock::now();
        std::vector<int64_t> geometry{batch_dense?1:0,shared_inputs?1:0,batch_ops?1:0,shared_overlap?1:0,view_inputs?1:0,packed_routed?1:0,batch_msa?1:0,coalesce_inputs?1:0};
        for(int row=0;row<n;++row)geometry.push_back(kv_length(first+row+1,g));
        const bool replanned=state.reconfigure(geometry,int(g.layers));
        if(shared_inputs)state.prepare_shared(p.backend,g,n,first,p.causal_f16,batch_ops);
        const double reconfigure_ms=1000*Impl::elapsed(reconfigure_started);
        if(!state.cpu)state.cpu=std::make_unique<CpuBatchedVerify>(p.cpu,p.cpu_pool,unsigned(p.cfg.threads),8);
        double dense_ms=0,cpu_ms=0,build_ms=0,upload_ms=0,compute_ms=0,read_ms=0,fill_ms=0,hybrid_ms=0,cpu_ffn_ms=0,shared_submit_ms=0,shared_wait_ms=0;
        if(shared_inputs){
            const auto fill_started=Impl::Clock::now();
            if(batch_ops){std::vector<int32_t> positions(size_t(n),0);for(int row=0;row<n;++row)positions[size_t(row)]=int32_t(first+row);
                ggml_backend_tensor_set(state.position_matrix,positions.data(),0,positions.size()*sizeof(int32_t));}
            for(int row=0;row<n;++row){auto & input=*state.attention[size_t(row)];p.pos=first+row;
                p.fill_attention(input.positions,input.mask,input.bias,nullptr,1,input.nk,input.nk/g.block,!batch_ops);}
            p.pos=first;fill_ms=1000*Impl::elapsed(fill_started);
        }
        for(int layer=0;layer<g.layers;++layer){
            const auto dense_started=Impl::Clock::now();
            auto & owner=state.layers[size_t(layer)];
            if(!owner)owner=std::make_unique<DSparkVerifyState::Layer>();
            auto & storage=*owner;
            const bool built=!storage.graph;
            if(!storage.graph) {
            storage.graph=std::make_unique<Graph>(p.backend,p.cfg,first,nullptr,32*1024*1024,8192);
            auto & b=*storage.graph;auto c=b.c;
            b.dspark_view_inputs=view_inputs;
            b.dspark_coalesce_inputs=coalesce_inputs;
            auto & inputs=storage.inputs;inputs.resize(size_t(n));
            auto & packed_norm=storage.norm;auto & packed_residual=storage.residual;auto & packed_shared=storage.shared;
            auto & packed_ids=storage.ids;auto & packed_weights=storage.weights;auto & packed_output=storage.output;
            auto join=[&](ggml_tensor * & packed,ggml_tensor * value){
                if(!view_inputs||!ggml_is_contiguous(value))value=ggml_cont(c,value);
                packed=packed?ggml_concat(c,packed,value,1):value;
            };
            if(batch_dense) {
#include "m31_dspark_batched_graph.inc"
            } else {
            for(int row=0;row<n;++row){
                auto & in=inputs[size_t(row)];const auto position=first+row;in.nk=kv_length(position+1,g);
                if(shared_inputs){
                    auto & source=*state.attention[size_t(row)];
                    in.hidden=b.borrow(state.hidden_rows[size_t(row)]);in.positions=b.borrow(source.positions);
                    in.mask=b.borrow(source.mask);in.bias=b.borrow(source.bias);
                }else{
                    in.hidden=b.input(GGML_TYPE_F32,g.embd,1);in.positions=b.input(GGML_TYPE_I32,1);
                    in.mask=b.input(p.causal_f16?GGML_TYPE_F16:GGML_TYPE_F32,in.nk,1,1);
                    in.bias=b.input(GGML_TYPE_F32,in.nk/g.block,1,1);
                }
                // n==1 is intentional. It selects v8's packed projections and
                // score-sorted sparse KV gather rather than the prefill graph.
                auto attention=p.attention(b,layer,position,1,in.hidden,in.positions,in.mask,in.bias,true);
                auto residual=ggml_add(c,attention,in.hidden);
                const std::string prefix="blk."+std::to_string(layer)+".";
                auto normalized=rms(b,residual,p.dense->get(prefix+"ffn_norm.weight"),g.rms_eps);
                if(!g.routed(layer)){
                    auto output=ggml_add(c,ffn(b,*p.dense,normalized,prefix,""),residual);
                    join(packed_output,output);continue;
                }
                auto shared=ffn(b,*p.dense,normalized,prefix,"_shexp");
                auto probabilities=ggml_sigmoid(c,ggml_mul_mat(c,p.dense->get(prefix+"ffn_gate_inp.weight"),normalized));
                auto ids=ggml_top_k(c,ggml_add(c,probabilities,p.dense->get(prefix+"exp_probs_b.bias")),int(g.used));
                auto weights=ggml_get_rows(c,ggml_reshape_3d(c,probabilities,1,g.experts,1),ids);
                weights=ggml_reshape_2d(c,weights,g.used,1);
                if(g.norm_routes){auto sum=ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY);weights=ggml_div(c,weights,sum);}
                if(g.route_scale!=1)weights=ggml_scale(c,weights,g.route_scale);
                join(packed_norm,normalized);join(packed_residual,residual);join(packed_shared,shared);
                join(packed_ids,ggml_reshape_2d(c,ids,g.used,1));join(packed_weights,weights);
            }
            }
            if(g.routed(layer)&&packed_routed){
                storage.packed_routes=std::make_unique<DSparkPackedRoutes>(dspark_pack_routes(c,packed_norm,packed_residual,packed_weights,packed_ids,int(g.experts)));
                // The external shared graph borrows the original device norm;
                // gallocr must retain that output independently of the packed copy.
                if(shared_overlap)b.allocate({storage.packed_routes->tensor,packed_norm});
                else b.allocate({storage.packed_routes->tensor,packed_shared});
            }
            else if(g.routed(layer)&&shared_overlap)b.allocate({packed_norm,packed_residual,packed_ids,packed_weights});
            else if(g.routed(layer))b.allocate({packed_norm,packed_residual,packed_shared,packed_ids,packed_weights});
            else b.allocate({packed_output});
            if(g.routed(layer)&&shared_overlap){
                storage.shared_graph=std::make_unique<Graph>(p.backend,p.cfg,first,nullptr,2*1024*1024,2048);
                auto & sb=*storage.shared_graph;
                sb.dspark_view_inputs=view_inputs;
                sb.dspark_coalesce_inputs=coalesce_inputs;
                auto input=sb.borrow(packed_norm);
                std::vector<ggml_tensor *> rows;
                for(int row=0;row<n;++row)rows.push_back(dspark_column(sb,input,row,g.embd));
                const auto output=dspark_ffn(sb,*p.dense,rows,"blk."+std::to_string(layer)+".","_shexp",batch_ops);
                storage.shared=dspark_join(sb,output,0,n);sb.allocate({storage.shared});
            }
            }
            if(built)build_ms+=1000*Impl::elapsed(dense_started);
            auto & b=*storage.graph;
            auto & inputs=storage.inputs;
            auto packed_norm=storage.norm,packed_residual=storage.residual,packed_shared=storage.shared;
            auto packed_ids=storage.ids,packed_weights=storage.weights,packed_output=storage.output;
            const auto upload_started=Impl::Clock::now();
            if(shared_inputs)ggml_backend_tensor_set(state.hidden_matrix,hidden.data(),0,hidden.size()*sizeof(float));
            else for(int row=0;row<n;++row){
                auto & in=inputs[size_t(row)];
                ggml_backend_tensor_set(in.hidden,hidden.data()+size_t(row)*size_t(g.embd),0,size_t(g.embd)*sizeof(float));
                p.pos=first+row;p.fill_attention(in.positions,in.mask,in.bias,nullptr,1,in.nk,in.nk/g.block);
            }
            p.pos=first;upload_ms+=1000*Impl::elapsed(upload_started);
            const auto compute_started=Impl::Clock::now();b.compute();compute_ms+=1000*Impl::elapsed(compute_started);
            const auto read_started=Impl::Clock::now();
            if(!g.routed(layer)){hidden=Graph::read(packed_output);read_ms+=1000*Impl::elapsed(read_started);}
            else {
                DSparkPackedRoutesHost legacy_metadata;
                auto & metadata=packed_routed?state.packed_host:legacy_metadata;
                auto & input=metadata.norm;auto & residual=metadata.residual;
                auto & weights=metadata.weights;auto & ids=metadata.ids;
                std::vector<float> shared;
                if(packed_routed){
                    dspark_read_packed_routes(*storage.packed_routes,state.packed_scratch,metadata);
                    if(!shared_overlap)shared=Graph::read(packed_shared);
                }else{
                    input=Graph::read(packed_norm);residual=Graph::read(packed_residual);
                    if(!shared_overlap)shared=Graph::read(packed_shared);
                    weights=Graph::read(packed_weights);ids.resize(size_t(n)*size_t(g.used));
                    ggml_backend_tensor_get(packed_ids,ids.data(),0,ids.size()*sizeof(int32_t));
                }
                read_ms+=1000*Impl::elapsed(read_started);
                dense_ms+=1000*Impl::elapsed(dense_started);
                struct SharedDrain {
                    ggml_backend_t backend;bool pending=false;
                    ~SharedDrain(){if(pending)ggml_backend_synchronize(backend);}
                    void join(){if(pending){ggml_backend_synchronize(backend);pending=false;}}
                } shared_work{p.backend};
                if(shared_overlap){
                    const auto started=Impl::Clock::now();shared_work.pending=true;
                    require(ggml_backend_graph_compute_async(p.backend,storage.shared_graph->graph)==GGML_STATUS_SUCCESS,"async batched shared FFN submission failed");
                    shared_submit_ms+=1000*Impl::elapsed(started);
                }
                if(p.cfg.route_observer)p.cfg.route_observer(layer,n,ids,weights);
                const std::string prefix="blk."+std::to_string(layer)+".ffn_";
                auto scale=[&](const char * role){auto tensor=p.host.maybe(prefix+role+"_exps.scale");return tensor?static_cast<const float *>(tensor->data):nullptr;};
                auto gate_weight=p.host.get(prefix+"gate_exps.weight"),up_weight=p.host.get(prefix+"up_exps.weight"),down_weight=p.host.get(prefix+"down_exps.weight");
                const auto cpu_started=Impl::Clock::now();
                bool cached=false;
                if(gpu_cached){
                    const auto hybrid_started=Impl::Clock::now();
                    cached=p.gpu_experts->forward_cached_batched(layer,n,input,ids,weights,
                        [&](const std::vector<uint8_t> & mask,std::vector<float> & terms){
                            const auto misses_started=Impl::Clock::now();
                            state.cpu->forward_masked(gate_weight,up_weight,down_weight,scale("gate"),scale("up"),scale("down"),input,ids,weights,mask,terms);
                            cpu_ffn_ms+=1000*Impl::elapsed(misses_started);
                        },hidden);
                    hybrid_ms+=1000*Impl::elapsed(hybrid_started);
                }
                if(!cached){
                    const auto misses_started=Impl::Clock::now();
                    state.cpu->forward(gate_weight,up_weight,down_weight,scale("gate"),scale("up"),scale("down"),input,ids,weights,hidden);
                    cpu_ffn_ms+=1000*Impl::elapsed(misses_started);
                }
                // Whole routed wall includes optional hot-seed prime, GPU joins,
                // CPU misses and original-rank merge, not merely the CPU callback.
                cpu_ms+=1000*Impl::elapsed(cpu_started);
                if(shared_overlap){
                    const auto started=Impl::Clock::now();shared_work.join();shared=Graph::read(packed_shared);
                    shared_wait_ms+=1000*Impl::elapsed(started);
                }
                for(size_t i=0;i<hidden.size();++i){const float ff=hidden[i]+shared[i];hidden[i]=ff+residual[i];}
            }
            if(!g.routed(layer))dense_ms+=1000*Impl::elapsed(dense_started);
            const auto feature=std::find(p.cfg.draft_feature_layers.begin(),p.cfg.draft_feature_layers.end(),layer+1);
            if(feature!=p.cfg.draft_feature_layers.end()){
                const size_t slot=size_t(feature-p.cfg.draft_feature_layers.begin());
                const int skip=n-p.captured_rows;
                for(int row=skip;row<n;++row)std::copy_n(hidden.data()+size_t(row)*size_t(g.embd),size_t(g.embd),
                    p.captured_features.data()+size_t(row-skip)*feature_width+slot*size_t(g.embd));
            }
            if(!persistent)owner.reset(); // retain the V2 transient memory lifetime
        }
        // Preserve the ordinary single-query head's RMS and MMV dispatch too.
        {
            if(!state.head) {
                state.head=std::make_unique<Graph>(p.backend,p.cfg,first,nullptr,4*1024*1024,1024);
                auto & b=*state.head;ggml_tensor * packed=nullptr;
                b.dspark_view_inputs=view_inputs;
                b.dspark_coalesce_inputs=coalesce_inputs;
                state.head_inputs.resize(size_t(n));
                std::vector<ggml_tensor *> normalized;
                for(int row=0;row<n;++row){
                    auto input=state.head_inputs[size_t(row)]=shared_inputs?b.borrow(state.hidden_rows[size_t(row)]):b.input(GGML_TYPE_F32,g.embd,1);
                    normalized.push_back(rms(b,input,p.dense->get("output_norm.weight"),g.rms_eps));
                }
                if(batch_ops)normalized=dspark_ops_rms_rows(b.c,state.head_inputs,p.dense->get("output_norm.weight"),g.rms_eps);
                if(batch_dense){
                    const auto outputs=dspark_projection(b,p.dense->get("output.weight"),normalized);
                    packed=dspark_join(b,outputs,0,n);
                }else for(int row=0;row<n;++row){
                    auto output=ggml_mul_mat(b.c,p.dense->get("output.weight"),normalized[size_t(row)]);
                    packed=packed?ggml_concat(b.c,packed,output,1):output;
                }
                state.head_output=packed;b.allocate({packed});
            }
            auto & b=*state.head;
            if(shared_inputs)ggml_backend_tensor_set(state.hidden_matrix,hidden.data(),0,hidden.size()*sizeof(float));
            else for(int row=0;row<n;++row)ggml_backend_tensor_set(state.head_inputs[size_t(row)],hidden.data()+size_t(row)*size_t(g.embd),0,size_t(g.embd)*sizeof(float));
            b.compute();logits=Graph::read(state.head_output);
        }
        ++state.rounds;
        if(p.cfg.profile&&(replanned||state.rounds==1)) {
            size_t free_bytes=0,total_bytes=0;
            ggml_backend_dev_memory(ggml_backend_get_device(p.backend),&free_bytes,&total_bytes);
            int main_nodes=0,max_main_nodes=0,shared_nodes=0;
            for(const auto & layer:state.layers)if(layer){
                if(layer->graph){const int nodes=ggml_graph_n_nodes(layer->graph->graph);main_nodes+=nodes;max_main_nodes=std::max(max_main_nodes,nodes);}
                if(layer->shared_graph)shared_nodes+=ggml_graph_n_nodes(layer->shared_graph->graph);
            }
            std::fprintf(stderr,"M31 verify-state coalesce_inputs=%d batch_msa=%d packed_routed=%d view_inputs=%d shared_overlap=%d batch_ops=%d shared_inputs=%d batch_dense=%d persistent=%d pos=%lld n=%d workspace_bytes=%zu free_bytes=%zu total_bytes=%zu rounds=%llu main_nodes=%d max_main_nodes=%d shared_nodes=%d\n",int(coalesce_inputs),int(batch_msa),int(packed_routed),int(view_inputs),int(shared_overlap),int(batch_ops),int(shared_inputs),int(batch_dense),int(persistent),(long long)first,n,state.workspace_bytes(),free_bytes,total_bytes,(unsigned long long)state.rounds,main_nodes,max_main_nodes,shared_nodes);
        }
        p.verified_logits=logits;
        p.hidden=std::move(hidden);p.pos=first+n;p.last_n=n;p.poisoned=false;
        if(p.cfg.profile)std::fprintf(stderr,"M31 verify pos=%lld n=%d dense_ms=%.3f cpu_ms=%.3f total_ms=%.3f build_ms=%.3f fill_ms=%.3f upload_ms=%.3f compute_ms=%.3f read_ms=%.3f hybrid_ms=%.3f cpu_ffn_ms=%.3f shared_submit_ms=%.3f shared_wait_ms=%.3f reconfigure_ms=%.3f reconfigures=%llu\n",
            (long long)first,n,dense_ms,cpu_ms,1000*Impl::elapsed(verify_started),build_ms,fill_ms,upload_ms,compute_ms,read_ms,hybrid_ms,cpu_ffn_ms,shared_submit_ms,shared_wait_ms,reconfigure_ms,(unsigned long long)state.reconfigures);
        if(p.cfg.profile&&gpu_cached){const auto & s=p.gpu_experts->stats();
            std::fprintf(stderr,"M31 verify-cache cumulative_calls=%llu routes=%llu hits=%llu hit_rate=%.6f seed_ms=%.3f wall_ms=%.3f\n",
                (unsigned long long)s.cached_batch_calls,(unsigned long long)s.cached_batch_routes,(unsigned long long)s.cached_batch_hits,
                double(s.cached_batch_hits)/double(std::max<uint64_t>(1,s.cached_batch_routes)),1000*s.cached_batch_seed_seconds,1000*s.cached_batch_wall_seconds);
            p.gpu_experts->print_cached_batch_profile("verify");
        }
        return true;
    }catch(const std::exception & e){error=e.what();return false;}
}
