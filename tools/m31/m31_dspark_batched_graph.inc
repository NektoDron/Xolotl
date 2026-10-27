// Graph-build fragment inside the optional verifier. All elementwise reductions,
// MSA selection and FA retain n1 shapes. Only marked dense projections group<=4.
            std::vector<ggml_tensor *> attn_norms(size_t(n),nullptr);
            const std::string prefix="blk."+std::to_string(layer)+".";
            for(int row=0;row<n;++row){
                auto & in=inputs[size_t(row)];in.nk=kv_length(first+row+1,g);
                if(shared_inputs){
                    auto & source=*state.attention[size_t(row)];
                    in.hidden=b.borrow(state.hidden_rows[size_t(row)]);in.positions=b.borrow(source.positions);
                    in.mask=b.borrow(source.mask);in.bias=b.borrow(source.bias);
                }else{
                    in.hidden=b.input(GGML_TYPE_F32,g.embd,1);in.positions=b.input(GGML_TYPE_I32,1);
                    in.mask=b.input(p.causal_f16?GGML_TYPE_F16:GGML_TYPE_F32,in.nk,1,1);
                    in.bias=b.input(GGML_TYPE_F32,in.nk/g.block,1,1);
                }
                attn_norms[size_t(row)]=rms(b,in.hidden,p.dense->get(prefix+"attn_norm.weight"),g.rms_eps);
            }
            if(batch_ops){std::vector<ggml_tensor *> hidden_rows;for(const auto & in:inputs)hidden_rows.push_back(in.hidden);
                attn_norms=dspark_ops_rms_rows(c,hidden_rows,p.dense->get(prefix+"attn_norm.weight"),g.rms_eps);}
            std::vector<Impl::ProjectedAttention> projected(size_t(n),Impl::ProjectedAttention{});
            auto qkv_bundle=p.dense->maybe(prefix+"m31_qkv.weight");
            if(qkv_bundle){
                const auto joined=dspark_projection(b,qkv_bundle,attn_norms);
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];auto value=joined[size_t(row)];
                    a.q=ggml_reshape_3d(c,dspark_column(b,value,0,g.head_dim*g.heads),g.head_dim,g.heads,1);
                    a.k=ggml_reshape_3d(c,dspark_column(b,value,0,g.head_dim*g.kv_heads,g.head_dim*g.heads),g.head_dim,g.kv_heads,1);
                    a.v=ggml_reshape_3d(c,dspark_column(b,value,0,g.head_dim*g.kv_heads,g.head_dim*(g.heads+g.kv_heads)),g.head_dim,g.kv_heads,1);
                }
            }else{
                const auto q=dspark_projection(b,p.dense->get(prefix+"attn_q.weight"),attn_norms);
                const auto k=dspark_projection(b,p.dense->get(prefix+"attn_k.weight"),attn_norms);
                const auto v=dspark_projection(b,p.dense->get(prefix+"attn_v.weight"),attn_norms);
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];
                    a.q=ggml_reshape_3d(c,q[size_t(row)],g.head_dim,g.heads,1);
                    a.k=ggml_reshape_3d(c,k[size_t(row)],g.head_dim,g.kv_heads,1);
                    a.v=ggml_reshape_3d(c,v[size_t(row)],g.head_dim,g.kv_heads,1);
                }
            }
            auto index_bundle=p.dense->maybe(prefix+"m31_index_qk.weight");
            if(index_bundle){
                const auto joined=dspark_projection(b,index_bundle,attn_norms);
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];auto value=joined[size_t(row)];
                    a.iq=ggml_reshape_3d(c,dspark_column(b,value,0,g.index_dim*g.index_heads),g.index_dim,g.index_heads,1);
                    a.ik=ggml_reshape_3d(c,dspark_column(b,value,0,g.index_dim,g.index_dim*g.index_heads),g.index_dim,1,1);
                }
            }else{
                const auto iq=dspark_projection(b,p.dense->get(prefix+"indexer.q_proj.weight"),attn_norms);
                const auto ik=dspark_projection(b,p.dense->get(prefix+"indexer.k_proj.weight"),attn_norms);
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];
                    a.iq=ggml_reshape_3d(c,iq[size_t(row)],g.index_dim,g.index_heads,1);
                    a.ik=ggml_reshape_3d(c,ik[size_t(row)],g.index_dim,1,1);
                }
            }
            if(batch_ops){
                std::vector<ggml_tensor *> q,k,iq,ik;
                for(const auto & item:projected){q.push_back(item.q);k.push_back(item.k);iq.push_back(item.iq);ik.push_back(item.ik);}
                auto positions=b.borrow(state.position_matrix);
                q=dspark_ops_norm_rope_rows(c,q,p.dense->get(prefix+"attn_q_norm.weight"),g.rms_eps,positions,g);
                k=dspark_ops_norm_rope_rows(c,k,p.dense->get(prefix+"attn_k_norm.weight"),g.rms_eps,positions,g);
                iq=dspark_ops_norm_rope_rows(c,iq,p.dense->get(prefix+"indexer.q_norm.weight"),g.rms_eps,positions,g);
                ik=dspark_ops_norm_rope_rows(c,ik,p.dense->get(prefix+"indexer.k_norm.weight"),g.rms_eps,positions,g);
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];a.q=q[size_t(row)];a.k=k[size_t(row)];a.iq=iq[size_t(row)];a.ik=ik[size_t(row)];a.encoded=true;}
            }
            if(batch_msa){
                auto & kv=p.cache[size_t(layer)];
                // All queries in this layer already have their normalized/RoPE
                // values. Expand every n1 write BEFORE any shared keys CAST.
                for(int row=0;row<n;++row){auto & a=projected[size_t(row)];auto positions=inputs[size_t(row)].positions;
                    for(const auto & item:{std::make_pair(kv.k,a.k),std::make_pair(kv.v,a.v)}){
                        auto dst=ggml_reshape_2d(c,item.first,g.head_dim*g.kv_heads,p.capacity);
                        auto src=ggml_reshape_2d(c,item.second,g.head_dim*g.kv_heads,1);
                        ggml_build_forward_expand(b.graph,ggml_set_rows(c,dst,src,positions));
                    }
                    ggml_build_forward_expand(b.graph,ggml_set_rows(c,kv.index,ggml_reshape_2d(c,a.ik,g.index_dim,1),positions));
                    a.kv_written=true;
                }
                std::map<int64_t,ggml_tensor *> keys_by_nk;
                for(int first_row=0;first_row<n;){
                    const int64_t nk=inputs[size_t(first_row)].nk;
                    int count=1;while(count<4&&first_row+count<n&&inputs[size_t(first_row+count)].nk==nk)++count;
                    auto & keys=keys_by_nk[nk];
                    if(!keys)keys=ggml_cast(c,ggml_view_2d(c,kv.index,g.index_dim,nk,kv.index->nb[1],0),GGML_TYPE_F32);
                    std::vector<ggml_tensor *> queries,masks,biases;
                    for(int i=0;i<count;++i){const int row=first_row+i;queries.push_back(projected[size_t(row)].iq);masks.push_back(inputs[size_t(row)].mask);biases.push_back(inputs[size_t(row)].bias);}
                    const auto selected=dspark_msa_select_rows(c,keys,queries,masks,biases,g);
                    for(int i=0;i<count;++i)projected[size_t(first_row+i)].selected_idx=selected[size_t(i)];
                    first_row+=count;
                }
            }
            std::vector<ggml_tensor *> attentions;attentions.reserve(size_t(n));
            for(int row=0;row<n;++row){auto & a=projected[size_t(row)];auto & in=inputs[size_t(row)];
                a.norm=attn_norms[size_t(row)];a.defer_output=true;
                attentions.push_back(p.attention(b,layer,first+row,1,in.hidden,in.positions,in.mask,in.bias,true,&a));
            }
            const auto attention_outputs=dspark_projection(b,p.dense->get(prefix+"attn_output.weight"),attentions);
            std::vector<ggml_tensor *> residuals(size_t(n),nullptr),norms(size_t(n),nullptr);
            for(int row=0;row<n;++row){
                residuals[size_t(row)]=ggml_add(c,attention_outputs[size_t(row)],inputs[size_t(row)].hidden);
                norms[size_t(row)]=rms(b,residuals[size_t(row)],p.dense->get(prefix+"ffn_norm.weight"),g.rms_eps);
            }
            if(batch_ops)norms=dspark_ops_rms_rows(c,residuals,p.dense->get(prefix+"ffn_norm.weight"),g.rms_eps);
            if(!g.routed(layer)){
                const auto outputs=dspark_ffn(b,*p.dense,norms,prefix,"",batch_ops);
                for(int row=0;row<n;++row)join(packed_output,ggml_add(c,outputs[size_t(row)],residuals[size_t(row)]));
            }else{
                std::vector<ggml_tensor *> shared;
                if(!shared_overlap)shared=dspark_ffn(b,*p.dense,norms,prefix,"_shexp",batch_ops);
                const auto route_logits=dspark_projection(b,p.dense->get(prefix+"ffn_gate_inp.weight"),norms);
                DSparkBatchedRouterRows routed;
                if(batch_ops)routed=dspark_ops_router_rows(c,route_logits,p.dense->get(prefix+"exp_probs_b.bias"),g);
                for(int row=0;row<n;++row){
                    ggml_tensor * ids=nullptr,*weights=nullptr;
                    if(batch_ops){ids=routed.ids[size_t(row)];weights=routed.weights[size_t(row)];}
                    else {
                    auto probabilities=ggml_sigmoid(c,route_logits[size_t(row)]);
                    ids=ggml_top_k(c,ggml_add(c,probabilities,p.dense->get(prefix+"exp_probs_b.bias")),int(g.used));
                    weights=ggml_get_rows(c,ggml_reshape_3d(c,probabilities,1,g.experts,1),ids);
                    weights=ggml_reshape_2d(c,weights,g.used,1);
                    if(g.norm_routes){auto sum=ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY);weights=ggml_div(c,weights,sum);}
                    if(g.route_scale!=1)weights=ggml_scale(c,weights,g.route_scale);
                    }
                    join(packed_norm,norms[size_t(row)]);join(packed_residual,residuals[size_t(row)]);
                    if(!shared_overlap)join(packed_shared,shared[size_t(row)]);
                    join(packed_ids,ggml_reshape_2d(c,ids,g.used,1));join(packed_weights,weights);
                }
            }
