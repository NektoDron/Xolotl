#pragma once
// Private Engine implementation header, included after Graph is defined.
// One current K/per-query-nk geometry. Buffers survive speculative rollback;
// reset/release destroys graphs BEFORE their shared inputs and immutable KV owners.
struct DSparkVerifyState {
    struct QueryInputs {
        ggml_tensor * hidden=nullptr,*positions=nullptr,*mask=nullptr,*bias=nullptr;
        int64_t nk=0;
    };
    struct Layer {
        std::unique_ptr<Graph> graph;
        // Reverse destruction keeps the shared consumer/captures alive only
        // while the main graph's output norm remains owned.
        std::unique_ptr<Graph> shared_graph;
        std::unique_ptr<DSparkPackedRoutes> packed_routes;
        std::vector<QueryInputs> inputs;
        ggml_tensor * norm=nullptr,*residual=nullptr,*shared=nullptr;
        ggml_tensor * ids=nullptr,*weights=nullptr,*output=nullptr;
    };
    std::vector<int64_t> geometry;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Graph> head;
    std::vector<ggml_tensor *> head_inputs;
    ggml_tensor * head_output=nullptr;
    std::unique_ptr<CpuBatchedVerify> cpu;
    std::vector<float> packed_scratch;
    DSparkPackedRoutesHost packed_host;
    // Optional once-per-verification attention values and one H2D hidden matrix
    // per layer. Every layer graph borrows leaf aliases, never producing graphs.
    std::vector<std::unique_ptr<AttentionInputs>> attention;
    ggml_context * hidden_context=nullptr;
    ggml_backend_buffer_t hidden_buffer=nullptr;
    ggml_tensor * hidden_matrix=nullptr,*position_matrix=nullptr;
    std::vector<ggml_tensor *> hidden_rows;
    uint64_t rounds=0,reconfigures=0;
    void clear_geometry() {
        head.reset();head_inputs.clear();head_output=nullptr;
        layers.clear(); // Graph destructors clear captures while all source buffers live.
        attention.clear();hidden_rows.clear();hidden_matrix=nullptr;position_matrix=nullptr;
        if(hidden_buffer){ggml_backend_buffer_free(hidden_buffer);hidden_buffer=nullptr;}
        if(hidden_context){ggml_free(hidden_context);hidden_context=nullptr;}
    }
    ~DSparkVerifyState(){clear_geometry();}
    bool reconfigure(const std::vector<int64_t> & next,int layer_count) {
        if(geometry==next)return false;
        clear_geometry();layers.resize(size_t(layer_count));geometry=next;++reconfigures;
        return true;
    }
    void prepare_shared(ggml_backend_t backend,const Geometry & g,int n,int64_t first,bool causal_f16,bool packed_positions=false) {
        if(hidden_buffer)return;
        hidden_context=ggml_init({128*1024,nullptr,true});require(hidden_context,"cannot allocate verifier input metadata");
        hidden_matrix=ggml_new_tensor_2d(hidden_context,GGML_TYPE_F32,g.embd,n);ggml_set_input(hidden_matrix);
        if(packed_positions){position_matrix=ggml_new_tensor_1d(hidden_context,GGML_TYPE_I32,n);ggml_set_input(position_matrix);}
        hidden_buffer=ggml_backend_alloc_ctx_tensors_from_buft(hidden_context,ggml_backend_get_default_buffer_type(backend));
        require(hidden_buffer,"cannot allocate verifier shared hidden input");
        for(int row=0;row<n;++row){
            auto input=ggml_new_tensor_2d(hidden_context,GGML_TYPE_F32,g.embd,1);ggml_set_input(input);
            require(ggml_backend_tensor_alloc(hidden_buffer,input,static_cast<uint8_t *>(hidden_matrix->data)+size_t(row)*size_t(g.embd)*sizeof(float))==GGML_STATUS_SUCCESS,"cannot bind verifier hidden row");
            hidden_rows.push_back(input);
            attention.push_back(std::make_unique<AttentionInputs>(backend,g,1,kv_length(first+row+1,g),causal_f16));
            if(position_matrix){
                auto position=ggml_new_tensor_1d(hidden_context,GGML_TYPE_I32,1);ggml_set_input(position);
                require(ggml_backend_tensor_alloc(hidden_buffer,position,static_cast<uint8_t *>(position_matrix->data)+size_t(row)*sizeof(int32_t))==GGML_STATUS_SUCCESS,"cannot bind verifier shared position row");
                attention.back()->positions=position;
            }
        }
    }
    size_t workspace_bytes() const {
        size_t bytes=0;
        for(const auto & layer:layers)if(layer){
            if(layer->graph)bytes+=ggml_gallocr_get_buffer_size(layer->graph->allocator,0);
            if(layer->shared_graph)bytes+=ggml_gallocr_get_buffer_size(layer->shared_graph->allocator,0);
        }
        if(head)bytes+=ggml_gallocr_get_buffer_size(head->allocator,0);
        if(hidden_buffer)bytes+=ggml_backend_buffer_get_size(hidden_buffer);
        for(const auto & input:attention)if(input)bytes+=ggml_backend_buffer_get_size(input->buffer);
        return bytes;
    }
};
