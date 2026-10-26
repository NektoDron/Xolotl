// Compact role-span byte/event ownership probe; --backend cuda requires aictl.
#include "prefill_pipeline.hpp"
#include "prefill_background.hpp"
#include "prefill_plan.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#ifdef M31_PREFILL_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include "json.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <string>

using namespace strata::m31;
using json = nlohmann::json;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

struct CopyGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> outputs;
    CopyGraph(ggml_backend_t stream, const std::vector<ggml_tensor *> & sources) : backend(stream) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        check(ctx, "copy graph metadata allocation");
        graph = ggml_new_graph_custom(ctx, 256, false);
        for (auto src : sources) {
            auto out = ggml_dup(ctx, src);
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
            outputs.push_back(out);
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        check(allocator && ggml_gallocr_alloc_graph(allocator, graph), "copy graph allocation");
    }
    ~CopyGraph() {
        if (backend) ggml_backend_synchronize(backend);
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }
};

static void plan_tests() {
    Geometry g;
    g.embd=6144; g.expert_ff=3072; g.dense_ff=9216; g.heads=48;
    g.head_dim=128; g.index_heads=4; g.index_dim=128; g.experts=128; g.used=4; g.block=128;
    PrefillBudget b;
    b.free_after_resident=4ULL<<30; b.streaming_slots=128; b.context=65536;
    b.remaining=6000;
    auto p=plan_prefill(g,b);
    check(p.chunk==2048 && p.padded_keys==2048 && p.worst_distinct_experts==128, "large full-layer batch plan");
    b.position=32767;
    auto long_p=plan_prefill(g,b);
    check(long_p.chunk>0 && long_p.chunk<2048 && long_p.chunk%128==0 && long_p.scratch_estimate>p.scratch_estimate,
          "long-context MSA budget shrinks chunk");
    b.position=0; b.streaming_slots=7;
    p=plan_prefill(g,b);
    check(p.chunk==1 && p.worst_distinct_experts==4, "small slot pool worst-case union bound");
    b.streaming_slots=3;
    check(plan_prefill(g,b).chunk==0, "cannot alter top-k to fit too few slots");
    b.streaming_slots=128; b.remaining=73;
    p=plan_prefill(g,b);
    check(p.chunk==73 && p.padded_keys==256, "short tail and exact pad256");
    b.free_after_resident=1;
    check(plan_prefill(g,b).chunk==0, "fixed-reserve refusal");
    bool rejected=false;
    try { (void)prefill_product(UINT64_MAX,2); } catch (const std::exception &) { rejected=true; }
    check(rejected,"budget multiplication overflow");
    // New larger windows: one layer stream can serve a complete ~4K request.
    b.free_after_resident=24ULL<<30; b.streaming_slots=128;
    b.max_chunk=8192; b.context=32768; b.position=0; b.remaining=8192;
    p=plan_prefill(g,b);
    check(p.chunk==8192 && p.worst_distinct_experts==128,"8192-row layer-batch proposal");
    b.position=24576; b.remaining=8192; b.free_after_resident=4ULL<<30;
    p=plan_prefill(g,b);
    check(p.chunk>0 && p.chunk<8192 && p.chunk%128==0,"large-chunk long-context shrink");
    b.max_chunk=8193;
    rejected=false;
    try { (void)plan_prefill(g,b); } catch(const std::exception &) { rejected=true; }
    check(rejected,"8193-row cap rejected");
    b.max_chunk=8192; b.context=65536; b.position=57344; b.remaining=8192;
    b.free_after_resident=8ULL<<30; b.attention_query_tile=0;
    const auto whole_query=plan_prefill(g,b);
    check(whole_query.chunk<8192,"whole score cannot accept int32-overflowing8192x64K");
    b.attention_query_tile=512; b.f16_mask=true;
    const auto tiled=plan_prefill(g,b);
    check(tiled.chunk==8192 && tiled.attention_query_rows==512 && tiled.worst_distinct_experts==128,
          "query tiles must retain full8192 MoE batch within8GiB");
    check(tiled.persistent_scratch_estimate>=(2ULL<<30) && tiled.scratch_estimate<(8ULL<<30),
          "full causal storage must not shrink with query tiles");
    b.attention_query_tile=8193; rejected=false;
    try {(void)plan_prefill(g,b);}catch(const std::exception&){rejected=true;}
    check(rejected,"invalid attention query tile accepted");
    // Actual M3.1 projection geometry: direct F16 TOP_K_MASK plus an in-place
    // score ADD admits whole8192 queries at32K keys without changing dot shape.
    g.heads=64;g.kv_heads=4;g.dense_ff=12288;g.shared=1;
    b.context=40000;b.position=24576;b.remaining=8192;b.max_chunk=8192;
    b.free_after_resident=10ULL<<30;b.reserve_bytes=1ULL<<30;b.extra_scratch_bytes=1ULL<<30;
    b.attention_query_tile=0;b.top_k_mask=true;b.inplace_score=true;b.f16_mask=true;
    const auto direct=plan_prefill(g,b);
    check(direct.chunk==8192 && direct.attention_query_rows==8192,
          "direct whole-query8192 proposal should fit10GiB and2GiB reserve");
    check(direct.score_stage_estimate==(4ULL<<30) && direct.mask_stage_estimate==(2ULL<<30),
          "actual4head32Kx8K stage storage differs");
    b.inplace_score=false;
    check(plan_prefill(g,b).chunk<8192,"two full score buffers must lower chunk at10GiB");
}

int main(int argc, char ** argv) {
    ggml_backend_t backend=nullptr;
    ggml_context * slots_ctx=nullptr;
    ggml_backend_buffer_t slot_buffer=nullptr;
    try {
        std::string kind="cpu", model;
        for(int i=1;i<argc;++i) {
            check(i+1<argc,"missing option argument");
            const std::string key=argv[i], value=argv[++i];
            if(key=="--backend") kind=value;
            else if(key=="--model") model=value;
            else throw std::runtime_error("unexpected option "+key);
        }
        if(kind=="cuda") {
            const char * session=std::getenv("M31_CARD_SESSION");
            const char * visible=std::getenv("CUDA_VISIBLE_DEVICES");
            check(session && std::string(session)=="1" && visible && std::string(visible)=="0",
                  "CUDA probe requires aictl/flock runner and CUDA_VISIBLE_DEVICES=0");
#ifdef M31_PREFILL_HAVE_CUDA
            backend=ggml_backend_cuda_init(0);
#else
            throw std::runtime_error("probe built without CUDA");
#endif
        } else if(kind=="cpu") backend=ggml_backend_cpu_init();
        else throw std::runtime_error("invalid backend");
        check(backend,"backend initialization");
        if(kind=="cpu") ggml_backend_cpu_set_n_threads(backend,2);
        plan_tests();
        constexpr size_t bytes=4096, slot_count=3, batches=17;
        slots_ctx=ggml_init({1024*1024,nullptr,true});
        check(slots_ctx,"slot metadata allocation");
        std::vector<ggml_tensor *> slots;
        for(size_t i=0;i<slot_count;++i) {
            auto t=ggml_new_tensor_1d(slots_ctx,GGML_TYPE_F32,int64_t(bytes/4));
            ggml_set_input(t);
            slots.push_back(t);
        }
        slot_buffer=ggml_backend_alloc_ctx_tensors(slots_ctx,backend);
        check(slot_buffer,"slot allocation");
        json result;
        {
            PrefillUploads uploads(backend,bytes,2);
            std::vector<std::unique_ptr<CopyGraph>> graphs;
            std::vector<std::vector<uint8_t>> expected;
            std::vector<std::vector<uint8_t>> current(slot_count, std::vector<uint8_t>(bytes, 0xa5));
            for (size_t slot=0; slot<slot_count; ++slot)
                ggml_backend_tensor_set(slots[slot], current[slot].data(), 0, bytes);
            uint64_t uploaded=0, raw=0;
            for(size_t batch=0;batch<batches;++batch) {
                // No host synchronization between batches. begin_batch must
                // protect previous consumers before these same slots are reused.
                uploads.begin_batch();
                for(size_t slot=0;slot<slot_count;++slot) {
                    std::array<uint8_t,187> gate{};
                    std::array<uint8_t,379> down{};
                    for(size_t i=0;i<gate.size();++i) gate[i]=uint8_t(batch*19+slot*53+i);
                    for(size_t i=0;i<down.size();++i) down[i]=uint8_t(batch*11+slot*71+i*3);
                    // Role geometry changes on every batch, like slots reused by
                    // different quant types/layers. Old nonzero data becomes new
                    // padding and must be reset, while all unreachable gaps remain.
                    const size_t off_gate=17+(batch%3)*23, off_down=1800+(batch%4)*41;
                    const size_t extent_gate=gate.size()+31+(batch%5)*7;
                    const size_t extent_down=down.size()+67+(batch%3)*11;
                    auto & golden=current[slot];
                    std::memset(golden.data()+off_gate,0,extent_gate);
                    std::memset(golden.data()+off_down,0,extent_down);
                    std::memcpy(golden.data()+off_gate,gate.data(),gate.size());
                    std::memcpy(golden.data()+off_down,down.data(),down.size());
                    expected.push_back(golden);
                    uploads.enqueue_spans(slots[slot],{
                        {off_gate,gate.data(),gate.size(),extent_gate},
                        {off_down,down.data(),down.size(),extent_down}});
                    uploaded+=extent_gate+extent_down;
                    raw+=gate.size()+down.size();
                    // Simulate the next arena packing job reusing its source.
                    gate.fill(0xff); down.fill(0xff);
                    // Publish one group and queue its consumer before uploading
                    // the next disjoint slot. CPU fallback returns no event.
                    const auto ready=uploads.checkpoint();
                    check((ready!=nullptr)==uploads.stats().asynchronous,"checkpoint event mode differs");
                    graphs.push_back(std::make_unique<CopyGraph>(backend,std::vector<ggml_tensor *>{slots[slot]}));
                    check(ggml_backend_graph_compute_async(backend,graphs.back()->graph)==GGML_STATUS_SUCCESS,
                          "checkpoint consumer copy graph failed");
                }
                uploads.finish();
            }
            uploads.synchronize();
            std::vector<uint8_t> got(bytes);
            size_t compared=0;
            for(const auto & graph:graphs) for(auto output:graph->outputs) {
                ggml_backend_tensor_get(output,got.data(),0,bytes);
                check(got==expected[compared],"slot bytes/padding/source or consumer lifetime mismatch");
                ++compared;
            }
            bool closed_checkpoint_rejected=false;
            try { (void)uploads.checkpoint(); } catch(const std::exception &) { closed_checkpoint_rejected=true; }
            check(closed_checkpoint_rejected,"checkpoint outside batch was accepted");
            // Rejected malformed parts must not damage state or consume staging.
            uploads.begin_batch();
            bool bad=false;
            try { uploads.enqueue(slots[0],{{bytes-1,got.data(),2}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"out-of-range source part was accepted");
            bad=false;
            try { uploads.enqueue(slots[0],{{0,got.data(),16},{8,got.data(),16}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"overlapping source parts were accepted");
            // Compact spans reject overlap even when only their padding overlaps.
            bad=false;
            try { uploads.enqueue_spans(slots[0],{{0,got.data(),8,16},{8,got.data(),8,16}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"overlapping compact padding was accepted");
            bad=false;
            try { uploads.enqueue_spans(slots[0],{{bytes-8,got.data(),8,16}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"out-of-bounds compact extent was accepted");
            bad=false;
            try { uploads.enqueue_spans(slots[0],{{0,got.data(),16,8}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"raw bytes larger than compact extent accepted");
            bad=false;
            try { uploads.enqueue_spans(slots[0],{{0,nullptr,8,16}}); }
            catch(const std::exception &) { bad=true; }
            check(bad,"nonempty compact null input accepted");
            uploads.finish(); uploads.synchronize();
            const auto & s=uploads.stats();
            check(s.upload_bytes==uploaded && s.packed_bytes==raw,"compact byte counters differ");
            check(s.upload_bytes<batches*slot_count*bytes,"compact DMA saved no bytes");
            check(s.uploads==batches*slot_count && s.staging_bytes==2*bytes,"bounded staging/counter mismatch");
            if(kind=="cuda") check(s.asynchronous && s.pinned && s.staging_reuses>0,"GPU did not exercise event pipeline");
            else check(!s.asynchronous,"CPU fallback unexpectedly asynchronous");
            result={{"backend",kind},{"byte_parity",true},{"compared_slots",compared},
                    {"compared_bytes",compared*bytes},{"backend_padding_zeroed",true},{"unread_gaps_preserved",true},{"upload_bytes",s.upload_bytes},{"original_slot_upload_bytes",batches*slot_count*bytes},
                    {"consumer_overwrite_protected",true},{"disjoint_group_checkpoints",batches*slot_count},{"plan_checks",17},
                    {"asynchronous",s.asynchronous},{"pinned",s.pinned},
                    {"staging_bytes",s.staging_bytes},{"staging_reuses",s.staging_reuses},
                    {"host_pack_seconds",s.pack_seconds},
                    {"host_staging_wait_seconds",s.staging_wait_seconds},
                    {"host_submit_seconds",s.submit_seconds}};
#ifdef M31_PREFILL_HAVE_CUDA
            if(kind=="cuda") ggml_backend_cuda_graph_cache_clear(backend);
#endif
        }
        {
            // The worker owns a separate backend/ring. While it uploads slot1/2,
            // the caller can consume unchanged slot0 through the main backend.
            BackgroundPrefillUploads background(backend,bytes);
            std::array<uint8_t,187> gate{}, up{}, down{};
            uint64_t checked=0, uploaded=0;
            for(size_t batch=0;batch<17;++batch) {
                gate.fill(uint8_t(batch+1));up.fill(uint8_t(batch+31));down.fill(uint8_t(batch+63));
                std::vector<BackgroundPrefillUploads::Job> jobs;
                for(size_t slot=1;slot<slot_count;++slot) jobs.push_back({slots[slot],{{
                    {17,gate.data(),gate.size(),224},
                    {1000,up.data(),up.size(),224},
                    {2001,down.data(),down.size(),256}}}});
                background.start(std::move(jobs));
                CopyGraph concurrent(backend,{slots[0]});
                check(ggml_backend_graph_compute_async(backend,concurrent.graph)==GGML_STATUS_SUCCESS,
                      "main consumer during background prefetch failed");
                const auto stats=background.wait();
                check(stats.uploads==2 && stats.batches==1 && stats.upload_bytes==2*(224+224+256),
                      "background statistics are not per-job deltas");
                uploaded+=stats.upload_bytes;
                std::vector<uint8_t> got(bytes);
                for(size_t slot=1;slot<slot_count;++slot) {
                    ggml_backend_tensor_get(slots[slot],got.data(),0,bytes);
                    check(std::all_of(got.begin()+17,got.begin()+17+187,[&](uint8_t v){return v==uint8_t(batch+1);}),
                          "background gate bytes differ");
                    check(std::all_of(got.begin()+17+187,got.begin()+17+224,[](uint8_t v){return !v;}),
                          "background padding differs");
                    ++checked;
                }
            }
            // A partial upload followed by malformed span must drain before
            // propagating, and the persistent worker must recover for a new job.
            const BackgroundPrefillUploads::Job valid{slots[1],{{
                {17,gate.data(),gate.size(),224},{1000,up.data(),up.size(),224},{2001,down.data(),down.size(),256}}}};
            auto invalid=valid; invalid.destination=slots[2];invalid.spans[0].extent=bytes;
            background.start({valid,invalid});
            bool failed=false;
            try {(void)background.wait();}catch(const std::exception &){failed=true;}
            check(failed && !background.active(),"background partial failure did not propagate/join");
            background.start({valid});
            check(background.wait().uploads==1,"background did not recover after partial upload");
            result["background_prefetch"]={{"checked_slots",checked},{"upload_bytes",uploaded},
                {"private_backend",true},{"main_disjoint_consumer",true},{"partial_failure_recovery",true}};
        }
        if(!model.empty()) {
            const auto file=strata::GgufModel::open(model);
            const auto g=read_geometry(file);
            size_t free=0,total=0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend),&free,&total);
            json proposals=json::array();
            for(int64_t ctx:{4096,32768,65536}) {
                // Offline examples use the same 6GiB expert pool and explicit
                // CUDA/context reserve; these are not measured allocator peaks.
                const uint64_t weights=g.other_bytes-g.embedding_bytes-g.expert_scale_bytes;
                const uint64_t kv=g.kv_bytes((ctx+255)/256*256);
                const uint64_t expert_pool=248ULL*25909632;
                const uint64_t resident=prefill_sum(prefill_sum(weights,kv),expert_pool);
                PrefillBudget b;
                b.free_after_resident=(32110ULL<<20)>resident?(32110ULL<<20)-resident:0;
                b.reserve_bytes=1ULL<<30; b.extra_scratch_bytes=256ULL<<20;
                b.streaming_slots=248; b.context=ctx; b.remaining=ctx; b.position=ctx/2;
                b.remaining=ctx-b.position;
                const auto p=plan_prefill(g,b);
                proposals.push_back({{"context",ctx},{"position",b.position},{"dense_weight_bytes",weights},
                    {"kv_f16_capacity_bytes",kv},{"expert_pool_bytes",expert_pool},
                    {"free_after_resident",b.free_after_resident},{"chunk",p.chunk},
                    {"padded_keys",p.padded_keys},{"scratch_estimate",p.scratch_estimate},
                    {"headroom_bytes",p.headroom_bytes},{"reason",p.reason}});
            }
            result["model_budget_proposals"]=proposals;
            result["budget_limits"]="F16 KV; fixed 32110MiB model card/248-slot 6GiB pool; scratch estimates, not measured peaks. Runtime should pass actual free bytes after allocations.";
        }
        ggml_backend_buffer_free(slot_buffer);slot_buffer=nullptr;
        ggml_free(slots_ctx);slots_ctx=nullptr;
        ggml_backend_free(backend);backend=nullptr;
        std::cout<<result.dump(2)<<'\n';
        return 0;
    } catch(const std::exception & e) {
        if(backend) ggml_backend_synchronize(backend);
        if(slot_buffer)ggml_backend_buffer_free(slot_buffer);
        if(slots_ctx)ggml_free(slots_ctx);
        if(backend)ggml_backend_free(backend);
        std::cerr<<"port5_prefill_check: "<<e.what()<<'\n';
        return 1;
    }
}
