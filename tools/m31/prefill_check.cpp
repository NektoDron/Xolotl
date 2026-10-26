// Small original-byte/event ownership probe; --backend cuda requires aictl.
#include "prefill_pipeline.hpp"
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
            for(size_t batch=0;batch<batches;++batch) {
                // No host synchronization between batches. begin_batch must
                // protect previous consumers before these same slots are reused.
                uploads.begin_batch();
                for(size_t slot=0;slot<slot_count;++slot) {
                    std::array<uint8_t,187> gate{};
                    std::array<uint8_t,379> down{};
                    for(size_t i=0;i<gate.size();++i) gate[i]=uint8_t(batch*19+slot*53+i);
                    for(size_t i=0;i<down.size();++i) down[i]=uint8_t(batch*11+slot*71+i*3);
                    std::vector<uint8_t> golden(bytes,0);
                    std::memcpy(golden.data()+17,gate.data(),gate.size());
                    std::memcpy(golden.data()+2001,down.data(),down.size());
                    expected.push_back(std::move(golden));
                    uploads.enqueue(slots[slot],{{17,gate.data(),gate.size()},{2001,down.data(),down.size()}});
                    // Simulate the next arena packing job reusing its source.
                    gate.fill(0xff); down.fill(0xff);
                }
                uploads.finish();
                graphs.push_back(std::make_unique<CopyGraph>(backend,slots));
                check(ggml_backend_graph_compute_async(backend,graphs.back()->graph)==GGML_STATUS_SUCCESS,
                      "asynchronous consumer copy graph failed");
            }
            uploads.synchronize();
            std::vector<uint8_t> got(bytes);
            size_t compared=0;
            for(const auto & graph:graphs) for(auto output:graph->outputs) {
                ggml_backend_tensor_get(output,got.data(),0,bytes);
                check(got==expected[compared],"slot bytes/padding/source or consumer lifetime mismatch");
                ++compared;
            }
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
            uploads.finish(); uploads.synchronize();
            const auto & s=uploads.stats();
            check(s.uploads==batches*slot_count && s.staging_bytes==2*bytes,"bounded staging/counter mismatch");
            if(kind=="cuda") check(s.asynchronous && s.pinned && s.staging_reuses>0,"GPU did not exercise event pipeline");
            else check(!s.asynchronous,"CPU fallback unexpectedly asynchronous");
            result={{"backend",kind},{"byte_parity",true},{"compared_slots",compared},
                    {"compared_bytes",compared*bytes},{"padding_zeroed",true},
                    {"consumer_overwrite_protected",true},{"plan_checks",7},
                    {"asynchronous",s.asynchronous},{"pinned",s.pinned},
                    {"staging_bytes",s.staging_bytes},{"staging_reuses",s.staging_reuses}};
#ifdef M31_PREFILL_HAVE_CUDA
            if(kind=="cuda") ggml_backend_cuda_graph_cache_clear(backend);
#endif
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
        std::cerr<<"prefill_check: "<<e.what()<<'\n';
        return 1;
    }
}
