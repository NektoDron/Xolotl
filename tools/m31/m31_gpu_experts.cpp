#include "m31_gpu_experts.hpp"
#include "cache_seed_rank.hpp"
#include "cache_seed_receipt.hpp"
#include "prefill_pipeline.hpp"
#include "prefill_background.hpp"
#include "prefill_route_history.hpp"
#include "shortprefill_cost.hpp"
#include "shortprefill_cpu_lane.hpp"
#include "prefill_layer_banks.hpp"
#include "gpu_native_gu.hpp"
#include "cpu_pack_spans.hpp"
#include "cached_decode_merge.hpp"
#include "cached_decode_merge_rank_outer.hpp"

#include "ggml-alloc.h"
#ifdef M31_CPU_QUANTS_CUDA
#include "gpu_cpu_quants.hpp"
#include "gpu_cached_exact.hpp"
#include "cpu_gpu_host_ops.hpp"
#include "cpu_gpu_host_executor.hpp"
#endif
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef M31_FAST_FFN_CUDA
#include "gpu_fast_ffn.hpp"
#endif
#ifdef M31_REGISTERED_ARENA_CUDA
#include "gpu_registered_arena.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::m31 {
namespace {
void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("M31 GPU experts: " + message);
}
void clear_captures(ggml_backend_t backend) {
#ifdef M31_ENGINE_CUDA
    ggml_backend_cuda_graph_cache_clear(backend);
#else
    (void) backend;
#endif
}
size_t aligned(size_t value, size_t alignment) {
    check(alignment && value <= std::numeric_limits<size_t>::max() - (alignment - 1), "alignment overflow");
    return (value + alignment - 1) / alignment * alignment;
}
size_t common_multiple(size_t a, size_t b) {
    check(a && b, "invalid stride multiple");
    const size_t divided = a / std::gcd(a, b);
    check(divided <= std::numeric_limits<size_t>::max() / b, "stride multiple overflow");
    return divided * b;
}
bool debug_expert_layer(int layer, int n) {
    const char * selected = std::getenv("M31_GPU_QUANTS_DEBUG_LAYER");
    if (!selected || !*selected) return layer == 3;
    if (std::strcmp(selected, "all") == 0) return n > 1;
    char * end = nullptr;
    const long target = std::strtol(selected, &end, 10);
    check(end != selected && end && !*end && target >= 0 && target < 1024,
          "M31_GPU_QUANTS_DEBUG_LAYER must be all or a layer index in0..1023");
    return layer == target;
}
bool debug_expert_intermediates() {
    const char * selected = std::getenv("M31_GPU_QUANTS_DEBUG_LAYER");
    return !selected || std::strcmp(selected, "all") != 0;
}
struct Context {
    ggml_context * value = nullptr;
    explicit Context(size_t bytes) : value(ggml_init({bytes, nullptr, true})) {
        check(value != nullptr, "metadata allocation failed");
    }
    ~Context() { if (value) ggml_free(value); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
};
} // namespace

struct GpuExperts::Impl {
    using Clock = std::chrono::steady_clock;
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now()-start).count();
    }
    struct Layer {
        std::array<ggml_type, 3> types{};
        std::array<const ExpertArena::Slice *, 3> sources{};
        std::array<std::vector<float>, 3> scales;
        std::array<bool, 3> has_scale{};
        std::array<size_t, 3> offsets{}, raw_bytes{}, allocation_bytes{};
        size_t span = 0;
    };
    struct Entry {
        size_t slot = 0;
        uint64_t last_used = 0;
    };
    Geometry g;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t pool = nullptr, staging = nullptr;
    std::unique_ptr<CpuPackSpans> pack_pool;
    std::unique_ptr<PrefillUploads> uploads;
    std::unique_ptr<BackgroundPrefillUploads> background;
    PrefillUploads::Copier copier;
    struct PendingPrefetch { uint64_t key; size_t slot; };
    std::vector<PendingPrefetch> pending_prefetch;
    std::vector<std::vector<int>> prefetch_order;
    std::unique_ptr<PrefillRouteHistory> prefetch_history;
    std::unique_ptr<Context> slot_metadata;
    ggml_tensor * pool_proxy = nullptr;
    std::vector<ggml_tensor *> slot_proxies;
    std::vector<size_t> free_slots;
    std::vector<uint8_t> pageable_staging;
    ggml_gallocr_t allocator = nullptr;
    std::vector<Layer> layers;
    std::map<uint64_t, Entry> entries;
    std::set<uint64_t> seeded_keys;
    std::set<uint64_t> decode_seeded_keys;
    std::vector<uint32_t> decode_seed_order;
    bool decode_seed_armed = false;
    std::set<uint64_t> active_read_keys;
    struct CachedDecodeGraph {
        std::unique_ptr<Context> context;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * output = nullptr;
        std::array<ggml_tensor *,3> scale_inputs{};
        std::array<std::vector<float>,3> scales;
        std::vector<int32_t> mapped;
        std::vector<float> gpu_terms, cpu_terms, missing_weights;
        std::vector<int32_t> missing_ids;
        std::vector<uint8_t> resident;
        size_t scratch = 0;
        ~CachedDecodeGraph() { if(allocator)ggml_gallocr_free(allocator); }
    };
    std::vector<std::unique_ptr<CachedDecodeGraph>> cached_decode_graphs;
    Stats stat;
    uint64_t clock = 0;
    int max_chunk = 0;
    bool ready = false;
    bool profile = false;
    bool compact_uploads = false;
    bool reuse_exact_buffers = false;
    bool parallel_reduce = false;
    bool ranked_prefetch = false;
    bool suppress_prefetch = false;
    std::unique_ptr<ShortPrefillCpuLane> short_cpu_lane;
    bool native_gu_concat = false;
    bool cached_decode_exact = false,cached_verify_enabled = false;
    bool layer_banks_requested = false;
    bool bank_prefetch_enabled = false;
#ifdef M31_REGISTERED_ARENA_CUDA
    std::unique_ptr<DirectLayerUploads> layer_banks;
    std::array<int,2> bank_layers{{-1,-1}};
#endif
    size_t prefetch_limit = 0;
    size_t upload_depth = 2;
#ifdef M31_GPU_PREFILL_TESTING
    bool test_upload_failure_fired = false;
#endif
#ifdef M31_CPU_QUANTS_CUDA
    std::unique_ptr<GpuCpuQuants> cpu_quants;
    std::unique_ptr<GpuCachedExact> cached_exact_worker,cached_batch_worker;
    uint64_t cached_rank_outer_calls=0;
    std::array<std::vector<float>, 3> exact_outputs;
    std::vector<float> exact_activation;
#endif
#ifdef M31_FAST_FFN_CUDA
    std::unique_ptr<GpuFastFfn> fast_ffn;
#endif
#ifdef M31_REGISTERED_ARENA_CUDA
    std::unique_ptr<GpuRegisteredArena> owned_registered_arena;
    const GpuRegisteredArena * registered_arena=nullptr;
#endif

    ~Impl() {
        short_cpu_lane.reset(); // Join a borrowed Engine CPU callback before teardown.
        // A worker borrows arena role spans and destination pool metadata.
        // Join it before any of those can be freed, including failure shutdown.
#ifdef M31_CPU_QUANTS_CUDA
        cached_batch_worker.reset(); // Both actual raw readers drain before pool/registration.
        cached_exact_worker.reset();
#endif
        background.reset();
#ifdef M31_REGISTERED_ARENA_CUDA
        layer_banks.reset(); // Copy streams and actual consumers drain before pool/registration.
#endif
#ifdef M31_FAST_FFN_CUDA
        // The raw fast stream borrows pool slots and uploader ready events.
        // Drain it before either borrowed resource can be destroyed.
        fast_ffn.reset();
#endif
        if (uploads) {
            const auto & u = uploads->stats();
            std::fprintf(stderr, "M31 prefill uploads: batches=%llu uploads=%llu upload_bytes=%llu staging_bytes=%zu reuses=%llu pinned=%d async=%d\n",
                         (unsigned long long)u.batches, (unsigned long long)u.uploads,
                         (unsigned long long)u.upload_bytes, u.staging_bytes,
                         (unsigned long long)u.staging_reuses, int(u.pinned), int(u.asynchronous));
            uploads.reset(); // Transfer/computation streams drain before slot tensors/pool die.
        }
#ifdef M31_REGISTERED_ARENA_CUDA
        // All DMA readers have drained; the borrowed RAM arena outlives us.
        owned_registered_arena.reset();
        registered_arena=nullptr;
#endif
        if (backend) {
            ggml_backend_synchronize(backend);
            clear_captures(backend);
        }
        if (allocator) ggml_gallocr_free(allocator);
        if (pool) ggml_backend_buffer_free(pool);
        if (staging) ggml_backend_buffer_free(staging);
    }
    uint64_t key(int layer, int expert) const {
        return uint64_t(layer) * uint64_t(g.experts) + uint64_t(expert);
    }
    size_t reserve_slot(const std::set<uint64_t> & protected_keys, bool required) {
        if (!free_slots.empty()) {
            const size_t slot=free_slots.back();
            free_slots.pop_back();
            return slot;
        }
        auto victim=entries.end();
        for (auto it=entries.begin();it!=entries.end();++it)
            if (!protected_keys.count(it->first) && !seeded_keys.count(it->first) && !decode_seeded_keys.count(it->first) && !active_read_keys.count(it->first) &&
                (victim==entries.end() || it->second.last_used<victim->second.last_used)) victim=it;
        if (victim==entries.end()) {
            check(!required, "all cache slots are protected; increase expert cache budget");
            return std::numeric_limits<size_t>::max();
        }
        const size_t slot=victim->second.slot;
        entries.erase(victim);
        ++stat.evictions;
        return slot;
    }
    void observe_routes(int layer, int n, const std::vector<uint8_t> & selected) {
        // Prediction affects only a speculative future upload. Actual routes
        // below still enter load() unconditionally, including history misses.
        if (prefetch_history) prefetch_history->remember(size_t(layer), n, selected);
        if (!profile) return;
        size_t required = 0, used = 0, unused = 0;
        const auto & raw = layers[size_t(layer)].raw_bytes;
        const uint64_t expert_bytes = uint64_t(raw[0]) + raw[1] + raw[2];
        for (auto value : selected) required += value != 0;
        for (const auto & pending : pending_prefetch) {
            if (pending.key / uint64_t(g.experts) != uint64_t(layer)) continue;
            const size_t expert = size_t(pending.key % uint64_t(g.experts));
            if (selected[expert]) ++used; else ++unused;
        }
        stat.required_experts += required;
        stat.required_bytes += required * expert_bytes;
        stat.prefetch_used_experts += used;
        stat.prefetch_unused_experts += unused;
        stat.prefetch_used_bytes += used * expert_bytes;
        stat.prefetch_unused_bytes += unused * expert_bytes;
        if (n > 1) std::fprintf(stderr,
            "M31 GPU route demand layer=%d n=%d required=%zu required_bytes=%llu predicted_used=%zu predicted_unused=%zu predicted_bytes=%llu history=%d\n",
            layer,n,required,(unsigned long long)(required*expert_bytes),used,unused,
            (unsigned long long)((used+unused)*expert_bytes),int(bool(prefetch_history)));
    }
    void leave_decode_seed() {
        if (decode_seed_order.empty()) return;
        // Decode-only pins never reduce the next prefill streaming capacity.
        decode_seeded_keys.clear();
        decode_seed_armed = true;
    }
    void prime_decode_seed() {
        if (!decode_seed_armed || decode_seed_order.empty()) return;
        const auto start=Clock::now();
        finish_prefetch();
        const uint64_t before=stat.upload_bytes;
        std::set<uint64_t> pins;
        for(auto id:decode_seed_order)pins.insert(id);
        try {
            if(uploads)uploads->begin_batch();
            for(auto id:decode_seed_order)load(int(id / g.experts),int(id % g.experts),pins);
            if(uploads){uploads->finish();uploads->synchronize();}
            ggml_backend_synchronize(backend);
            decode_seeded_keys=std::move(pins);decode_seed_armed=false;
            ++stat.decode_seed_primes;stat.decode_seed_upload_bytes+=stat.upload_bytes-before;
            stat.decode_seed_seconds+=elapsed(start);
            std::fprintf(stderr,"M31 decode-only seed: hot=%zu upload_bytes=%llu seconds=%.6f\n",
                decode_seeded_keys.size(),(unsigned long long)(stat.upload_bytes-before),elapsed(start));
        } catch (...) {
            // A partially uploaded seed cannot be exposed through stale cache metadata.
            ready=false; // Set before draining: a second CUDA error must stay terminal.
            try { if(uploads)uploads->synchronize();ggml_backend_synchronize(backend); } catch (...) {}
            throw;
        }
    }
    void begin_prefetch(int layer, int n, const std::set<uint64_t> & current_keys) {
        if (suppress_prefetch) return;
#ifdef M31_REGISTERED_ARENA_CUDA
        if(layer_banks){
            if(bank_prefetch_enabled&&n>1&&layer+1<g.layers&&g.routed(layer+1))ensure_bank(layer+1,true);
            return;
        }
#endif
        // Speculative full-next-layer work is worthwhile only for a substantial
        // layer batch. Router/histogram/compute shape of the current layer is unchanged.
        if (!background || n<128 || layer+1>=g.layers || !g.routed(layer+1)) return;
        check(pending_prefetch.empty() && !background->active(), "uncollected prefetch");
        const int next=layer+1;
        const auto & source=layers[size_t(next)];
        auto protected_keys=current_keys;
        // A warm next-layer entry must not be selected as another prefetch's victim.
        for (int expert=0;expert<g.experts;++expert)
            if (entries.count(key(next,expert))) protected_keys.insert(key(next,expert));
        std::vector<BackgroundPrefillUploads::Job> jobs;
        jobs.reserve(size_t(g.experts));
        const bool history_known = prefetch_history && prefetch_history->has(size_t(next));
        if (profile && history_known) ++stat.prefetch_history_batches;
        for (const int expert:prefetch_order[size_t(next)]) {
            if (prefetch_history && !prefetch_history->predicts(size_t(next), size_t(expert))) {
                if (profile) ++stat.prefetch_history_skipped_experts;
                continue;
            }
            if (prefetch_limit && jobs.size()>=prefetch_limit) break;
            const auto id=key(next,expert);
            if (entries.count(id)) continue;
            const auto slot=reserve_slot(protected_keys,false);
            if (slot==std::numeric_limits<size_t>::max()) break;
            pending_prefetch.push_back({id,slot});
            BackgroundPrefillUploads::Job job;
            job.destination=slot_proxies[slot];
            for (size_t r=0;r<3;++r) job.spans[r]={source.offsets[r],
                source.sources[r]->data+size_t(expert)*source.sources[r]->expert_stride,
                source.raw_bytes[r],source.allocation_bytes[r]};
            jobs.push_back(job);
        }
        if (!jobs.empty()) {
#ifdef M31_REGISTERED_ARENA_CUDA
            background->start(std::move(jobs),compact_uploads,registered_arena);
#else
            background->start(std::move(jobs),compact_uploads);
#endif
        }
    }
#ifdef M31_REGISTERED_ARENA_CUDA
    void ensure_bank(int layer,bool speculative) {
        const size_t bank=size_t(layer)&1;
        if(bank_layers[bank]==layer)return;
        check(layer_banks&&registered_arena,"layer banks require registered source arena");
        const auto &source=layers[size_t(layer)];
        std::vector<DirectLayerUploads::Job> jobs;
        jobs.reserve(size_t(g.experts));
        for(int expert=0;expert<g.experts;++expert){
            DirectLayerUploads::Job job;
            job.destination=slot_proxies[bank*size_t(g.experts)+size_t(expert)];
            for(size_t role=0;role<3;++role)job.spans[role]={source.offsets[role],
                source.sources[role]->data+size_t(expert)*source.sources[role]->expert_stride,
                source.raw_bytes[role],source.allocation_bytes[role]};
            jobs.push_back(job);
        }
        const auto before=layer_banks->stats();
        bank_layers[bank]=-1; // Partial submission must never publish valid weights.
        layer_banks->enqueue(bank,jobs,*registered_arena);
        bank_layers[bank]=layer;
        const auto &after=layer_banks->stats();
        const auto bytes=after.upload_bytes-before.upload_bytes;
        stat.upload_bytes+=bytes;stat.misses+=size_t(g.experts);
        stat.occupied_slots=size_t(g.experts)*size_t((bank_layers[0]>=0)+(bank_layers[1]>=0));
        if(speculative){++stat.prefetch_batches;stat.prefetched_experts+=size_t(g.experts);stat.prefetch_bytes+=bytes;}
    }
#endif
    void finish_prefetch() {
        if (!background || !background->active()) return;
        const auto wait_start=profile?Clock::now():Clock::time_point{};
        const auto copied=background->wait(); // Throws before publishing partially written slots.
        if (profile) stat.prefetch_wait_seconds+=elapsed(wait_start);
        for (const auto & item:pending_prefetch) entries.emplace(item.key,Entry{item.slot,++clock});
        stat.misses+=pending_prefetch.size();
        stat.upload_bytes+=copied.upload_bytes;
        ++stat.prefetch_batches;
        stat.prefetched_experts+=pending_prefetch.size();
        stat.prefetch_bytes+=copied.upload_bytes;
        stat.prefetch_pack_seconds+=copied.pack_seconds;
        stat.prefetch_staging_wait_seconds+=copied.staging_wait_seconds;
        stat.prefetch_submit_seconds+=copied.submit_seconds;
        stat.occupied_slots=entries.size();
        pending_prefetch.clear();
    }
    size_t load(int layer, int expert, const std::set<uint64_t> & protected_keys) {
        const uint64_t id = key(layer, expert);
        auto found = entries.find(id);
        if (found != entries.end()) {
            found->second.last_used = ++clock;
            ++stat.hits;
            return found->second.slot;
        }
        const size_t slot=reserve_slot(protected_keys,true);
        const auto & source = layers[size_t(layer)];
        const auto upload_start = profile ? Clock::now() : Clock::time_point{};
        if (uploads) {
            // Identical original role bytes and zero padding, packed into a bounded
            // two-buffer ring while the preceding slot is transferred on its stream.
            const auto spans = {
                PrefillUploads::Span{source.offsets[0], source.sources[0]->data + size_t(expert)*source.sources[0]->expert_stride, source.raw_bytes[0], source.allocation_bytes[0]},
                PrefillUploads::Span{source.offsets[1], source.sources[1]->data + size_t(expert)*source.sources[1]->expert_stride, source.raw_bytes[1], source.allocation_bytes[1]},
                PrefillUploads::Span{source.offsets[2], source.sources[2]->data + size_t(expert)*source.sources[2]->expert_stride, source.raw_bytes[2], source.allocation_bytes[2]},
            };
#ifdef M31_REGISTERED_ARENA_CUDA
            if (registered_arena) uploads->enqueue_registered_spans(slot_proxies[slot], spans, *registered_arena);
            else
#endif
            if (compact_uploads) uploads->enqueue_spans(slot_proxies[slot], spans);
            /* spans retain role padding even when fixed-slot alignment is skipped. */
            else uploads->enqueue(slot_proxies[slot], {
                {source.offsets[0], source.sources[0]->data + size_t(expert)*source.sources[0]->expert_stride, source.raw_bytes[0]},
                {source.offsets[1], source.sources[1]->data + size_t(expert)*source.sources[1]->expert_stride, source.raw_bytes[1]},
                {source.offsets[2], source.sources[2]->data + size_t(expert)*source.sources[2]->expert_stride, source.raw_bytes[2]},
            });
#ifdef M31_GPU_PREFILL_TESTING
            const char * inject = std::getenv("M31_GPU_PREFILL_TEST_FAIL_AFTER");
            if (inject && !test_upload_failure_fired && std::strtoull(inject, nullptr, 10) == stat.misses+1) {
                test_upload_failure_fired = true;
                throw std::runtime_error("M31 prefill injected upload failure");
            }
#endif
        } else {
            auto * stage = staging ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(staging)) : pageable_staging.data();
            const auto pack_start = profile ? Clock::now() : Clock::time_point{};
            // Includes each role's CUDA quant padding and slot-alignment padding.
            // External NVFP4 scales live in the graph, not in the byte stride.
            std::memset(stage, 0, stat.slot_stride);
            for (int r = 0; r < 3; ++r) {
                const auto * slice = source.sources[size_t(r)];
                std::memcpy(stage + source.offsets[size_t(r)],
                            slice->data + size_t(expert)*slice->expert_stride, source.raw_bytes[size_t(r)]);
            }
            if (profile) stat.pack_seconds += elapsed(pack_start);
            ggml_backend_tensor_set(slot_proxies[slot], stage, 0, stat.slot_stride);
        }
        if (profile) stat.upload_submit_seconds += elapsed(upload_start);
        bool compact_transfer = uploads && compact_uploads;
#ifdef M31_REGISTERED_ARENA_CUDA
        compact_transfer = compact_transfer || bool(registered_arena);
#endif
        stat.upload_bytes += compact_transfer ?
            std::accumulate(source.allocation_bytes.begin(), source.allocation_bytes.end(), size_t(0)) : stat.slot_stride;
        ++stat.misses;
        entries.emplace(id, Entry{slot, ++clock});
        stat.occupied_slots = entries.size();
        return slot;
    }
    ggml_tensor * weight_view(ggml_context * c, int layer, int role) const {
        const auto & info = layers[size_t(layer)];
        const int64_t columns = role == 2 ? g.expert_ff : g.embd;
        const int64_t rows = role == 2 ? g.embd : g.expert_ff;
        auto tensor = ggml_new_tensor_3d(c, info.types[size_t(role)], columns, rows, int64_t(stat.slot_count));
        tensor->nb[2] = stat.slot_stride;
        tensor->nb[3] = stat.slot_stride * stat.slot_count;
        check(stat.slot_stride % ggml_type_size(tensor->type) == 0, "CUDA expert stride is not an integral number of quant blocks");
        const size_t offset = info.offsets[size_t(role)];
        check(offset + ggml_nbytes(tensor) <= stat.pool_bytes, "weight view exceeds slot pool");
        check(offset + ggml_backend_buft_get_alloc_size(buffer_type, tensor) <= stat.pool_bytes,
              "weight view padding exceeds slot pool");
        // Explicit borrowed reinterpret view over an I8 byte-storage tensor. Calling
        // tensor_alloc() on a naked quant tensor would zero its trailing quant padding:
        // the last slot may contain another layer with different role offsets/types.
        // CUDA's view initializer deliberately skips that write. All required padding
        // was already zeroed inside the selected expert's own slot before its upload.
        tensor->view_src = pool_proxy;
        tensor->view_offs = offset;
        check(ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS, "cannot bind borrowed weight view");
        return tensor;
    }
    void execute(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & mapped_ids,
                               const std::vector<int32_t> & original_ids,
                               const std::vector<float> & route_weights,
                               const std::array<std::vector<float>, 3> & scales,
                               std::vector<float> & output,
                               ggml_backend_event_t weights_ready = nullptr,
                               const std::set<uint64_t> * next_prefetch_keys = nullptr) {
#ifdef M31_FAST_FFN_CUDA
        if (fast_ffn) {
            const auto & info = layers[size_t(layer)];
            const auto * base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(pool));
            const char * debug_dir = std::getenv("M31_GPU_QUANTS_DEBUG_DIR");
            auto dump = [&](const std::string & name, const auto & values) {
                if (!debug_dir || !debug_expert_layer(layer, n)) return;
                std::filesystem::create_directories(debug_dir);
                const auto path = std::filesystem::path(debug_dir) / ("l"+std::to_string(layer)+"_fast_"+name+".bin");
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size()*sizeof(values[0])));
                check(bool(file), "cannot write requested fast FFN debug dump");
            };
            dump("input", x);
            dump("original_ids", original_ids);
            dump("mapped_ids", mapped_ids);
            dump("route_weights", route_weights);
            std::vector<int32_t> logical_to_physical(size_t(g.experts), -1);
            for (size_t i = 0; i < original_ids.size(); ++i) {
                auto & physical = logical_to_physical[size_t(original_ids[i])];
                check(physical == -1 || physical == mapped_ids[i], "inconsistent fast FFN route remapping");
                physical = mapped_ids[i];
            }
            std::vector<int32_t> logical, physical;
            logical.reserve(size_t(g.experts));
            physical.reserve(size_t(g.experts));
            for (int expert = 0; expert < g.experts; ++expert) {
                if (logical_to_physical[size_t(expert)] < 0) continue;
                logical.push_back(expert);
                physical.push_back(logical_to_physical[size_t(expert)]);
            }
            try {
                fast_ffn->prepare(info.types[0], info.types[1], info.types[2],
                                  int(g.embd), int(g.expert_ff), int(g.used), n,
                                  x, original_ids, route_weights);
                // Growth may call cudaMalloc/cudaFree, which can synchronize the
                // device. Start speculative N+1 DMA only after these reserves.
                if (next_prefetch_keys) begin_prefetch(layer, n, *next_prefetch_keys);
                fast_ffn->enqueue(base + info.offsets[0], base + info.offsets[1], base + info.offsets[2],
                                  stat.slot_stride, logical, physical,
                                  info.scales[0], info.scales[1], info.scales[2], weights_ready);
                fast_ffn->read_into(output);
                dump("result", output);
                stat.scratch_bytes = std::max(stat.scratch_bytes, fast_ffn->scratch_bytes());
                ++stat.graph_batches;
                ++stat.fast_forwards;
                return;
            } catch (...) {
                fast_ffn->abort();
                throw;
            }
        }
#else
        (void)original_ids;
        (void)weights_ready;
        (void)next_prefetch_keys;
#endif
#ifdef M31_CPU_QUANTS_CUDA
        if (cpu_quants) {
            const char * debug_dir = std::getenv("M31_GPU_QUANTS_DEBUG_DIR");
            auto dump = [&](const std::string & name, const auto & values) {
                if (!debug_dir || !debug_expert_layer(layer, n)) return;
                if (!debug_expert_intermediates() && (name == "gate" || name == "up" || name == "activation" || name == "down")) return;
                std::filesystem::create_directories(debug_dir);
                const auto path = std::filesystem::path(debug_dir) / ("l"+std::to_string(layer)+"_"+name+".bin");
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size()*sizeof(values[0])));
                check(bool(file), "cannot write requested GPU quant debug dump");
            };
            dump("input", x);
            dump("mapped_ids", mapped_ids);
            dump("route_weights", route_weights);
            if (debug_dir && debug_expert_layer(layer, n)) {
                std::vector<int32_t> original_ids(mapped_ids.size(), -1);
                for (const auto & entry : entries) if (int(entry.first/uint64_t(g.experts)) == layer)
                    for (size_t route = 0; route < mapped_ids.size(); ++route)
                        if (mapped_ids[route] == int32_t(entry.second.slot)) original_ids[route] = int32_t(entry.first%uint64_t(g.experts));
                dump("original_ids", original_ids);
            }
            const auto & info = layers[size_t(layer)];
            const auto * pool_base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(pool));
            std::array<std::vector<float>, 3> local_outputs;
            auto mm = [&](int role, const std::vector<float> & activation) -> const std::vector<float> & {
                const int columns = role == 2 ? int(g.expert_ff) : int(g.embd);
                const int rows = role == 2 ? int(g.embd) : int(g.expert_ff);
                auto & out = reuse_exact_buffers ? exact_outputs[size_t(role)] : local_outputs[size_t(role)];
                if (reuse_exact_buffers)
                    cpu_quants->multiply_into(info.types[size_t(role)], pool_base + info.offsets[size_t(role)],
                                             stat.slot_stride, columns, rows, int(g.used), n, activation, mapped_ids, out);
                else out = cpu_quants->multiply(info.types[size_t(role)], pool_base + info.offsets[size_t(role)],
                                               stat.slot_stride, columns, rows, int(g.used), n, activation, mapped_ids);
                if (info.has_scale[size_t(role)]) {
                    for (size_t route = 0; route < mapped_ids.size(); ++route)
                        for (int row = 0; row < rows; ++row)
                            out[route*size_t(rows)+size_t(row)] *= scales[size_t(role)][size_t(mapped_ids[route])];
                }
                return out;
            };
            auto operation_start = profile ? Clock::now() : Clock::time_point{};
            const auto & gate = mm(0, x);
            if (profile) stat.gate_seconds += elapsed(operation_start);
            operation_start = profile ? Clock::now() : Clock::time_point{};
            const auto & up = mm(1, x);
            if (profile) stat.up_seconds += elapsed(operation_start);
            dump("gate", gate);
            dump("up", up);
            std::vector<float> local_activation;
            auto & activation = reuse_exact_buffers ? exact_activation : local_activation;
            activation.resize(gate.size());
            operation_start = profile ? Clock::now() : Clock::time_point{};
            gpu_host_oai_rows(cpu_quants->host_executor(), size_t(g.expert_ff), gate, up, activation);
            if (profile) stat.oai_seconds += elapsed(operation_start);
            dump("activation", activation);
            operation_start = profile ? Clock::now() : Clock::time_point{};
            const auto & down = mm(2, activation);
            if (profile) stat.down_seconds += elapsed(operation_start);
            dump("down", down);
            operation_start = profile ? Clock::now() : Clock::time_point{};
            auto & result=output;
            result.resize(size_t(g.embd)*size_t(n));
            auto reduce_rows = [&](size_t begin, size_t end) {
            for (size_t token = begin; token < end; ++token) for (int row = 0; row < g.embd; ++row) {
                float sum = 0;
                for (int rank = 0; rank < g.used; ++rank) {
                    const size_t route = size_t(token)*size_t(g.used)+size_t(rank);
                    const float term = down[route*size_t(g.embd)+size_t(row)]*route_weights[route];
                    sum = rank == 0 ? term : sum+term;
                }
                result[size_t(token)*size_t(g.embd)+size_t(row)] = sum;
            }
            };
            if (parallel_reduce) cpu_quants->host_executor().parallel_rows(size_t(n), reduce_rows);
            else reduce_rows(0, size_t(n));
            dump("result", result);
            if (profile) stat.reduce_seconds += elapsed(operation_start);
            stat.scratch_bytes = std::max(stat.scratch_bytes, cpu_quants->scratch_bytes());
            ++stat.graph_batches;
            return;
        }
#endif
        Context context(8 * 1024 * 1024);
        auto c = context.value;
        auto graph = ggml_new_graph_custom(c, 8192, false);
        struct CaptureGuard {
            ggml_backend_t backend;
            ~CaptureGuard() { ggml_backend_synchronize(backend); clear_captures(backend); }
        } guard{backend};
        auto input = ggml_new_tensor_3d(c, GGML_TYPE_F32, g.embd, 1, n);
        auto ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, g.used, n);
        auto rw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, g.used, n);
        for (auto tensor : {input, ids, rw}) ggml_set_input(tensor);
        std::array<ggml_tensor *, 3> scale_inputs{};
        const auto & info = layers[size_t(layer)];
        auto mm = [&](int role, ggml_tensor * activation) {
            auto weight = weight_view(c, layer, role);
            auto out = ggml_mul_mat_id(c, weight, activation, ids);
            if (weight->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(out, GGML_PREC_BF16);
            if (stat.q8_activations) ggml_prec_set_src(out, GGML_PREC_Q8, 1);
            if (info.has_scale[size_t(role)]) {
                auto scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, int64_t(stat.slot_count));
                ggml_set_input(scale);
                scale_inputs[size_t(role)] = scale;
                auto s = ggml_repeat_4d(c, ggml_reshape_3d(c, scale, 1, int64_t(stat.slot_count), 1),
                                      1, int64_t(stat.slot_count), n, 1);
                out = ggml_mul(c, out, ggml_get_rows(c, s, ids));
            }
            return out;
        };
        // Preserve the original [used,n] shape: GPU dispatch/activation quantization
        // must not change merely because an expert received fewer than nine tokens.
        ggml_tensor * activation = nullptr;
        if(native_gu_concat&&!info.has_scale[0]&&!info.has_scale[1])
            activation=native_grouped_gate_up(c,weight_view(c,layer,0),weight_view(c,layer,1),input,ids,stat.q8_activations);
        if(!activation){
            auto gate=mm(0,input),up=mm(1,input);
            activation=ggml_swiglu_oai(c,gate,up,1.702f,7.0f);
        }
        auto down = mm(2, activation);
        auto weighted = ggml_mul(c, down, rw);
        ggml_tensor * result = nullptr;
        for (int64_t rank = 0; rank < g.used; ++rank) {
            auto term = ggml_view_2d(c, weighted, g.embd, n, weighted->nb[2], size_t(rank) * weighted->nb[1]);
            result = result ? ggml_add(c, result, term) : term;
        }
        result = ggml_cont(c, result);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            check(ggml_backend_dev_supports_op(device, node), std::string("backend does not support ") + ggml_op_name(node->op));
        }
        // Graph shapes/scale topology change across layers. Rebuild the alias plan,
        // retaining sufficiently large existing buffers rather than allocating afresh.
        check(ggml_gallocr_reserve(allocator, graph), "graph scratch reserve failed");
        check(ggml_gallocr_alloc_graph(allocator, graph), "graph scratch allocation failed");
        stat.scratch_bytes = std::max(stat.scratch_bytes, ggml_gallocr_get_buffer_size(allocator, 0));
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, mapped_ids.data(), 0, mapped_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(rw, route_weights.data(), 0, route_weights.size() * sizeof(float));
        for (int role = 0; role < 3; ++role)
            if (scale_inputs[size_t(role)])
                ggml_backend_tensor_set(scale_inputs[size_t(role)], scales[size_t(role)].data(), 0, stat.slot_count * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
        output.resize(size_t(g.embd) * size_t(n));
        check(ggml_is_contiguous(result) && ggml_nelements(result) == int64_t(output.size()), "invalid final expert result layout");
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        ++stat.graph_batches;
        return;
    }
#include "gpu_device_experts_impl.hpp"
};

GpuExperts::GpuExperts() : p_(std::make_unique<Impl>()) {}
GpuExperts::~GpuExperts() = default;

void GpuExperts::init(const GgufModel & model, const Geometry & g, const ExpertArena & arena,
                      ggml_backend_t backend, double cache_gib, int max_chunk, bool q8_activations) {
    check(!p_->ready && p_->backend == nullptr, "repeated initialization");
    check(backend != nullptr && ggml_backend_get_device(backend) != nullptr, "backend/device is missing");
    check(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU,
          "this opt-in path requires a GPU backend");
    const char * session = std::getenv("M31_CARD_SESSION");
    check(session && std::string(session) == "1", "GPU requires the aictl/flock runner");
    check(std::isfinite(cache_gib) && cache_gib > 0 && cache_gib <= 512,
          "cache budget must be finite and positive (GiB)");
    check(max_chunk > 0 && max_chunk <= 32768 && g.embd > 0 && g.expert_ff > 0 && g.layers > g.dense_lead &&
          g.used > 0 && g.used <= g.experts && g.experts <= 4096 && g.layers <= 1024,
          "invalid geometry/chunk");
    check(arena.buffer() != nullptr && arena.layers().size() == size_t(g.layers), "complete RAM arena is required");
    p_->g = g;
    p_->backend = backend;
    p_->device = ggml_backend_get_device(backend);
    p_->buffer_type = ggml_backend_get_default_buffer_type(backend);
    p_->max_chunk = max_chunk;
    const char * profile = std::getenv("M31_GPU_EXPERT_PROFILE");
    p_->profile = profile && std::string(profile) == "1";
    const char * compact = std::getenv("M31_GPU_PREFILL_COMPACT");
    p_->compact_uploads = compact && std::string(compact) == "1";
    const char * reuse = std::getenv("M31_GPU_REUSE_BUFFERS");
    p_->reuse_exact_buffers = reuse && std::string(reuse) == "1";
    const char * parallel_reduce = std::getenv("M31_GPU_PARALLEL_REDUCE");
    p_->parallel_reduce = parallel_reduce && std::string(parallel_reduce) == "1";
    const char * ranked_prefetch = std::getenv("M31_GPU_PREFETCH_RANK");
    p_->ranked_prefetch = ranked_prefetch && std::string(ranked_prefetch) == "1";
    if(const char *value=std::getenv("M31_GPU_NATIVE_GU_CONCAT"))p_->native_gu_concat=std::string(value)=="1";
    if(const char *value=std::getenv("M31_GPU_LAYER_BANKS"))p_->layer_banks_requested=std::string(value)=="1";
    if (const char * limit=std::getenv("M31_GPU_PREFETCH_LIMIT")) {
        const std::string text(limit);
        check(!text.empty() && text.find_first_not_of("0123456789")==std::string::npos,
              "prefetch limit must be an unsigned integer");
        const auto count=std::stoull(text);
        check(count<=uint64_t(g.experts),"prefetch limit exceeds experts per layer");
        p_->prefetch_limit=size_t(count);
    }
    if (const char * history = std::getenv("M31_GPU_PREFETCH_HISTORY"); history && std::string(history) == "1")
        p_->prefetch_history = std::make_unique<PrefillRouteHistory>(size_t(g.layers), size_t(g.experts));
    p_->prefetch_order.resize(size_t(g.layers));
    for (auto & order:p_->prefetch_order) {
        order.resize(size_t(g.experts));
        std::iota(order.begin(),order.end(),0);
    }
    p_->stat.budget_bytes = size_t(cache_gib * 1073741824.0);
    p_->stat.q8_activations = q8_activations;
    p_->layers.resize(size_t(g.layers));
    p_->cached_decode_graphs.resize(size_t(g.layers));
    const size_t alignment = ggml_backend_buft_get_alignment(p_->buffer_type);
    size_t stride_multiple = alignment, maximum_span = 0;
    const char * role_names[] = {"gate", "up", "down"};
    Context metadata(64 * 1024);
    for (int l = 0; l < g.layers; ++l) {
        if (!g.routed(l)) continue;
        auto & layer = p_->layers[size_t(l)];
        ggml_reset(metadata.value);
        for (int r = 0; r < 3; ++r) {
            const std::string name = layer_name(l, std::string("ffn_") + role_names[r] + "_exps.weight");
            size_t shard = 0;
            const auto * info = model.find(name, &shard);
            const int64_t n0 = r == 2 ? g.expert_ff : g.embd;
            const int64_t n1 = r == 2 ? g.embd : g.expert_ff;
            const std::vector<uint64_t> shape = {uint64_t(n0), uint64_t(n1), uint64_t(g.experts)};
            check(info && info->shape == shape && info->type < GGML_TYPE_COUNT && model.in_bounds(*info, shard), "invalid weight " + name);
            const auto type = ggml_type(info->type);
            layer.types[size_t(r)] = type;
            layer.sources[size_t(r)] = arena.find(name);
            check(layer.sources[size_t(r)] != nullptr, "missing arena slice " + name);
            auto tensor = ggml_new_tensor_2d(metadata.value, type, n0, n1);
            layer.raw_bytes[size_t(r)] = ggml_nbytes(tensor);
            layer.allocation_bytes[size_t(r)] = ggml_backend_buft_get_alloc_size(p_->buffer_type, tensor);
            layer.offsets[size_t(r)] = aligned(layer.span, alignment);
            layer.span = layer.offsets[size_t(r)] + aligned(layer.allocation_bytes[size_t(r)], alignment);
            check(layer.raw_bytes[size_t(r)] <= layer.sources[size_t(r)]->expert_stride, "invalid arena stride");
            // CUDA casts nb[2]/type_size to a quant-block stride: rounding to 128
            // bytes alone silently truncates the expert offset for IQ3/IQ4/NVFP4.
            stride_multiple = common_multiple(stride_multiple, ggml_type_size(type));
            layer.scales[size_t(r)].assign(size_t(g.experts), 1.0f);
            const std::string scale_name = name.substr(0, name.size() - 6) + "scale";
            const auto * scales = model.find(scale_name, &shard);
            layer.has_scale[size_t(r)] = scales != nullptr;
            if (scales) {
                check(scales->type == GGML_TYPE_F32 && scales->shape == std::vector<uint64_t>{uint64_t(g.experts)} &&
                      model.in_bounds(*scales, shard), "invalid external scales " + scale_name);
                std::memcpy(layer.scales[size_t(r)].data(), model.shard(shard).tensor_data(*scales), size_t(g.experts) * sizeof(float));
            } else check(type != GGML_TYPE_NVFP4, "missing external NVFP4 scale");
            for (float scale : layer.scales[size_t(r)]) check(std::isfinite(scale), "non-finite external scale");
        }
        maximum_span = std::max(maximum_span, layer.span);
    }
    p_->stat.slot_stride = aligned(maximum_span, stride_multiple);
    p_->stat.minimum_budget_bytes = p_->stat.slot_stride;
    check(p_->stat.budget_bytes >= p_->stat.slot_stride,
          "cache budget must fit one complete expert slot: minimum bytes=" + std::to_string(p_->stat.slot_stride));
    const size_t total_experts = size_t(g.experts) * size_t(g.layers - g.dense_lead);
    p_->stat.slot_count = std::min(p_->stat.budget_bytes / p_->stat.slot_stride, total_experts);
    if(p_->layer_banks_requested){
        check(p_->stat.slot_count>=2*size_t(g.experts),"two layer banks require budget for256 expert slots");
        p_->stat.slot_count=2*size_t(g.experts);
    }
    p_->stat.pool_bytes = p_->stat.slot_count * p_->stat.slot_stride;
    check(p_->stat.slot_count && p_->stat.slot_count <= size_t(std::numeric_limits<int32_t>::max()), "invalid slot count");
    p_->pool = ggml_backend_buft_alloc_buffer(p_->buffer_type, p_->stat.pool_bytes);
    check(p_->pool != nullptr, "fixed VRAM slot allocation failed (budget excludes dense/KV/graph scratch)");
    check(ggml_backend_buffer_get_size(p_->pool) <= p_->stat.budget_bytes, "backend allocated more than the requested cache budget");
    ggml_backend_buffer_set_usage(p_->pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    p_->stat.resident_bytes = ggml_backend_buffer_get_size(p_->pool);
    p_->stat.peak_resident_bytes = p_->stat.resident_bytes;
    p_->slot_metadata = std::make_unique<Context>((p_->stat.slot_count + 1) * ggml_tensor_overhead() + 4096);
    p_->pool_proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.pool_bytes));
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(p_->pool));
    check(ggml_backend_tensor_alloc(p_->pool, p_->pool_proxy, base) == GGML_STATUS_SUCCESS, "cannot bind pool byte tensor");
    for (size_t slot = 0; slot < p_->stat.slot_count; ++slot) {
        auto proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.slot_stride));
        check(ggml_backend_tensor_alloc(p_->pool, proxy, base + slot * p_->stat.slot_stride) == GGML_STATUS_SUCCESS,
              "cannot bind slot byte tensor");
        p_->slot_proxies.push_back(proxy);
        p_->free_slots.push_back(p_->stat.slot_count - slot - 1);
    }
    p_->allocator = ggml_gallocr_new(p_->buffer_type);
    check(p_->allocator != nullptr, "graph allocator initialization failed");
    const char * async = std::getenv("M31_GPU_PREFILL_ASYNC");
    if (async && std::string(async) == "1") {
        const unsigned pack_threads=cpu_pack_threads_from_environment();
        if (pack_threads>1) {
            p_->pack_pool=std::make_unique<CpuPackSpans>(pack_threads);
            auto * pack=p_->pack_pool.get();
            p_->copier=[pack](const PrefillUploads::CopySpan * spans,size_t count) { pack->copy(spans,count); };
        }
        p_->uploads = std::make_unique<PrefillUploads>(backend, p_->stat.slot_stride, 2,true,256*1024*1024,p_->copier);
        p_->upload_depth=p_->uploads->stats().depth;
        p_->stat.pinned_staging = p_->uploads->stats().pinned;
        std::fprintf(stderr, "M31 GPU experts: opt-in prefill upload ring async=%d depth=%zu staging_bytes=%zu\n",
                     int(p_->uploads->stats().asynchronous), p_->uploads->stats().depth, p_->uploads->stats().staging_bytes);
    } else {
        const auto host_type = ggml_backend_dev_host_buffer_type(p_->device);
        if (host_type) p_->staging = ggml_backend_buft_alloc_buffer(host_type, p_->stat.slot_stride);
        p_->stat.pinned_staging = p_->staging != nullptr;
        if (!p_->staging) p_->pageable_staging.resize(p_->stat.slot_stride);
    }
    const char * prefetch=std::getenv("M31_GPU_PREFETCH_NEXT");
#ifdef M31_REGISTERED_ARENA_CUDA
    if (const char * direct = std::getenv("M31_GPU_REGISTER_ARENA"); direct && std::string(direct) == "1") {
        check(bool(p_->uploads), "registered direct H2D requires M31_GPU_PREFILL_ASYNC=1");
        check(p_->uploads->stats().asynchronous,
              "registered direct H2D requires an asynchronous backend with event support");
        p_->registered_arena=arena.registration();
        if(!p_->registered_arena){
            p_->owned_registered_arena=std::make_unique<GpuRegisteredArena>(backend,
                ggml_backend_buffer_get_base(arena.buffer()), arena.bytes());
            p_->registered_arena=p_->owned_registered_arena.get();
        }
        check(p_->registered_arena->contains(ggml_backend_buffer_get_base(arena.buffer()),arena.bytes()),
              "RAM arena registration has not completed");
    }
#else
    if (const char * direct = std::getenv("M31_GPU_REGISTER_ARENA"); direct && std::string(direct) == "1")
        check(false, "registered direct H2D is unavailable in this build");
#endif
    if (prefetch && std::string(prefetch)=="1" && !p_->layer_banks_requested) {
        check(bool(p_->uploads), "next-layer prefetch requires M31_GPU_PREFILL_ASYNC=1");
        p_->background=std::make_unique<BackgroundPrefillUploads>(backend,p_->stat.slot_stride,p_->copier);
    }
    std::fprintf(stderr, "M31 GPU experts: fixed_slots=%zu slot_stride=%zu pool_bytes=%zu budget=%zu pinned=%d source_q8=%d (layer mul_mat_id; graph scratch separate)\n",
                 p_->stat.slot_count, p_->stat.slot_stride, p_->stat.pool_bytes, p_->stat.budget_bytes,
                 int(p_->stat.pinned_staging), int(p_->stat.q8_activations));
    if(const char *value=std::getenv("M31_GPU_CACHED_DECODE_EXACT"))p_->cached_decode_exact=std::string(value)=="1";
    if(const char *value=std::getenv("M31_DSPARK_GPU_CACHED"))p_->cached_verify_enabled=std::string(value)=="1";
#ifndef M31_CPU_QUANTS_CUDA
    check(!p_->cached_decode_exact&&!p_->cached_verify_enabled,"exact cached subset requires STRATA_M31_CPU_QUANTS_CUDA");
#endif
    const char * exact_mode = std::getenv("M31_GPU_EXACT");
    const bool requested_exact = exact_mode && std::string(exact_mode) == "1";
    const char * fast_mode = std::getenv("M31_GPU_FAST_FFN");
    // Native ggml CUDA/MMQ is the ordinary path. The alternative resident
    // implementation and CPU-exact verification each require an explicit opt-in.
    const bool requested_fast = !requested_exact && fast_mode && std::string(fast_mode) == "1";
#ifdef M31_FAST_FFN_CUDA
    if (requested_fast) {
        bool supported = true;
        for (int l = int(g.dense_lead); l < g.layers; ++l) {
            const auto & types = p_->layers[size_t(l)].types;
            supported = supported && GpuFastFfn::supported(types[0], types[1], types[2]);
        }
        if (supported) {
            p_->fast_ffn = std::make_unique<GpuFastFfn>();
            std::fprintf(stderr, "M31 GPU experts: opt-in experimental resident FFN (M31_GPU_FAST_FFN=1); M31_GPU_EXACT=1 selects host verification\n");
        } else {
            check(!fast_mode || std::string(fast_mode) != "1", "explicit fast CUDA FFN does not support this weight type");
            std::fprintf(stderr, "M31 GPU experts: fast FFN unavailable for these types; using existing backend path\n");
        }
    }
#else
    (void)requested_fast;
#endif
#ifdef M31_CPU_QUANTS_CUDA
    bool use_exact = requested_exact;
#ifdef M31_FAST_FFN_CUDA
    use_exact = use_exact && !p_->fast_ffn;
#endif
    if (use_exact) {
        for (int l = int(g.dense_lead); l < g.layers; ++l)
            for (auto type : p_->layers[size_t(l)].types)
                check(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_NVFP4,
                      "CPU-compatible CUDA dot does not support this weight type");
        p_->cpu_quants = std::make_unique<GpuCpuQuants>();
        std::fprintf(stderr, "M31 GPU experts: experimental CPU Q8_K/Q8_0 activation quants, CPU OAI, CUDA AVX2-compatible dot\n");
    }
#else
    check(!requested_exact, "M31_GPU_EXACT requires STRATA_M31_CPU_QUANTS_CUDA=ON");
#endif
    check((!p_->cached_decode_exact&&!p_->cached_verify_enabled)||(!requested_exact&&!requested_fast&&!p_->layer_banks_requested),
          "exact cached subset requires native prefill with the ordinary LRU pool");
    if(p_->layer_banks_requested){
        check(!requested_exact&&!requested_fast,"layer banks require native ggml experts");
        const char *chain=std::getenv("M31_GPU_PREFILL_CHAIN");
        check(chain&&std::string(chain)=="1","layer banks require M31_GPU_PREFILL_CHAIN=1");
#ifdef M31_REGISTERED_ARENA_CUDA
        check(p_->registered_arena,"layer banks require M31_GPU_REGISTER_ARENA=1");
        p_->layer_banks=std::make_unique<DirectLayerUploads>(backend,p_->stat.slot_stride);
        p_->bank_prefetch_enabled=prefetch&&std::string(prefetch)=="1";
#else
        check(false,"layer banks require registered arena CUDA support");
#endif
    }
#ifdef M31_CPU_QUANTS_CUDA
    if(p_->cached_decode_exact){
        // Allocate its small fixed buffers before the planner snapshot. The
        // helper restores an explicit pool affinity mask inside its worker.
        p_->cached_exact_worker=std::make_unique<GpuCachedExact>(int(g.embd),int(g.expert_ff),int(g.used));
        p_->stat.decode_scratch_bytes=p_->cached_exact_worker->stats().device_bytes;
    }
#endif
#ifdef M31_CPU_QUANTS_CUDA
    if(p_->cached_verify_enabled){
        p_->cached_batch_worker=std::make_unique<GpuCachedExact>(int(g.embd),int(g.expert_ff),int(g.used),8);
        p_->stat.decode_scratch_bytes+=p_->cached_batch_worker->stats().device_bytes;
    }
#endif
    p_->ready = true; // No partial helper allocation is exposed after init failure.
    const char * decode_hot_text=std::getenv("M31_GPU_DECODE_SEED_HOT");
    const bool deferred_decode_seed=decode_hot_text&&*decode_hot_text&&std::string(decode_hot_text)!="0";
    const char * hot_text = deferred_decode_seed?decode_hot_text:std::getenv("M31_GPU_SEED_HOT");
    if(deferred_decode_seed){
        const char *ordinary=std::getenv("M31_GPU_SEED_HOT");
        check(!ordinary||std::string(ordinary)=="0","ordinary and decode-only seeds are mutually exclusive");
        check(!p_->layer_banks_requested&&!requested_exact&&!requested_fast,"decode-only seeds require native LRU experts");
    }
    if (hot_text && *hot_text) {
        try {
            const std::string count_text(hot_text);
            check(count_text.find_first_not_of("0123456789") == std::string::npos, "seed hot count must be an unsigned integer");
            const uint64_t hot = std::stoull(count_text);
            if (hot) {
                auto env = [](const char * name) {
                    const char * value = std::getenv(name);
                    check(value && *value, std::string("missing required seed environment: ")+name);
                    return std::string(value);
                };
                const auto rank_path = env("M31_GPU_SEED_PATH");
                const auto identity = cache_seed::hex_digest(env("M31_GPU_SEED_IDENTITY"));
                const auto rank_sha = cache_seed::hex_digest(env("M31_GPU_SEED_SHA256"));
                const auto receipt = cache_seed::read_receipt(env("M31_GPU_SEED_RECEIPT"),
                    cache_seed::hex_digest(env("M31_GPU_SEED_RECEIPT_SHA256")));
                check(receipt.expected.identity == identity && receipt.expected.file_sha256 == rank_sha,
                      "seed environment hashes differ from trusted preflight receipt");
                check(std::filesystem::canonical(rank_path).string() == receipt.rank_path,
                      "seed environment rank path differs from trusted preflight receipt");
                check(receipt.hot == hot && receipt.slots == p_->stat.slot_count,
                      "seed hot count/cache slots differ from trusted preflight receipt");
                cache_seed::verify_current_model(receipt, model);
                if(deferred_decode_seed){
                    const auto rank=cache_seed::read(rank_path,receipt.expected);
                    check(hot<=rank.ids.size()&&hot<=p_->stat.slot_count,"decode seed exceeds cache slots");
                    p_->decode_seed_order.assign(rank.ids.begin(),rank.ids.begin()+size_t(hot));
                    p_->decode_seed_armed=true;
                }else seed_hot(rank_path, receipt.expected, size_t(hot));
            }
        } catch (...) {
            p_->ready = false; // A failed env initialization must never expose a usable partial cache.
            throw;
        }
    }
}

std::vector<float> GpuExperts::forward(int layer, int n, const std::vector<float> & x,
                                      const std::vector<int32_t> & ids, const std::vector<float> & weights) {
    std::vector<float> output;
    forward_into(layer,n,x,ids,weights,output);
    return output;
}

void GpuExperts::forward_into(int layer, int n, const std::vector<float> & x,
                              const std::vector<int32_t> & ids, const std::vector<float> & weights,
                              std::vector<float> & output) {
    check(&output!=&x && &output!=&weights,"output vector must not alias inputs");
    const auto prepare_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    check(p_->ready && p_->g.routed(layer), "uninitialized backend or invalid routed layer");
    check(!p_->layer_banks_requested,"layer banks require the device-only prefill API");
    check(n > 0 && n <= p_->max_chunk, "invalid token count");
    const auto & g = p_->g;
    check(x.size() == size_t(g.embd) * size_t(n) && ids.size() == size_t(g.used) * size_t(n) && weights.size() == ids.size(),
          "input dimensions differ");
    std::vector<uint8_t> selected_mask(size_t(g.experts), 0);
    std::vector<int> selected;
    std::set<uint64_t> protected_keys;
    for (int t = 0; t < n; ++t) for (int rank = 0; rank < g.used; ++rank) {
        const size_t route = size_t(t) * size_t(g.used) + size_t(rank);
        const int expert = ids[route];
        check(expert >= 0 && expert < g.experts && std::isfinite(weights[route]), "invalid route ID/weight");
        for (int j = 0; j < rank; ++j)
            check(ids[size_t(t) * size_t(g.used) + size_t(j)] != expert, "duplicate top-k expert");
        selected_mask[size_t(expert)] = 1;
    }
    // One cache-key insertion per expert, rather than per route. At large chunk
    // sizes most routes revisit the same 128 experts; preserve ascending order.
    for (int expert = 0; expert < g.experts; ++expert) if (selected_mask[size_t(expert)]) {
        selected.push_back(expert);
        protected_keys.insert(p_->key(layer, expert));
    }
    check(selected.size() <= p_->stat.slot_count,
          "layer needs " + std::to_string(selected.size()) + " distinct expert slots; cache has " + std::to_string(p_->stat.slot_count) +
          "; minimum batch cache bytes=" + std::to_string(selected.size() * p_->stat.slot_stride));
    if (p_->profile) p_->stat.prepare_seconds += Impl::elapsed(prepare_start);
    if(n>1)p_->leave_decode_seed();
    try {
        // Keep N+1 DMA in flight through its dense attention graph. Publish its
        // entries before any load/reservation can reuse a reserved destination.
        // A failed wait is handled by the same drain-and-discard recovery below.
        p_->observe_routes(layer, n, selected_mask);
        p_->finish_prefetch();
        if (p_->uploads) p_->uploads->begin_batch();
        std::vector<int32_t> expert_to_slot(size_t(g.experts), -1);
        std::array<std::vector<float>, 3> scales;
        for (auto & values : scales) values.assign(p_->stat.slot_count, 1.0f);
        for (int expert : selected) {
            const size_t slot = p_->load(layer, expert, protected_keys);
            expert_to_slot[size_t(expert)] = int32_t(slot);
            for (int role = 0; role < 3; ++role)
                scales[size_t(role)][slot] = p_->layers[size_t(layer)].scales[size_t(role)][size_t(expert)];
        }
        std::vector<int32_t> mapped_ids(ids.size());
        const auto map_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        for (size_t i = 0; i < ids.size(); ++i) mapped_ids[i] = expert_to_slot[size_t(ids[i])];
        if (p_->profile) p_->stat.prepare_seconds += Impl::elapsed(map_start);
        const auto wait_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        ggml_backend_event_t weights_ready = nullptr;
        if (p_->uploads) {
            p_->uploads->finish();
#ifdef M31_FAST_FFN_CUDA
            if (p_->fast_ffn) weights_ready = p_->uploads->completion_event();
#endif
#ifdef M31_CPU_QUANTS_CUDA
            // The diagnostic exact-CPU dot uses CUDA's default stream, separate
            // from ggml's compute stream on which finish() installs its event wait.
            if (p_->cpu_quants) p_->uploads->synchronize();
#endif
        }
        if (p_->profile) p_->stat.upload_wait_seconds += Impl::elapsed(wait_start);
        bool fast = false;
#ifdef M31_FAST_FFN_CUDA
        fast = bool(p_->fast_ffn);
#endif
        if (!fast) p_->begin_prefetch(layer,n,protected_keys);
        const auto execute_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        check(p_->active_read_keys.empty(),"overlapping expert consumers are unsupported");
        p_->active_read_keys=protected_keys;
        struct ReadersLease {Impl *owner;~ReadersLease(){owner->active_read_keys.clear();}} readers{p_.get()};
        // execute() drains its raw private stream on success and failure. Keep
        // its slot lease until that completion, independently of ggml's stream.
        p_->execute(layer, n, x, mapped_ids, ids, weights, scales, output, weights_ready,
                    fast ? &protected_keys : nullptr);
        if (p_->profile) p_->stat.execute_seconds += Impl::elapsed(execute_start);
        ++p_->stat.forwards;
        return;
    } catch (...) {
        if (p_->background) p_->background->drain();
        p_->pending_prefetch.clear();
        if (p_->uploads) {
            // A failed enqueue may already have evicted a slot or queued a DMA.
            // Drain first, then discard ALL metadata; no pending bytes become hits.
            p_->uploads.reset();
#ifdef M31_CPU_QUANTS_CUDA
            const bool exact_cpu = bool(p_->cpu_quants);
            p_->cpu_quants.reset(); // cudaFree also drains its independent raw stream.
#endif
            p_->entries.clear();
            p_->seeded_keys.clear();
            p_->stat.seeded_slots = 0;
            p_->free_slots.clear();
            for (size_t slot = 0; slot < p_->stat.slot_count; ++slot)
                p_->free_slots.push_back(p_->stat.slot_count-slot-1);
            p_->stat.occupied_slots = 0;
            try {
                // Restore the configured upload depth/stream count rather than
                // silently reverting to the historical two-stage fallback.
                p_->uploads = std::make_unique<PrefillUploads>(p_->backend, p_->stat.slot_stride, p_->upload_depth,true,256*1024*1024,p_->copier);
#ifdef M31_CPU_QUANTS_CUDA
                if (exact_cpu) p_->cpu_quants = std::make_unique<GpuCpuQuants>();
#endif
            } catch (...) {
                p_->ready = false; // Failed recovery is terminal; forward refuses reuse.
            }
        }
        throw;
    }
}

#include "gpu_device_forward_impl.hpp"
#include "gpu_shortprefill_impl.hpp"
#include "gpu_cached_decode_impl.hpp"
#include "gpu_cached_batched_impl.hpp"

bool GpuExperts::forward_cached(int layer, int n, const std::vector<float> & x,
                                const std::vector<int32_t> & ids, const std::vector<float> & weights,
                                std::vector<float> & output) {
    check(p_->ready && p_->g.routed(layer), "uninitialized backend or invalid routed layer");
    check(n > 0 && n <= p_->max_chunk, "invalid token count");
    const auto & g = p_->g;
    check(x.size() == size_t(g.embd)*size_t(n) && ids.size() == size_t(g.used)*size_t(n) && weights.size() == ids.size(),
          "input dimensions differ");
    // Preflight every route before touching metadata or output. No replacement,
    // host packing or DMA occurs on a miss, including partially resident batches.
    for (int token = 0; token < n; ++token) for (int rank = 0; rank < g.used; ++rank) {
        const size_t i = size_t(token)*size_t(g.used)+size_t(rank);
        check(ids[i] >= 0 && ids[i] < g.experts && std::isfinite(weights[i]), "invalid route ID/weight");
        for (int j = 0; j < rank; ++j)
            check(ids[size_t(token)*size_t(g.used)+size_t(j)] != ids[i], "duplicate top-k expert");
    }
    for (const auto expert : ids)
        if (p_->entries.find(p_->key(layer, expert)) == p_->entries.end()) return false;
    const auto prepare_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    std::vector<int32_t> mapped(ids.size());
    std::array<std::vector<float>, 3> scales;
    for (auto & values : scales) values.assign(p_->stat.slot_count, 1.0f);
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto slot = p_->entries.at(p_->key(layer, ids[i])).slot;
        mapped[i] = int32_t(slot);
        for (int role = 0; role < 3; ++role)
            scales[size_t(role)][slot] = p_->layers[size_t(layer)].scales[size_t(role)][size_t(ids[i])];
    }
    if (p_->profile) p_->stat.prepare_seconds += Impl::elapsed(prepare_start);
    const auto wait_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    // Exact CUDA dots own a raw stream; drain earlier users of the borrowed
    // backend before their execution. Complete forward() uploads already drained.
    ggml_backend_synchronize(p_->backend);
    if (p_->profile) p_->stat.upload_wait_seconds += Impl::elapsed(wait_start);
    const auto execute_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    // Preserve the existing alias-safe API while retaining normal caller capacity.
    // A miss already returned above and never reaches either output path.
    std::vector<float> aliased_result;
    const bool aliases=&output==&x || &output==&weights;
    p_->execute(layer, n, x, mapped, ids, weights, scales, aliases?aliased_result:output);
    if (p_->profile) p_->stat.execute_seconds += Impl::elapsed(execute_start);
    std::set<uint64_t> touched;
    for (auto expert : ids) touched.insert(p_->key(layer, expert));
    for (auto id : touched) { p_->entries.at(id).last_used = ++p_->clock; ++p_->stat.hits; }
    ++p_->stat.cached_forwards;
    ++p_->stat.forwards;
    if (aliases) output = std::move(aliased_result);
    return true;
}

void GpuExperts::print_profile(const char * label) const {
    const auto & s = p_->stat;
    std::fprintf(stderr, "M31 GPU expert profile label=%s forwards=%llu cached=%llu fast=%llu hits=%llu misses=%llu bytes=%llu control_ms=%.3f pack_ms=%.3f upload_submit_ms=%.3f upload_wait_ms=%.3f execute_wall_ms=%.3f gate_wall_ms=%.3f up_wall_ms=%.3f oai_host_ms=%.3f down_wall_ms=%.3f reduce_host_ms=%.3f\n",
        label ? label : "live", (unsigned long long)s.forwards, (unsigned long long)s.cached_forwards,
        (unsigned long long)s.fast_forwards,
        (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.upload_bytes,
        1000*s.prepare_seconds, 1000*s.pack_seconds, 1000*s.upload_submit_seconds,
        1000*s.upload_wait_seconds, 1000*s.execute_seconds, 1000*s.gate_seconds,
        1000*s.up_seconds, 1000*s.oai_seconds, 1000*s.down_seconds, 1000*s.reduce_seconds);
    if (p_->profile) std::fprintf(stderr,
        "M31 GPU route demand profile label=%s required_experts=%llu required_bytes=%llu predicted_used=%llu predicted_unused=%llu predicted_used_bytes=%llu predicted_unused_bytes=%llu history_batches=%llu history_skipped=%llu\n",
        label?label:"live",(unsigned long long)s.required_experts,(unsigned long long)s.required_bytes,
        (unsigned long long)s.prefetch_used_experts,(unsigned long long)s.prefetch_unused_experts,
        (unsigned long long)s.prefetch_used_bytes,(unsigned long long)s.prefetch_unused_bytes,
        (unsigned long long)s.prefetch_history_batches,(unsigned long long)s.prefetch_history_skipped_experts);
    std::fprintf(stderr,"M31 GPU cached hybrid label=%s forwards=%llu route_lookups=%llu resident_routes=%llu hit_rate=%.6f submit_ms=%.3f cpu_miss_ms=%.3f join_ms=%.3f scratch_bytes=%zu seed_primes=%llu seed_bytes=%llu seed_ms=%.3f\n",
        label?label:"live",(unsigned long long)s.hybrid_forwards,(unsigned long long)s.decode_route_lookups,
        (unsigned long long)s.decode_resident_routes,double(s.decode_resident_routes)/double(std::max<uint64_t>(1,s.decode_route_lookups)),
        1000*s.hybrid_submit_seconds,1000*s.hybrid_cpu_seconds,1000*s.hybrid_join_seconds,s.decode_scratch_bytes,
        (unsigned long long)s.decode_seed_primes,(unsigned long long)s.decode_seed_upload_bytes,1000*s.decode_seed_seconds);
    std::fprintf(stderr,"M31 GPU cached batched label=%s calls=%llu forwards=%llu tokens=%llu routes=%llu hits=%llu hit_rate=%.6f allmiss=%llu seed_ms=%.3f submit_ms=%.3f cpu_ms=%.3f join_ms=%.3f wall_ms=%.3f\n",
        label?label:"live",(unsigned long long)s.cached_batch_calls,(unsigned long long)s.cached_batch_forwards,
        (unsigned long long)s.cached_batch_tokens,(unsigned long long)s.cached_batch_routes,(unsigned long long)s.cached_batch_hits,
        double(s.cached_batch_hits)/double(std::max<uint64_t>(1,s.cached_batch_routes)),(unsigned long long)s.cached_batch_allmiss,
        1000*s.cached_batch_seed_seconds,1000*s.cached_batch_submit_seconds,1000*s.cached_batch_cpu_seconds,
        1000*s.cached_batch_join_seconds,1000*s.cached_batch_wall_seconds);
    std::fprintf(stderr, "M31 GPU prefetch profile label=%s batches=%llu experts=%llu bytes=%llu join_host_ms=%.3f pack_host_ms=%.3f staging_wait_host_ms=%.3f submit_host_ms=%.3f\n",
        label?label:"live",(unsigned long long)s.prefetch_batches,(unsigned long long)s.prefetched_experts,
        (unsigned long long)s.prefetch_bytes,1000*s.prefetch_wait_seconds,1000*s.prefetch_pack_seconds,
        1000*s.prefetch_staging_wait_seconds,1000*s.prefetch_submit_seconds);
    if(s.short_hybrid_forwards)std::fprintf(stderr,
        "M31 shortprefill profile label=%s forwards=%llu cpu_experts=%llu gpu_experts=%llu cached_experts=%llu cpu_routes=%llu gpu_routes=%llu cold_gpu_bytes=%llu cpu_ms=%.3f wall_ms=%.3f\n",
        label?label:"live",(unsigned long long)s.short_hybrid_forwards,
        (unsigned long long)s.short_cpu_experts,(unsigned long long)s.short_gpu_experts,
        (unsigned long long)s.short_cached_experts,(unsigned long long)s.short_cpu_routes,
        (unsigned long long)s.short_gpu_routes,(unsigned long long)s.short_cold_gpu_bytes,
        1000*s.short_cpu_seconds,1000*s.short_wall_seconds);
    if (p_->uploads) {
        const auto & u = p_->uploads->stats();
        std::fprintf(stderr, "M31 GPU upload profile label=%s bytes=%llu pack_host_ms=%.3f staging_wait_host_ms=%.3f submit_host_ms=%.3f\n",
            label ? label : "live", (unsigned long long)u.upload_bytes,
            1000*u.pack_seconds, 1000*u.staging_wait_seconds, 1000*u.submit_seconds);
    }
#ifdef M31_REGISTERED_ARENA_CUDA
    if(p_->layer_banks){
        const auto &u=p_->layer_banks->stats();
        std::fprintf(stderr,"M31 GPU layer banks label=%s batches=%llu experts=%llu copies=%llu bytes=%llu padding_bytes=%llu\n",
            label?label:"live",(unsigned long long)u.batches,(unsigned long long)u.experts,
            (unsigned long long)u.copies,(unsigned long long)u.upload_bytes,(unsigned long long)u.padding_bytes);
    }
#endif
#ifdef M31_CPU_QUANTS_CUDA
    if (p_->cpu_quants) p_->cpu_quants->print_profile(label);
    if(p_->cached_batch_worker){
        const auto e=p_->cached_batch_worker->stats();
        std::fprintf(stderr,"M31 GPU cached batch worker label=%s jobs=%llu assignments=%llu joins=%llu H2D_bytes=%llu D2H_bytes=%llu worker_ms=%.3f gate_up_ms=%.3f oai_ms=%.3f down_ms=%.3f device_bytes=%zu pinned_bytes=%zu\n",
            label?label:"live",(unsigned long long)e.jobs,(unsigned long long)e.experts,(unsigned long long)e.joins,
            (unsigned long long)e.h2d_bytes,(unsigned long long)e.d2h_bytes,1000*e.worker_seconds,
            1000*e.gate_up_seconds,1000*e.oai_seconds,1000*e.down_seconds,e.device_bytes,e.pinned_bytes);
    }
    if(p_->cached_exact_worker){
        const auto e=p_->cached_exact_worker->stats();
        std::fprintf(stderr,"M31 GPU cached exact label=%s jobs=%llu experts=%llu joins=%llu H2D_bytes=%llu D2H_bytes=%llu worker_ms=%.3f gate_up_ms=%.3f oai_ms=%.3f down_ms=%.3f device_bytes=%zu pinned_bytes=%zu\n",
            label?label:"live",(unsigned long long)e.jobs,(unsigned long long)e.experts,(unsigned long long)e.joins,
            (unsigned long long)e.h2d_bytes,(unsigned long long)e.d2h_bytes,1000*e.worker_seconds,
            1000*e.gate_up_seconds,1000*e.oai_seconds,1000*e.down_seconds,e.device_bytes,e.pinned_bytes);
        if(e.iq3_imma_calls||e.joined_gu_jobs||p_->cached_rank_outer_calls)std::fprintf(stderr,"M31 GPU cached acceleration label=%s iq3_imma_calls=%llu iq3_imma_rows=%llu rank_outer_calls=%llu joined_gu_jobs=%llu joined_gu_assignments=%llu joined_gu_groups=%llu\n",
            label?label:"live",(unsigned long long)e.iq3_imma_calls,(unsigned long long)e.iq3_imma_rows,(unsigned long long)p_->cached_rank_outer_calls,
            (unsigned long long)e.joined_gu_jobs,(unsigned long long)e.joined_gu_assignments,(unsigned long long)e.joined_gu_groups);
    }
#endif
#ifdef M31_FAST_FFN_CUDA
    if (p_->fast_ffn) p_->fast_ffn->print_profile(label);
#endif
}

void GpuExperts::print_cached_batch_profile(const char *label) const {
#ifdef M31_CPU_QUANTS_CUDA
    if(!p_->cached_batch_worker)return;
    const auto e=p_->cached_batch_worker->stats();
    if(!e.grouped_jobs)return;
    std::fprintf(stderr,"M31 GPU cached grouped label=%s jobs=%llu assignments=%llu grouped_jobs=%llu grouped_assignments=%llu unique_grouped_experts=%llu multiply_calls=%llu saved_multiply_calls=%llu worker_ms=%.3f gate_up_ms=%.3f oai_ms=%.3f down_ms=%.3f device_bytes=%zu pinned_bytes=%zu\n",
        label?label:"verify",(unsigned long long)e.jobs,(unsigned long long)e.experts,
        (unsigned long long)e.grouped_jobs,(unsigned long long)e.grouped_assignments,
        (unsigned long long)e.grouped_unique_experts,(unsigned long long)e.multiply_calls,
        (unsigned long long)(3*e.experts-e.multiply_calls),1000*e.worker_seconds,
        1000*e.gate_up_seconds,1000*e.oai_seconds,1000*e.down_seconds,e.device_bytes,e.pinned_bytes);
#else
    (void)label;
#endif
}

void GpuExperts::seed_hot(const std::string & rank_path, const cache_seed::ExpectedRank & expected,
                           size_t hot_count) {
    check(p_->ready, "seed initialization requires a ready cache");
    check(!p_->layer_banks_requested,"layer banks do not support seeded LRU slots");
    check(p_->entries.empty() && p_->stat.forwards == 0 && p_->seeded_keys.empty(),
          "seed initialization is allowed only before the first forward on a cold cache");
    check(expected.layers == uint32_t(p_->g.layers) && expected.experts == uint32_t(p_->g.experts) &&
          expected.dense_lead == uint32_t(p_->g.dense_lead), "seed geometry differs from loaded model");
    const auto ranking = cache_seed::read(rank_path, expected);
    if (p_->ranked_prefetch) {
        for (auto & order:p_->prefetch_order) order.clear();
        for (const auto id:ranking.ids)
            p_->prefetch_order[size_t(id / p_->g.experts)].push_back(int(id % p_->g.experts));
        std::fprintf(stderr,"M31 GPU prefetch: next-layer expert order from verified rank\n");
    }
    const auto plan = cache_seed::budget(p_->stat.slot_count, uint32_t(p_->g.experts),
        uint32_t(p_->g.layers), uint32_t(p_->g.dense_lead), uint32_t(p_->g.used), hot_count);
    std::set<uint64_t> pins;
    for (size_t i = 0; i < plan.hot; ++i) pins.insert(ranking.ids[i]);
    try {
        if (p_->uploads) p_->uploads->begin_batch();
        for (size_t i = 0; i < plan.hot; ++i) {
            const uint32_t id = ranking.ids[i];
            p_->load(int(id / p_->g.experts), int(id % p_->g.experts), pins);
        }
        if (p_->uploads) {
            p_->uploads->finish();
            p_->uploads->synchronize(); // Cover ggml and exact-dot CUDA streams before publishing pins.
        }
        ggml_backend_synchronize(p_->backend);
        p_->seeded_keys = std::move(pins);
        p_->stat.seeded_slots = plan.hot;
        std::fprintf(stderr, "M31 seed cache: hot=%zu stream=%zu maximum_hot=%zu recommendation=%zu\n",
                     plan.hot, plan.stream, plan.maximum_hot, plan.recommended_hot);
    } catch (...) {
        // A partial seed is never exposed as a usable cache. A fresh Engine is required.
        p_->ready = false;
        p_->uploads.reset(); // Drain any queued DMA before discarding metadata.
        p_->entries.clear();
        p_->seeded_keys.clear();
        p_->stat.seeded_slots = 0;
        p_->stat.occupied_slots = 0;
        throw;
    }
}

const GpuExperts::Stats & GpuExperts::stats() const { return p_->stat; }
bool GpuExperts::exact_cpu_math() const {
#ifdef M31_CPU_QUANTS_CUDA
    return bool(p_->cpu_quants);
#else
    return false;
#endif
}
bool GpuExperts::fast_gpu_math() const {
#ifdef M31_FAST_FFN_CUDA
    return bool(p_->fast_ffn);
#else
    return false;
#endif
}
bool GpuExperts::native_gpu_math() const {
    if(!p_->ready)return false;
#ifdef M31_FAST_FFN_CUDA
    if(p_->fast_ffn)return false;
#endif
#ifdef M31_CPU_QUANTS_CUDA
    if(p_->cpu_quants)return false;
#endif
    return true;
}
size_t GpuExperts::workspace_bytes() const {return p_->stat.scratch_bytes+p_->stat.decode_scratch_bytes;}
void GpuExperts::release_prefill_workspace() {
    check(p_->ready&&native_gpu_math(),"release prefill workspace requires ready native experts");
    check(p_->active_read_keys.empty(),"cannot release scratch with an active expert consumer");
    p_->finish_prefetch();
#ifdef M31_CPU_QUANTS_CUDA
    if(p_->cached_batch_worker)p_->cached_batch_worker->drain();
    if(p_->cached_exact_worker)p_->cached_exact_worker->drain();
#endif
    if(p_->uploads)p_->uploads->synchronize();
#ifdef M31_REGISTERED_ARENA_CUDA
    if(p_->layer_banks)p_->layer_banks->drain();
#endif
    ggml_backend_synchronize(p_->backend);clear_captures(p_->backend);
    if(p_->allocator)ggml_gallocr_free(p_->allocator);
    p_->allocator=ggml_gallocr_new(p_->buffer_type);
    if(!p_->allocator){p_->ready=false;throw std::runtime_error("M31 GPU experts: empty prefill allocator recreation failed");}
    p_->stat.scratch_bytes=0;
}
int GpuExperts::fast_expert_batch_rows() const {
#ifdef M31_FAST_FFN_CUDA
    return p_->fast_ffn ? p_->fast_ffn->expert_batch_rows() : 0;
#else
    return 0;
#endif
}
bool GpuExperts::fast_fused_reduction() const {
#ifdef M31_FAST_FFN_CUDA
    return p_->fast_ffn && p_->fast_ffn->fused_reduction();
#else
    return false;
#endif
}
bool GpuExperts::fast_rank_phases() const {
#ifdef M31_FAST_FFN_CUDA
    return p_->fast_ffn && p_->fast_ffn->rank_phases();
#else
    return false;
#endif
}
void GpuExperts::reset_stats() {
    // A caller may reset between arbitrary layer forwards. Complete speculative
    // uploads first so their counters belong wholly to the interval before reset.
    p_->finish_prefetch();
    const auto old = p_->stat;
    p_->stat = {};
    p_->stat.resident_bytes = old.resident_bytes;
    p_->stat.peak_resident_bytes = old.resident_bytes;
    p_->stat.budget_bytes = old.budget_bytes;
    p_->stat.minimum_budget_bytes = old.minimum_budget_bytes;
    p_->stat.scratch_bytes = old.scratch_bytes;
    p_->stat.pinned_staging = old.pinned_staging;
    p_->stat.q8_activations = old.q8_activations;
    p_->stat.slot_stride = old.slot_stride;
    p_->stat.slot_count = old.slot_count;
    p_->stat.pool_bytes = old.pool_bytes;
    p_->stat.occupied_slots = old.occupied_slots;
    p_->stat.seeded_slots = old.seeded_slots;
    p_->stat.decode_scratch_bytes = old.decode_scratch_bytes;
}
} // namespace strata::m31
