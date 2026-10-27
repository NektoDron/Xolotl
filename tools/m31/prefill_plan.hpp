// Conservative chunk proposal, not a substitute for ggml allocator reservation.
#pragma once
#include "m31_geometry.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::m31 {
struct PrefillBudget {
    // Actual device free bytes AFTER dense weights, complete KV capacity and the
    // fixed expert pool have already been allocated. Never subtract them twice.
    uint64_t free_after_resident = 0;
    uint64_t reserve_bytes = 512ULL * 1024 * 1024;
    uint64_t extra_scratch_bytes = 256ULL * 1024 * 1024;
    uint64_t streaming_slots = 0; // slots available to one batch, excluding protected hot entries
    // 4096/8192 rows amortize the complete routed-weight stream over more
    // prompt tokens. The runtime MSA graph still needs its allocator check.
    int max_chunk = 2048;
    // 0 retains the original whole-chunk MSA envelope. Otherwise projection,
    // hidden states, causal inputs and MoE remain full n, while score/mask/FA
    // temporaries cover at most this many independent query rows.
    int attention_query_tile = 0;
    // Tile only indexer GEMM/pool/top-k. FA keeps the original full query
    // geometry, avoiding shape-dependent attention rounding.
    int indexer_query_tile = 0;
    bool causal_f16 = false; // exact 0/-inf shared mask; index score casts its tile to F32
    bool gpu_prefill_chain = false; // Engine-owned device hidden ping-pong pair
    uint64_t retained_hidden_bytes = 0;
    bool native_gpu_ffn = false; // persistent stock GGML expert gallocr
    bool fast_gpu_ffn = false; // route gate/up/OAI/down tensors remain on device
    bool fast_rank_phases = false; // one original rank at a time, also removes full routed matrix
    bool fast_fused_reduction = false; // direct weighted accumulation removes full routed output
    int fast_expert_rows = 1024; // executor slices every expert, preserving all routes
    uint64_t retained_dense_bytes = 0, retained_routed_bytes = 0;
    bool f16_mask = false; // native F16 get_rows and F16 causal-add path
    bool top_k_mask = false; // direct F16 mask instead of expanded selected/transpose buffers
    bool inplace_score = false; // raw index score + causal mask share one allocation
    int64_t position = 0, remaining = 0, context = 0;
};
struct PrefillPlan {
    int chunk = 0;
    int64_t padded_keys = 0;
    uint64_t worst_distinct_experts = 0, scratch_estimate = 0, headroom_bytes = 0;
    uint64_t persistent_scratch_estimate = 0, dense_scratch_estimate = 0, routed_scratch_estimate = 0;
    uint64_t score_stage_estimate = 0, mask_stage_estimate = 0;
    int attention_query_rows = 0;
    std::string reason;
};
inline uint64_t prefill_product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::runtime_error("M31 prefill budget multiplication overflow");
    return a * b;
}
inline uint64_t prefill_sum(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::runtime_error("M31 prefill budget addition overflow");
    return a + b;
}
inline PrefillPlan plan_prefill(const Geometry & g, const PrefillBudget & b) {
    if (g.embd <= 0 || g.expert_ff <= 0 || g.dense_ff <= 0 || g.heads <= 0 ||
        g.head_dim <= 0 || g.index_heads <= 0 || g.index_dim <= 0 || g.used <= 0 ||
        g.experts < g.used || g.block <= 0 || g.block > 256 || 256 % g.block ||
        b.max_chunk < 1 || b.max_chunk > 32768 || b.attention_query_tile < 0 ||
        b.indexer_query_tile < 0 || b.indexer_query_tile > 8192 ||
        b.fast_expert_rows < 1 || b.fast_expert_rows > 32768 ||
        b.attention_query_tile > 8192 || b.position < 0 || b.remaining < 0 ||
        b.context < b.position || b.context > INT32_MAX)
        throw std::runtime_error("M31 prefill budget invalid geometry or request");
    PrefillPlan out;
    if (!b.remaining || b.position == b.context) { out.reason = "no remaining context rows"; return out; }
    const uint64_t reserved = prefill_sum(b.reserve_bytes, b.extra_scratch_bytes);
    if (reserved >= b.free_after_resident) { out.reason = "free VRAM does not cover fixed reserves"; return out; }
    const uint64_t scratch_budget = b.free_after_resident - reserved;
    int upper = int(std::min<int64_t>({b.max_chunk, b.remaining, b.context - b.position}));
    // A full 128-expert pool admits any token batch. Smaller pools must bound
    // the worst-case union; the router is never altered to satisfy the budget.
    if (b.streaming_slots < uint64_t(g.experts))
        upper = std::min(upper, int(b.streaming_slots / uint64_t(g.used)));
    if (!upper) { out.reason = "too few unprotected expert slots for one token"; return out; }
    // Keep normal prefill windows block-aligned, retaining a short prompt/tail.
    if (upper >= g.block && b.remaining > upper) upper = int(upper / g.block * g.block);
    for (int n = upper; n >= 1; --n) {
        if (n >= g.block && b.remaining > n && n % g.block) continue;
        const uint64_t nk = uint64_t((b.position + n + 255) / 256 * 256);
        const int attention_rows = b.attention_query_tile ? std::min(n,b.attention_query_tile) : n;
        const int score_rows = b.indexer_query_tile ? std::min(attention_rows,b.indexer_query_tile) : attention_rows;
        const uint64_t score_elements = prefill_product(prefill_product(nk, uint64_t(g.index_heads)), uint64_t(score_rows));
        const uint64_t mask_elements = prefill_product(prefill_product(nk,uint64_t(g.index_heads)),uint64_t(attention_rows));
        // Matches Engine::attention's int32 indexing bound before byte sizing.
        if (score_elements >= uint64_t(INT32_MAX)) continue;
        const uint64_t score = prefill_product(score_elements, 4);
        uint64_t scratch=0, persistent=0, dense_stage=0, routed_stage=0;
        uint64_t score_stage=0, mask_stage=0;
        const bool staged_attention = b.attention_query_tile || b.indexer_query_tile || b.top_k_mask || b.inplace_score;
        const bool stage_budget = staged_attention || b.fast_gpu_ffn || b.native_gpu_ffn || b.gpu_prefill_chain || b.retained_hidden_bytes;
        if (!stage_budget) {
            // Preserve the previous conservative envelope for the original path.
            scratch = prefill_product(score, 3);
            scratch = prefill_sum(scratch, prefill_product(prefill_product(nk, uint64_t(n)), 8));
            scratch = prefill_sum(scratch, prefill_product(prefill_product(nk, uint64_t(g.index_dim)), 4));
            const uint64_t routed_row = prefill_product(uint64_t(g.used),
                prefill_sum(prefill_product(uint64_t(g.expert_ff), 3), prefill_product(uint64_t(g.embd), 3)));
            const uint64_t dense_row = prefill_sum(prefill_product(uint64_t(g.embd), 8),
                prefill_sum(prefill_product(uint64_t(g.dense_ff), 3),
                            prefill_product(prefill_product(uint64_t(g.heads), uint64_t(g.head_dim)), 2)));
            scratch = prefill_sum(scratch, prefill_product(prefill_sum(routed_row, dense_row), uint64_t(n) * 4));
        } else {
            // Shared full-chunk causal mask/bias/positions live outside gallocr.
            persistent = prefill_product(prefill_product(nk,uint64_t(n)),b.causal_f16?2:4);
            persistent = prefill_sum(persistent,prefill_product(prefill_product(nk/uint64_t(g.block),uint64_t(n)),4));
            persistent = prefill_sum(persistent,prefill_product(prefill_sum(nk,uint64_t(n)),4));
            const uint64_t hidden_pair = b.gpu_prefill_chain ? prefill_product(prefill_product(uint64_t(g.embd),uint64_t(n)),8) : 0;
            persistent = prefill_sum(persistent,std::max(hidden_pair,b.retained_hidden_bytes));
            // Score -> pool/top-k and expanded/direct mask -> FA execute in
            // successive stages. Dead score storage can back the mask; do not
            // sum mutually exclusive multi-GiB allocations. A direct TOP_K_MASK
            // writes one F16 tensor and eliminates selected/transpose copies.
            score_stage = prefill_product(score,b.inplace_score?1:2);
            if (b.causal_f16) score_stage = prefill_sum(score_stage,prefill_product(prefill_product(nk,uint64_t(score_rows)),4));
            mask_stage = prefill_product(mask_elements,b.top_k_mask?2:(b.f16_mask?6:14));
            // Index tiles retain only compact top-k IDs. The final mask and FA
            // keep full-query geometry; no per-tile expanded masks coexist.
            if (b.indexer_query_tile) {
                const uint64_t top_ids = prefill_product(prefill_product(uint64_t(std::max<int64_t>(1,g.top_blocks)),uint64_t(g.index_heads)),uint64_t(n)*4);
                persistent = prefill_sum(persistent,top_ids);
            }
            uint64_t attention = std::max(score_stage,mask_stage);
            const uint64_t block_elements = prefill_product(prefill_product(nk/uint64_t(g.block),uint64_t(g.index_heads)),uint64_t(score_rows));
            attention = prefill_sum(attention,prefill_product(block_elements,16));
            attention = prefill_sum(attention,prefill_product(prefill_product(nk,uint64_t(g.index_dim)),4));
            // Full-chunk projection/residual/concat storage remains full n.
            // Dense FFN and attention temporaries peak in different graph stages;
            // their maxima, rather than their sum, determine gallocr capacity.
            uint64_t attention_row = prefill_product(uint64_t(g.embd),b.gpu_prefill_chain?4:5);
            attention_row = prefill_sum(attention_row,prefill_product(prefill_product(uint64_t(g.heads),uint64_t(g.head_dim)),4));
            attention_row = prefill_sum(attention_row,prefill_product(prefill_product(uint64_t(g.index_heads),uint64_t(g.head_dim)),2));
            attention_row = prefill_sum(attention_row,prefill_product(uint64_t(g.index_dim),uint64_t(g.index_heads)+1));
            const uint64_t full_attention = prefill_product(attention_row,uint64_t(n)*4);
            const uint64_t shared_ff = prefill_product(uint64_t(g.expert_ff),uint64_t(std::max<int64_t>(1,g.shared)));
            const uint64_t ffn_row = prefill_sum(prefill_product(uint64_t(g.embd),b.gpu_prefill_chain?3:4),prefill_product(std::max(uint64_t(g.dense_ff),shared_ff),3));
            dense_stage = std::max(prefill_sum(full_attention,attention),prefill_product(ffn_row,uint64_t(n)*4));
            // Exact CUDA has one high-water route output and quant input. The
            // gate/up/OAI host vectors are CPU memory, not additional VRAM.
            routed_stage = prefill_product(prefill_product(uint64_t(std::max(g.embd,g.expert_ff)),uint64_t(g.used)),uint64_t(n)*4);
            const uint64_t quant_row = std::max(uint64_t(g.embd),prefill_product(uint64_t(g.expert_ff),uint64_t(g.used)));
            routed_stage = prefill_sum(routed_stage,prefill_product(quant_row,uint64_t(n)*2));
            routed_stage = prefill_sum(routed_stage,prefill_product(prefill_product(uint64_t(g.used),uint64_t(n)),32));
            if (b.fast_gpu_ffn) {
                // Compensated F16/BF16 GEMM executor keeps input/output, routed
                // accumulation and per-expert dequant/activation scratch. Each
                // token selects an expert once, so max routes/expert <= n.
                // The executor chunks even a maximally popular expert into
                // bounded local GEMMs; this is not a mean routing estimate.
                const uint64_t global_row = prefill_sum(
                    prefill_product(uint64_t(g.embd),uint64_t(2-(b.gpu_prefill_chain?1:0)+((b.fast_fused_reduction||b.fast_rank_phases)?0:g.used))*4),
                    prefill_product(uint64_t(g.used),8));
                const uint64_t local_row = prefill_sum(
                    prefill_product(uint64_t(g.embd),8),
                    prefill_product(uint64_t(g.expert_ff),12));
                routed_stage = prefill_sum(prefill_product(global_row,uint64_t(n)),
                    prefill_sum(prefill_product(local_row,uint64_t(std::min(n,b.fast_expert_rows))),
                    prefill_product(prefill_product(uint64_t(g.embd),uint64_t(g.expert_ff)),4)));
                routed_stage = std::max(routed_stage,b.retained_routed_bytes);
                dense_stage = std::max(dense_stage,b.retained_dense_bytes);
            }
            if (b.native_gpu_ffn) {
                // Grouped GGML gate/up/OAI, down/weighting and rank-reduction
                // reuse graph storage. Keep a conservative live-stage maximum,
                // not a sum of every mutually dead tensor. Input borrowed by
                // the device chain removes one F32 embd*n allocation.
                const uint64_t ff_routes=prefill_product(uint64_t(g.expert_ff),uint64_t(g.used));
                const uint64_t down_routes=prefill_product(uint64_t(g.embd),uint64_t(g.used));
                const uint64_t rows=std::max({prefill_product(ff_routes,3),
                    prefill_sum(ff_routes,down_routes),prefill_sum(down_routes,prefill_product(uint64_t(g.embd),3))});
                const uint64_t with_input=prefill_sum(rows,b.gpu_prefill_chain?0:uint64_t(g.embd));
                routed_stage=std::max(prefill_product(with_input,uint64_t(n)*4),b.retained_routed_bytes);
                dense_stage=std::max(dense_stage,b.retained_dense_bytes);
                // Existing extra_scratch_bytes covers the independent vendor
                // MMQ/cublas pool, not counted in either gallocr's own bytes.
            }
            // Fast executor buffers remain allocated while the persistent
            // dense gallocr retains its previous high water. These independent
            // owners coexist and must be added; max is valid only for old
            // transient routed-stage estimates covered by the exact reserve.
            scratch = prefill_sum(persistent,(b.fast_gpu_ffn||b.native_gpu_ffn) ? prefill_sum(dense_stage,routed_stage) : std::max(dense_stage,routed_stage));
        }
        if (scratch > scratch_budget) continue;
        out.chunk = n;
        out.padded_keys = int64_t(nk);
        out.worst_distinct_experts = std::min(uint64_t(g.experts), uint64_t(g.used) * uint64_t(n));
        out.scratch_estimate = scratch;
        out.persistent_scratch_estimate = persistent;
        out.dense_scratch_estimate = dense_stage;
        out.routed_scratch_estimate = routed_stage;
        out.score_stage_estimate = score_stage;
        out.mask_stage_estimate = mask_stage;
        out.attention_query_rows = attention_rows;
        out.headroom_bytes = scratch_budget - scratch;
        out.reason = (b.fast_gpu_ffn||b.native_gpu_ffn)
            ? "proposal fits full causal storage plus coexisting retained dense/fast GPU buffers; allocator gate still required"
            : staged_attention ? "staged attention proposal fits full causal storage plus max GPU stage; allocator gate still required"
            : "proposal fits conservative scratch and worst-case route union; allocator gate still required";
        return out;
    }
    out.reason = "no chunk fits estimated scratch";
    return out;
}
} // namespace strata::m31
