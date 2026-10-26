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
    int max_chunk = 2048;
    int64_t position = 0, remaining = 0, context = 0;
};
struct PrefillPlan {
    int chunk = 0;
    int64_t padded_keys = 0;
    uint64_t worst_distinct_experts = 0, scratch_estimate = 0, headroom_bytes = 0;
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
        b.max_chunk < 1 || b.max_chunk > 2048 || b.position < 0 || b.remaining < 0 ||
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
        const uint64_t score = prefill_product(prefill_product(nk, uint64_t(g.index_heads)), uint64_t(n) * 4);
        // Separate score/dot/scatter temporaries and causal/expanded masks.
        uint64_t scratch = prefill_product(score, 3);
        scratch = prefill_sum(scratch, prefill_product(prefill_product(nk, uint64_t(n)), 8));
        scratch = prefill_sum(scratch, prefill_product(prefill_product(nk, uint64_t(g.index_dim)), 4));
        const uint64_t routed_row = prefill_product(uint64_t(g.used),
            prefill_sum(prefill_product(uint64_t(g.expert_ff), 3), prefill_product(uint64_t(g.embd), 3)));
        const uint64_t dense_row = prefill_sum(prefill_product(uint64_t(g.embd), 8),
            prefill_sum(prefill_product(uint64_t(g.dense_ff), 3),
                        prefill_product(prefill_product(uint64_t(g.heads), uint64_t(g.head_dim)), 2)));
        scratch = prefill_sum(scratch, prefill_product(prefill_sum(routed_row, dense_row), uint64_t(n) * 4));
        if (scratch > scratch_budget) continue;
        out.chunk = n;
        out.padded_keys = int64_t(nk);
        out.worst_distinct_experts = std::min(uint64_t(g.experts), uint64_t(g.used) * uint64_t(n));
        out.scratch_estimate = scratch;
        out.headroom_bytes = scratch_budget - scratch;
        out.reason = "proposal fits conservative scratch and worst-case route union; allocator gate still required";
        return out;
    }
    out.reason = "no chunk fits estimated scratch";
    return out;
}
} // namespace strata::m31
