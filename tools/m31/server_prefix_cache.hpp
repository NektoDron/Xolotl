#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::m31 {

struct PromptChunk {
    int64_t position = 0;
    int tokens = 0;
    int64_t padded_keys = 0;
};

// One immutable text prompt, owned by the worker that also owns Engine.
// Version1 deliberately accepts ONLY a single complete prefill chunk. This
// makes the current planner_at(0) check an exact fresh-pass geometry check:
// there is no unexecuted intermediate allocation that could change its budget.
// Decode rows, sampled/penalized logits and partial-prefix guesses never enter
// this cache. The model/config cannot change during the worker's lifetime.
class ExactPromptCache {
public:
    void clear() { ids_.clear(); raw_logits_.clear(); chunk_ = {}; }

    bool store(const std::vector<int32_t> & ids, const std::vector<PromptChunk> & recipe,
               const std::vector<float> & raw_logits, int vocab) {
        clear();
        if (ids.empty() || recipe.size() != 1 || vocab <= 0 ||
            raw_logits.size() != std::size_t(vocab) || recipe[0].position != 0 ||
            recipe[0].tokens != int64_t(ids.size()) || recipe[0].padded_keys < int64_t(ids.size())) return false;
        ids_ = ids;
        raw_logits_ = raw_logits;
        chunk_ = recipe[0];
        return true;
    }

    bool matches(const std::vector<int32_t> & ids, int planned_first_chunk,
                 int64_t current_position, int vocab) const {
        return !ids_.empty() && ids == ids_ && planned_first_chunk == chunk_.tokens &&
               current_position >= int64_t(ids_.size()) && vocab > 0 &&
               raw_logits_.size() == std::size_t(vocab);
    }

    // Return a copy before handing the vector to the sampler. Cached raw values
    // must remain unchanged even if a future sampler applies in-place filters.
    std::vector<float> raw_logits() const { return raw_logits_; }
    const PromptChunk & chunk() const { return chunk_; }
    bool empty() const { return ids_.empty(); }

private:
    std::vector<int32_t> ids_;
    std::vector<float> raw_logits_;
    PromptChunk chunk_;
};

} // namespace strata::m31
