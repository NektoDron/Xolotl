#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace strata::m31 {
// A bounded CPU-only residual window. Acquire just before lazy draft append;
// release before another forward. No GPU/backend owner travels with the lease.
class DraftFeatureView {
    std::shared_ptr<const float> storage_;
    std::weak_ptr<const uint64_t> epoch_;
    uint64_t generation_ = 0;
public:
    int64_t first_position = 0;
    int rows = 0;
    size_t width = 0;
    DraftFeatureView() = default;
    DraftFeatureView(std::shared_ptr<const float> storage,
                     const std::shared_ptr<uint64_t> & epoch, int64_t first,
                     int count, size_t columns)
        : storage_(std::move(storage)), epoch_(epoch), generation_(epoch ? *epoch : 0),
          first_position(first), rows(count), width(columns) {}
    bool valid() const noexcept {
        const auto epoch=epoch_.lock();
        return storage_ && epoch && *epoch==generation_;
    }
    const float * data() const {
        if (!valid()) throw std::runtime_error("invalidated draft feature view");
        return storage_.get();
    }
    void clear() { *this=DraftFeatureView{}; }
};
} // namespace strata::m31
