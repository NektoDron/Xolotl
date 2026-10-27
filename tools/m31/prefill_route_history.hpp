// Last large-prefill routes are speculative hints only. A missing prediction
// never suppresses the ordinary current-layer load of any actual expert.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
class PrefillRouteHistory {
public:
    PrefillRouteHistory(size_t layers, size_t experts)
        : experts_(experts), masks_(layers), known_(layers, false) {
        if (!layers || !experts) throw std::invalid_argument("empty route-history geometry");
    }
    void remember(size_t layer, int tokens, const std::vector<uint8_t> & selected) {
        if (layer >= masks_.size() || selected.size() != experts_)
            throw std::invalid_argument("route-history geometry differs");
        for (auto value : selected)
            if (value > 1) throw std::invalid_argument("invalid route-history mask");
        // Decode must not overwrite a useful large-prefill prediction.
        if (tokens < 128) return;
        masks_[layer] = selected;
        known_[layer] = true;
    }
    bool has(size_t layer) const { return known_.at(layer); }
    bool predicts(size_t layer, size_t expert) const {
        if (expert >= experts_) throw std::out_of_range("route-history expert");
        return !known_.at(layer) || masks_[layer][expert] != 0;
    }
private:
    size_t experts_;
    std::vector<std::vector<uint8_t>> masks_;
    std::vector<bool> known_;
};
} // namespace strata::m31
