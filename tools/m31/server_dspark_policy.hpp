#pragma once
#include <string>
namespace strata::m31 {
struct DSparkRequestPolicy {
    bool enabled = false;
    std::string reason = "disabled";
};
inline DSparkRequestPolicy dspark_request_policy(bool configured, bool cuda,
        bool persistent_pool, int expert_threshold, int target_layers,
        double temperature, bool grammar, int maximum) {
    if (!configured) return {};
    if (!cuda) return {false, "backend"};
    if (!persistent_pool) return {false, "persistent_cpu_pool_required"};
    if (expert_threshold <= 8) return {false, "expert_threshold"};
    if (target_layers < 60) return {false, "target_geometry"};
    if (temperature != 0) return {false, "sampling_temperature"};
    if (grammar) return {false, "grammar_check"};
    if (maximum <= 1) return {false, "max_tokens_le_one"};
    return {true, ""};
}
} // namespace strata::m31
