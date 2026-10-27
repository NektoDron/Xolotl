#pragma once
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::m31 {
// Opt-in prefill routing. Read once before each append, so a calibration process
// can compare policies against the same loaded model and allocation state.
struct ShortPrefillPolicy {
    enum class Mode { off, cpu, hybrid, automatic };
    Mode mode = Mode::off;
    int maximum = 1024;
    int cpu_maximum = 31;
    static int bounded(const char * name, int fallback) {
        const char * value = std::getenv(name);
        if (!value || !*value) return fallback;
        char * end = nullptr;
        errno = 0;
        const long result = std::strtol(value, &end, 10);
        if (errno || !end || *end || result < 0 || result > 32768)
            throw std::invalid_argument(std::string("M31: invalid ") + name);
        return int(result);
    }
    static ShortPrefillPolicy environment() {
        ShortPrefillPolicy policy;
        const char * value = std::getenv("M31_SHORT_PREFILL_MODE");
        const std::string mode = value ? value : "off";
        if (mode == "off") return policy;
        if (mode == "cpu") policy.mode = Mode::cpu;
        else if (mode == "hybrid") policy.mode = Mode::hybrid;
        else if (mode == "auto") policy.mode = Mode::automatic;
        else throw std::invalid_argument("M31: M31_SHORT_PREFILL_MODE must be off, cpu, hybrid or auto");
        policy.maximum = bounded("M31_SHORT_PREFILL_MAX_N", policy.maximum);
        policy.cpu_maximum = bounded("M31_SHORT_PREFILL_CPU_MAX_N", policy.cpu_maximum);
        return policy;
    }
    Mode select(int n) const {
        // Single-token decode is deliberately outside this policy.
        if (n <= 1 || n > maximum) return Mode::off;
        if (mode == Mode::automatic) return n <= cpu_maximum ? Mode::cpu : Mode::hybrid;
        return mode;
    }
};
} // namespace strata::m31
