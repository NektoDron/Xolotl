#include "shortprefill_policy.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    using Policy = strata::m31::ShortPrefillPolicy;
    using Mode = Policy::Mode;
    unsetenv("M31_SHORT_PREFILL_MODE");
    setenv("M31_SHORT_PREFILL_MAX_N", "invalid", 1);
    assert(Policy::environment().select(515) == Mode::off);
    setenv("M31_SHORT_PREFILL_MODE", "auto", 1);
    unsetenv("M31_SHORT_PREFILL_MAX_N");
    unsetenv("M31_SHORT_PREFILL_CPU_MAX_N");
    const auto calibrated = Policy::environment();
    assert(calibrated.maximum == 1024 && calibrated.cpu_maximum == 31);
    assert(calibrated.select(1) == Mode::off);
    assert(calibrated.select(16) == Mode::cpu);
    assert(calibrated.select(31) == Mode::cpu);
    assert(calibrated.select(32) == Mode::hybrid);
    assert(calibrated.select(515) == Mode::hybrid);
    assert(calibrated.select(1024) == Mode::hybrid);
    assert(calibrated.select(1025) == Mode::off);
    // Explicit tuning still overrides the measured defaults.
    setenv("M31_SHORT_PREFILL_MAX_N", "2048", 1);
    setenv("M31_SHORT_PREFILL_CPU_MAX_N", "128", 1);
    const auto policy = Policy::environment();
    assert(policy.select(1) == Mode::off);
    assert(policy.select(16) == Mode::cpu);
    assert(policy.select(128) == Mode::cpu);
    assert(policy.select(129) == Mode::hybrid);
    assert(policy.select(2048) == Mode::hybrid);
    assert(policy.select(2049) == Mode::off);
    for (const char * mode : {"cpu", "hybrid"}) {
        setenv("M31_SHORT_PREFILL_MODE", mode, 1);
        const auto forced = Policy::environment();
        assert(forced.select(515) == (std::string(mode) == "cpu" ? Mode::cpu : Mode::hybrid));
    }
    for (const char * value : {"-1", "32769", "128junk", "9999999999999999999999999999"}) {
        setenv("M31_SHORT_PREFILL_MAX_N", value, 1);
        bool rejected = false;
        try { (void)Policy::environment(); } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    setenv("M31_SHORT_PREFILL_MODE", "mistyped", 1);
    bool rejected = false;
    try { (void)Policy::environment(); } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    std::cout << "shortprefill policy PASS\n";
}
