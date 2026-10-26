#pragma once
#include "ggml.h"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

namespace strata::m31 {
// Explicit tuning only: an absent variable retains ggml's exact defaults.
inline void apply_cpu_pool_environment(ggml_threadpool_params & params) {
    auto number = [](const char * begin, const char * end, unsigned maximum) {
        unsigned value = 0;
        auto result = std::from_chars(begin, end, value);
        if (begin == end || result.ec != std::errc{} || result.ptr != end || value > maximum)
            throw std::invalid_argument("M31 CPU pool: invalid affinity/poll value");
        return value;
    };
    if (const char * text = std::getenv("M31_CPU_POLL"))
        params.poll = number(text, text + std::strlen(text), 100);
    if (const char * text = std::getenv("M31_CPU_AFFINITY")) {
        std::fill(std::begin(params.cpumask), std::end(params.cpumask), false);
        const char * end = text + std::strlen(text);
        if (text == end) throw std::invalid_argument("M31 CPU pool: empty affinity");
        while (text < end) {
            const char * separator = std::find(text, end, ',');
            const char * dash = std::find(text, separator, '-');
            unsigned first = number(text, dash, GGML_MAX_N_THREADS - 1);
            unsigned last = dash == separator ? first : number(dash + 1, separator, GGML_MAX_N_THREADS - 1);
            if (first > last) throw std::invalid_argument("M31 CPU pool: reversed affinity range");
            for (unsigned cpu = first; cpu <= last; ++cpu) params.cpumask[cpu] = true;
            if (separator == end) break;
            text = separator + 1;
            if (text == end) throw std::invalid_argument("M31 CPU pool: trailing affinity comma");
        }
        if (std::count(std::begin(params.cpumask), std::end(params.cpumask), true) < params.n_threads)
            throw std::invalid_argument("M31 CPU pool: affinity has fewer CPUs than threads");
        params.strict_cpu = true;
    }
}
}
