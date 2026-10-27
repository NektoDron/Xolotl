// Standalone proposal only: no ggml/runtime integration. Not built or run yet.
// g++-14 -std=c++20 -O2 -pthread this.cpp -o isolated-mwaitx-probe
// Run only in a coordinator-approved CPU window, under an external timeout.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#include <immintrin.h>
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static_assert(std::atomic<uint32_t>::is_always_lock_free);
using Clock = std::chrono::steady_clock;
struct alignas(128) Word { std::atomic<uint32_t> value{0}; };

// Baseline TU has no -march=native/-mmwaitx; feature instructions are isolated.
__attribute__((target("mwaitx"), noinline))
static void monitor_wait(Word &word, uint32_t expected, uint32_t clocks, unsigned hint) {
    _mm_monitorx(&word.value, 0, 0);
    // Monitor must be armed BEFORE checking the condition a second time.
    if (word.value.load(std::memory_order_acquire) == expected)
        _mm_mwaitx(2, hint, clocks); // intrinsic: ECX, EAX, EBX. clocks MUST != 0.
}

static bool cpuid_ok() {
    unsigned a, b, c, d;
    if (__get_cpuid_max(0x80000000, nullptr) < 0x80000001) return false;
    __cpuid(0x80000001, a, b, c, d);
    return (c & (1u << 29)) != 0;
}

// Probe before creating threads. Parent gets a clean fallback on SIGILL/#GP
// (Linux SIGSEGV), rather than installing a process-wide SIGILL handler in ggml.
static bool runtime_probe(int &status) {
    if (!cpuid_ok()) { status = -1; return false; }
    const pid_t child = fork();
    if (child < 0) { status = -2; return false; }
    if (child == 0) {
        alarm(1);
        Word word;
        monitor_wait(word, 0, 10000, 0xf0);
        _exit(0);
    }
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (waitpid(child, &status, WNOHANG) == 0) {
        if (Clock::now() > deadline) {
            kill(child, SIGKILL); waitpid(child, &status, 0); return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static void pin(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) std::exit(3);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) { std::perror("affinity"); std::exit(3); }
}

static uint64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    // mode=0 PAUSE, 1 timed MWAITX C0, 2 timed MWAITX C1. No defaults that
    // execute a feature instruction before an explicit caller selects a mode.
    if (argc != 8) {
        std::fprintf(stderr, "usage: probe mode sender_cpu waiter_cpu iterations gap_ns clocks forced_fallback\n");
        return 2;
    }
    const int mode = std::atoi(argv[1]);
    const int sender_cpu = std::atoi(argv[2]), waiter_cpu = std::atoi(argv[3]);
    const unsigned iterations = std::strtoul(argv[4], nullptr, 10);
    const uint64_t gap_ns = std::strtoull(argv[5], nullptr, 10);
    const uint32_t clocks = std::strtoul(argv[6], nullptr, 10);
    const bool forced_fallback = std::atoi(argv[7]) != 0;
    if (mode < 0 || mode > 2 || iterations < 100 || iterations > 1000000 || clocks == 0) return 2;
    int status = -3;
    const bool supported = !forced_fallback && (mode == 0 || runtime_probe(status));
    const bool use_wait = mode != 0 && supported;
    Word generation, ack;
    std::atomic<uint64_t> published_ns{0};
    std::vector<uint64_t> delays(iterations);
    std::atomic<unsigned> migrations{0};
    uint64_t waits = 0;
    std::thread waiter([&] {
        pin(waiter_cpu);
        for (unsigned i = 1; i <= iterations; ++i) {
            while (generation.value.load(std::memory_order_acquire) != i) {
                if (use_wait) { monitor_wait(generation, i - 1, clocks, mode == 1 ? 0xf0 : 0); ++waits; }
                else _mm_pause();
            }
            const uint64_t seen = ns();
            delays[i - 1] = seen - published_ns.load(std::memory_order_relaxed);
            if (sched_getcpu() != waiter_cpu) ++migrations;
            ack.value.store(i, std::memory_order_release);
        }
    });
    pin(sender_cpu);
    const uint64_t started = ns();
    for (unsigned i = 1; i <= iterations; ++i) {
        const uint64_t due = ns() + gap_ns;
        while (ns() < due) _mm_pause();
        published_ns.store(ns(), std::memory_order_relaxed);
        generation.value.store(i, std::memory_order_release);
        while (ack.value.load(std::memory_order_acquire) != i) _mm_pause();
    }
    waiter.join();
    const uint64_t elapsed = ns() - started;
    std::sort(delays.begin(), delays.end());
    const auto pct = [&](size_t p) { return delays[std::min(delays.size() - 1, delays.size() * p / 100)]; };
    std::printf("{\"mode\":%d,\"used_mwaitx\":%s,\"cpuid\":%s,\"probe_status\":%d,\"iterations\":%u,\"gap_ns\":%llu,\"clocks\":%u,\"wake_ns_p50\":%llu,\"wake_ns_p95\":%llu,\"wake_ns_p99\":%llu,\"elapsed_ns\":%llu,\"wait_calls\":%llu,\"migrations\":%u,\"scope\":\"publish-to-observe; component only; timestamp+coherence included\"}\n",
        mode, use_wait ? "true" : "false", cpuid_ok() ? "true" : "false", status, iterations,
        (unsigned long long)gap_ns, clocks, (unsigned long long)pct(50),
        (unsigned long long)pct(95), (unsigned long long)pct(99),
        (unsigned long long)elapsed, (unsigned long long)waits, migrations.load());
}
#else
int main() { std::puts("{\"supported\":false,\"reason\":\"Linux GNU-compatible x86_64 probe only\"}"); }
#endif
