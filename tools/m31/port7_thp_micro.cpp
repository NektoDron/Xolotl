// Limited own-arena THP feasibility probe. No GPU, mlock, sysctl or hugetlb.
#include "prefill_thp_mapping.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: port7_thp_micro GiB threads");
        const int gib = std::stoi(argv[1]), threads = std::stoi(argv[2]);
        if ((gib != 16 && gib != 32) || threads < 1 || threads > 16)
            throw std::runtime_error("probe is limited to16/32GiB and1..16threads");
        using Clock = std::chrono::steady_clock;
        using Mapping = strata::m31::PrefillThpMapping;
        auto policy = Mapping::system_policy();
        policy.force = true;
        const size_t bytes = size_t(gib) << 30;
        const auto start = Clock::now();
        auto mapping = Mapping::create(bytes, policy);
        if (!mapping->data()) throw std::runtime_error("forced own mapping was unavailable");
        // Stream a marker before touching: timeout preserves why a trial stalled.
        std::cerr << "touching own bytes=" << bytes << " threads=" << threads
                  << " free_huge_bytes=" << policy.free_huge_bytes << std::endl;
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < threads; ++i) workers.emplace_back([&] {
            auto * data = static_cast<volatile unsigned char *>(mapping->data());
            for (;;) {
                const size_t offset = next.fetch_add(Mapping::huge);
                if (offset >= bytes) break;
                // Cover every4KiB, including kernels choosing base-page fallback.
                for (size_t row = 0; row < Mapping::huge; row += 4096)
                    data[offset + row] = 0x6a;
            }
        });
        for (auto & worker : workers) worker.join();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const auto actual = mapping->anon_huge_pages();
        std::cout << "{\"scope\":\"limited own anonymous mapping feasibility; not pinned arena or model speed\","
                  << "\"bytes\":" << bytes << ",\"threads\":" << threads
                  << ",\"seconds\":" << seconds
                  << ",\"forced\":true,\"advised\":" << (mapping->stats().advised ? "true" : "false")
                  << ",\"free_huge_bytes_before\":" << policy.free_huge_bytes
                  << ",\"anon_huge_bytes\":" << actual.bytes
                  << ",\"anon_huge_exact\":" << (actual.exact ? "true" : "false")
                  << ",\"global_changes\":false,\"cuda_registration_tested\":false}\n";
        // A measured zero-THP result is a completed negative experiment.
        return actual.exact ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
