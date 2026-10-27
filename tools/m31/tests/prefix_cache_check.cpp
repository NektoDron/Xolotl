#include "server_prefix_cache.hpp"
#include <iostream>
#include <stdexcept>
using namespace strata::m31;
int main() {
    int checks = 0;
    auto check = [&](bool value) { if (!value) throw std::runtime_error("prefix cache check failed"); ++checks; };
    ExactPromptCache cache;
    const std::vector<int32_t> ids{1, 2, 3};
    const std::vector<float> raw{0.1f, -2.f, 4.f, 3.f};
    check(cache.empty());
    check(cache.store(ids, {{0, 3, 256}}, raw, 4));
    check(cache.matches(ids, 3, 5, 4));
    auto sampling = cache.raw_logits(); sampling[0] = 99;
    check(cache.raw_logits() == raw);
    check(!cache.matches({1,2,4}, 3, 5, 4));
    check(!cache.matches(ids, 2, 5, 4));
    check(!cache.matches(ids, 3, 2, 4));
    check(!cache.matches(ids, 3, 5, 5));
    check(!cache.store(ids, {{0,2,256},{2,1,256}}, raw, 4) && cache.empty());
    check(!cache.store(ids, {{1,3,256}}, raw, 4) && cache.empty());
    check(!cache.store(ids, {{0,2,256}}, raw, 4) && cache.empty());
    check(!cache.store(ids, {{0,3,2}}, raw, 4) && cache.empty());
    check(!cache.store(ids, {{0,3,256}}, {}, 4) && cache.empty());
    check(cache.store(ids, {{0,3,256}}, raw, 4));
    cache.clear(); check(cache.empty() && !cache.matches(ids, 3, 5, 4));
    std::cout << "{\"pass\":true,\"checks\":" << checks << "}\n";
}
