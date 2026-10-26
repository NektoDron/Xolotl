#include "server_sampling.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

static int previous(const std::vector<float> & logits, const std::set<int> & excluded, const std::set<int> * allowed) {
    std::vector<std::pair<float, int>> values;
    values.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
        if (!excluded.count(int(i)) && (!allowed || allowed->count(int(i)))) values.emplace_back(logits[i], int(i));
    }
    if (values.empty()) throw std::runtime_error("empty sampling support");
    return std::max_element(values.begin(), values.end(), [](auto a, auto b) { return a.first < b.first; })->second;
}
static size_t checks = 0;
static void compare(const std::vector<float> & logits, const std::set<int> & excluded, const std::set<int> * allowed) {
    std::mt19937_64 rng(123), original = rng;
    std::string old_error, new_error;
    int old_id = -1, new_id = -1;
    try { old_id = previous(logits, excluded, allowed); } catch (const std::exception & e) { old_error = e.what(); }
    try { new_id = sample(logits, 0, .95, 40, .05, excluded, rng, allowed); } catch (const std::exception & e) { new_error = e.what(); }
    if (old_id != new_id || old_error != new_error || rng != original) throw std::runtime_error("greedy mismatch");
    ++checks;
}
int main(int argc, char ** argv) {
    try {
        std::mt19937_64 random(17);
        for (int iteration = 0; iteration < 512; ++iteration) {
            const int n = 1 + random() % 65;
            std::vector<float> logits(n);
            std::set<int> excluded, allowed;
            for (int i = 0; i < n; ++i) {
                logits[i] = float(int(random() % 7) - 3); // many exact ties
                if (random() % 5 == 0) excluded.insert(i);
                if (random() % 3 == 0) allowed.insert(i);
            }
            compare(logits, excluded, nullptr);
            compare(logits, excluded, &allowed);
        }
        const std::set<int> exclude{0, 1, 2}, allow{1};
        compare({}, {}, nullptr);
        compare({1, 1, 1}, {}, nullptr);
        compare({1, 1, 1}, {0}, nullptr);
        compare({1, 1, 1}, {}, &allow);
        compare({1, 1, 1}, exclude, nullptr);
        for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
            compare({bad, 1, 2}, exclude, &allow); // nonfinite excluded tokens still error
        size_t real_rows = 0;
        for (int a = 1; a < argc; ++a) {
            std::ifstream input(argv[a], std::ios::binary);
            if (!input) throw std::runtime_error("cannot open real logits");
            std::vector<float> row(200064);
            while (input.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float))) {
                compare(row, {}, nullptr);
                compare(row, {200004, 200005, 200020}, nullptr);
                const std::set<int> subset{1, 17, 301, 3011, 17999, 124999, 200059};
                compare(row, {17, 200059}, &subset);
                ++real_rows;
            }
            if (input.gcount() != 0) throw std::runtime_error("partial real logits row");
        }
        std::cout << "greedy parity checks=" << checks << " real_rows=" << real_rows << " tie/exclusion/allowed/nonfinite/RNG=OK\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
