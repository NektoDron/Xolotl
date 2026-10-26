// Small CPU-only entry point used to test the public header against preflight receipts.
#include "cache_seed_rank.hpp"
#include <iostream>
#include <map>

int main(int argc, char ** argv) {
    using namespace strata::m31::cache_seed;
    try {
        std::map<std::string,std::string> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing option value");
            require(args.emplace(argv[i],argv[i+1]).second, "duplicate option");
        }
        auto number = [&](const std::string & name) {
            const auto & s = args.at(name);
            require(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos, "invalid integer");
            return std::stoull(s);
        };
        if (args.count("--hash-text")) {
            const auto & text = args.at("--hash-text");
            std::cout << digest_hex(sha256(reinterpret_cast<const uint8_t *>(text.data()),text.size())) << '\n';
            return 0;
        }
        const uint64_t layers = number("--layers"), experts = number("--experts"), leading = number("--leading");
        require(layers <= 1024 && experts <= 4096 && leading < layers, "expected dimensions out of bounds");
        const ExpectedRank expected{uint32_t(layers),uint32_t(experts),uint32_t(leading),
            hex_digest(args.at("--identity")),hex_digest(args.at("--sha256"))};
        const auto rank = read(args.at("--rank"),expected);
        const auto requested = args.count("--hot") ? size_t(number("--hot")) : std::numeric_limits<size_t>::max();
        const auto used = number("--used");
        require(used > 0 && used <= experts, "used experts out of bounds");
        const auto plan = budget(size_t(number("--slots")),uint32_t(experts),uint32_t(layers),uint32_t(leading),
                                 uint32_t(used),requested);
        std::cout << "{\"verified\":true,\"count\":" << rank.ids.size() << ",\"hot\":" << plan.hot
                  << ",\"stream\":" << plan.stream << ",\"max_hot\":" << plan.maximum_hot << "}\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
