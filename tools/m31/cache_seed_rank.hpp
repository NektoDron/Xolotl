// Dependency-free rank reader. Expected hashes must come from a trusted preflight
// receipt; reading a hash out of the same unverified binary is not verification.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31::cache_seed {

using Digest = std::array<uint8_t, 32>;
inline void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(std::string("M31 seed rank: ") + message);
}
inline Digest hex_digest(const std::string & text) {
    require(text.size() == 64, "expected a 64-character SHA256");
    auto digit = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return uint8_t(c - '0');
        if (c >= 'a' && c <= 'f') return uint8_t(c - 'a' + 10);
        throw std::runtime_error("M31 seed rank: invalid canonical hex SHA256");
    };
    Digest out{};
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = uint8_t((digit(text[2*i]) << 4) | digit(text[2*i+1]));
    return out;
}
inline std::string digest_hex(const Digest & value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (auto v : value) { out += hex[v >> 4]; out += hex[v & 15]; }
    return out;
}

// SHA-256 over a bounded small rank file, not over GGUF model weights.
inline Digest sha256(const uint8_t * input, size_t count) {
    static constexpr uint32_t k[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    require(count <= 64u * 1024u * 1024u, "SHA256 input exceeds rank bound");
    std::vector<uint8_t> bytes;
    bytes.reserve(count + 72);
    if (count) { require(input != nullptr, "null SHA256 input"); bytes.assign(input, input + count); }
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    const uint64_t bits = uint64_t(count) * 8;
    for (int i = 7; i >= 0; --i) bytes.push_back(uint8_t(bits >> (8*i)));
    std::array<uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                 0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto rotate = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | bytes[offset + 4*i + j];
        for (int i = 16; i < 64; ++i) {
            const auto a = w[i-15], b = w[i-2];
            w[i] = w[i-16] + (rotate(a,7)^rotate(a,18)^(a>>3)) + w[i-7] + (rotate(b,17)^rotate(b,19)^(b>>10));
        }
        auto a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotate(e,6)^rotate(e,11)^rotate(e,25)) + ((e&f)^((~e)&g)) + k[i] + w[i];
            const uint32_t t2 = (rotate(a,2)^rotate(a,13)^rotate(a,22)) + ((a&b)^(a&c)^(b&c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
    Digest result{};
    for (size_t i = 0; i < state.size(); ++i)
        for (size_t j = 0; j < 4; ++j) result[4*i+j] = uint8_t(state[i] >> (24-8*j));
    return result;
}

struct ExpectedRank {
    uint32_t layers = 0, experts = 0, dense_lead = 0;
    Digest identity{}, file_sha256{};
};
struct SeedBudget {
    size_t hot = 0, maximum_hot = 0, recommended_hot = 0, stream = 0;
};
inline SeedBudget budget(size_t slots, uint32_t experts, uint32_t layers, uint32_t leading,
                         uint32_t used, size_t requested = std::numeric_limits<size_t>::max()) {
    require(layers > leading && experts && used && used <= experts, "invalid seed geometry");
    const uint64_t total = uint64_t(layers-leading) * experts;
    require(slots >= experts && slots <= total, "slots cannot reserve one complete layer working set");
    const size_t maximum = slots-experts;
    const uint64_t working = uint64_t(layers-leading)*used;
    const size_t recommendation = std::min(maximum, slots > working ? slots-size_t(working) : size_t(0));
    const size_t hot = requested == std::numeric_limits<size_t>::max() ? recommendation : requested;
    require(hot <= maximum, "hot seeds leave too few stream slots");
    return {hot, maximum, recommendation, slots-hot};
}

struct Rank {
    uint32_t layers = 0, experts = 0, dense_lead = 0;
    Digest identity{};
    std::vector<uint32_t> ids; // Complete original layer*experts+expert permutation.
};
inline uint32_t read_u32(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline Rank read(const std::string & path, const ExpectedRank & expected) {
    require(expected.layers && expected.layers <= 1024 && expected.experts && expected.experts <= 4096 &&
            expected.dense_lead < expected.layers, "invalid expected geometry");
    const size_t count = size_t(expected.layers-expected.dense_lead) * expected.experts;
    const size_t exact_bytes = 56 + count * 4;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open rank file");
    require(file.tellg() == std::streamoff(exact_bytes), "truncated rank or trailing bytes");
    file.seekg(0);
    std::vector<uint8_t> bytes(exact_bytes);
    file.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size()));
    require(bool(file), "short rank read");
    require(sha256(bytes.data(), bytes.size()) == expected.file_sha256, "rank content SHA256 mismatch");
    const std::array<uint8_t,8> magic{'M','3','1','R','A','N','K',0};
    require(std::equal(magic.begin(),magic.end(),bytes.begin()), "invalid rank magic");
    require(read_u32(bytes.data()+8) == 1, "unsupported rank version");
    require(read_u32(bytes.data()+12) == expected.layers && read_u32(bytes.data()+16) == expected.experts &&
            read_u32(bytes.data()+20) == count, "rank geometry/count mismatch");
    require(std::equal(expected.identity.begin(),expected.identity.end(),bytes.begin()+24), "rank identity mismatch");
    Rank out{expected.layers, expected.experts, expected.dense_lead, expected.identity, {}};
    out.ids.reserve(count);
    std::vector<bool> seen(count, false);
    const uint32_t first = expected.dense_lead * expected.experts;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = read_u32(bytes.data()+56+4*i);
        require(id >= first && uint64_t(id) < uint64_t(expected.layers)*expected.experts,
                "rank contains dense/out-of-range expert ID");
        require(!seen[id-first], "duplicate rank expert ID");
        seen[id-first] = true;
        out.ids.push_back(id);
    }
    return out;
}
} // namespace strata::m31::cache_seed
