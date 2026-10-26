// Small sealed preflight receipt: no JSON dependency in the engine. This checks
// current model headers/stat; model payload hashes are inherited, not recomputed.
#pragma once
#include "cache_seed_rank.hpp"
#include <filesystem>
#include <sys/stat.h>

namespace strata::m31::cache_seed {
struct ShardReceipt {
    std::string path;
    uint64_t bytes = 0, mtime_ns = 0, header_bytes = 0;
    Digest header_sha256{};
};
struct Receipt {
    ExpectedRank expected;
    uint64_t slots = 0, hot = 0;
    std::string rank_path;
    std::vector<ShardReceipt> shards;
};
class ReceiptCursor {
    const std::vector<uint8_t> & bytes_;
    size_t position_ = 0;
public:
    explicit ReceiptCursor(const std::vector<uint8_t> & bytes) : bytes_(bytes) {}
    const uint8_t * take(size_t count) {
        require(count <= bytes_.size()-position_, "truncated preflight receipt");
        const auto * result = bytes_.data()+position_;
        position_ += count;
        return result;
    }
    uint32_t u32() { return read_u32(take(4)); }
    uint64_t u64() { const auto * p = take(8); return uint64_t(read_u32(p)) | (uint64_t(read_u32(p+4)) << 32); }
    Digest digest() { Digest result{}; const auto * p = take(32); std::copy(p,p+32,result.begin()); return result; }
    std::string string() {
        const auto count = u32();
        require(count && count <= 16384, "invalid receipt path length");
        const auto * p = take(count);
        require(std::find(p,p+count,uint8_t(0)) == p+count, "NUL in receipt path");
        return std::string(reinterpret_cast<const char *>(p),count);
    }
    bool finished() const { return position_ == bytes_.size(); }
};

inline Receipt read_receipt(const std::string & path, const Digest & expected_sha256) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(bool(file), "cannot open preflight receipt");
    const auto size = file.tellg();
    require(size >= 112 && size <= 1024*1024, "invalid preflight receipt size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);file.read(reinterpret_cast<char *>(bytes.data()),std::streamsize(bytes.size()));
    require(bool(file), "short preflight receipt read");
    require(sha256(bytes.data(),bytes.size()) == expected_sha256, "preflight receipt SHA256 mismatch");
    ReceiptCursor c(bytes);
    const std::array<uint8_t,8> magic{'M','3','1','S','E','E','D',0};
    const auto * actual_magic = c.take(8);
    require(std::equal(magic.begin(),magic.end(),actual_magic), "invalid preflight receipt magic");
    require(c.u32() == 1, "unsupported preflight receipt version");
    Receipt result;
    result.expected.layers=c.u32();result.expected.experts=c.u32();result.expected.dense_lead=c.u32();
    result.slots=c.u64();result.hot=c.u64();
    const auto shards=c.u32();
    require(shards && shards <= 1024, "invalid preflight shard count");
    require(result.expected.layers && result.expected.layers <= 1024 && result.expected.experts &&
            result.expected.experts <= 4096 && result.expected.dense_lead < result.expected.layers,
            "invalid receipt geometry");
    result.expected.identity=c.digest();result.expected.file_sha256=c.digest();result.rank_path=c.string();
    for (uint32_t i=0;i<shards;++i) {
        ShardReceipt shard;
        shard.path=c.string();shard.bytes=c.u64();shard.mtime_ns=c.u64();shard.header_bytes=c.u64();shard.header_sha256=c.digest();
        require(shard.bytes && shard.header_bytes && shard.header_bytes <= 32u*1024u*1024u &&
                shard.header_bytes <= shard.bytes, "invalid preflight header size");
        result.shards.push_back(std::move(shard));
    }
    require(c.finished(), "trailing preflight receipt bytes");
    return result;
}

inline uint64_t mtime_ns(const struct stat & s) {
#if defined(__linux__)
    return uint64_t(s.st_mtim.tv_sec)*1000000000ull+uint64_t(s.st_mtim.tv_nsec);
#elif defined(__APPLE__)
    return uint64_t(s.st_mtimespec.tv_sec)*1000000000ull+uint64_t(s.st_mtimespec.tv_nsec);
#else
    (void)s;
    throw std::runtime_error("M31 seed preflight: nanosecond file stat is unsupported on this platform");
#endif
}
template<class Model>
inline void verify_current_model(const Receipt & receipt,const Model & model) {
    require(receipt.shards.size()==model.size(), "preflight/current model shard count differs");
    for (size_t i=0;i<receipt.shards.size();++i) {
        const auto & expected=receipt.shards[i];
        const auto & shard=model.shard(i);
        require(std::filesystem::canonical(shard.path()).string()==expected.path,
                "preflight/current model shard path differs");
        struct stat before{},after{};
        require(::stat(expected.path.c_str(),&before)==0, "cannot stat current model shard");
        require(uint64_t(before.st_size)==expected.bytes && mtime_ns(before)==expected.mtime_ns,
                "current model size/mtime differs from preflight");
        require(shard.data_start()==expected.header_bytes, "current model header boundary differs from preflight");
        std::vector<uint8_t> header(size_t(expected.header_bytes));
        std::ifstream input(expected.path,std::ios::binary);
        require(bool(input), "cannot open current model header");
        input.read(reinterpret_cast<char *>(header.data()),std::streamsize(header.size()));
        require(bool(input), "short current model header read");
        require(sha256(header.data(),header.size())==expected.header_sha256,
                "current model header SHA256 differs from preflight");
        require(::stat(expected.path.c_str(),&after)==0 && before.st_size==after.st_size &&
                mtime_ns(before)==mtime_ns(after), "model shard changed during preflight validation");
    }
}
} // namespace strata::m31::cache_seed
