// Exact whole-prompt cache. Only immutable prompt inputs and completed KV prefixes
// are cached; this does not approximate or accelerate an unseen prompt.
#pragma once
#include "m31_engine.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace strata::m31::prompt_cache {
using Digest = std::array<uint8_t, 32>;
inline std::string hex(const Digest & value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (auto v : value) { out += digits[v >> 4]; out += digits[v & 15]; }
    return out;
}

// SHA256 with optional system-libcrypto acceleration. No mandatory dependency or
// build flag: portable fallback computes exactly the same digest on every host.
class Sha256 {
    std::array<uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t,64> tail_{};
    uint64_t size_ = 0;
    size_t tail_size_ = 0;
    void * fast_ = nullptr;
    struct Crypto {
        void * handle = nullptr;
        void * (*create)() = nullptr;
        void (*destroy)(void *) = nullptr;
        const void * (*sha256)() = nullptr;
        int (*init)(void *, const void *, void *) = nullptr;
        int (*update)(void *, const void *, size_t) = nullptr;
        int (*final)(void *, unsigned char *, unsigned int *) = nullptr;
        Crypto() {
#if defined(__linux__)
            handle = dlopen("libcrypto.so.3", RTLD_LAZY | RTLD_LOCAL);
            if (!handle) return;
            create = reinterpret_cast<decltype(create)>(dlsym(handle,"EVP_MD_CTX_new"));
            destroy = reinterpret_cast<decltype(destroy)>(dlsym(handle,"EVP_MD_CTX_free"));
            sha256 = reinterpret_cast<decltype(sha256)>(dlsym(handle,"EVP_sha256"));
            init = reinterpret_cast<decltype(init)>(dlsym(handle,"EVP_DigestInit_ex"));
            update = reinterpret_cast<decltype(update)>(dlsym(handle,"EVP_DigestUpdate"));
            final = reinterpret_cast<decltype(final)>(dlsym(handle,"EVP_DigestFinal_ex"));
            if (!(create && destroy && sha256 && init && update && final)) {
                dlclose(handle); handle = nullptr;
            }
#endif
        }
        // Keep the library resident while process-local hash objects may exist.
    };
    static Crypto & crypto() { static Crypto instance; return instance; }
    static uint32_t rotate(uint32_t x,int n) { return (x>>n)|(x<<(32-n)); }
    void block(const uint8_t * input) {
        static constexpr uint32_t k[] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64]{};
        for(int i=0;i<16;++i) for(int j=0;j<4;++j) w[i]=(w[i]<<8)|input[4*i+j];
        for(int i=16;i<64;++i) {auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));}
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for(int i=0;i<64;++i){auto t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+k[i]+w[i];auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
public:
    explicit Sha256(bool portable = false) {
        const char * force = std::getenv("M31_PROMPT_CACHE_HASH_PORTABLE");
        auto & api = crypto();
        if (!portable && !(force && std::string(force)=="1") && api.handle) {
            fast_ = api.create();
            if (fast_ && api.init(fast_,api.sha256(),nullptr)!=1) {api.destroy(fast_);fast_=nullptr;}
        }
    }
    Sha256(const Sha256 &)=delete;
    Sha256 &operator=(const Sha256 &)=delete;
    ~Sha256(){if(fast_)crypto().destroy(fast_);}
    bool accelerated() const {return fast_!=nullptr;}
    void add(const void * data,size_t count) {
        if(count && !data)throw std::runtime_error("prompt hash null input");
        if(count>std::numeric_limits<uint64_t>::max()/8 || size_>std::numeric_limits<uint64_t>::max()/8-count)throw std::runtime_error("prompt hash input overflow");
        size_+=count;
        if (!count) return;
        if(fast_){if(crypto().update(fast_,data,count)!=1)throw std::runtime_error("prompt hash update failed");return;}
        auto * p=static_cast<const uint8_t *>(data);
        if(tail_size_){auto n=std::min(count,64-tail_size_);std::memcpy(tail_.data()+tail_size_,p,n);tail_size_+=n;p+=n;count-=n;if(tail_size_==64){block(tail_.data());tail_size_=0;}}
        while(count>=64){block(p);p+=64;count-=64;}
        if(count){std::memcpy(tail_.data(),p,count);tail_size_=count;}
    }
    void number(uint64_t value){uint8_t bytes[8];for(int i=0;i<8;++i)bytes[i]=uint8_t(value>>(8*i));add(bytes,8);}
    void field(const void * data,size_t count){number(count);add(data,count);}
    Digest finish(){
        Digest out{};
        if(fast_){unsigned int n=0;if(crypto().final(fast_,out.data(),&n)!=1||n!=32)throw std::runtime_error("prompt hash final failed");return out;}
        const uint64_t bits=size_*8;
        tail_[tail_size_++]=0x80;
        if(tail_size_>56){std::fill(tail_.begin()+tail_size_,tail_.end(),0);block(tail_.data());tail_size_=0;}
        std::fill(tail_.begin()+tail_size_,tail_.begin()+56,0);
        for(int i=0;i<8;++i)tail_[56+i]=uint8_t(bits>>(56-8*i));
        block(tail_.data());
        for(size_t i=0;i<state_.size();++i)for(size_t j=0;j<4;++j)out[4*i+j]=uint8_t(state_[i]>>(24-8*j));
        return out;
    }
};

inline Digest input_key(const std::string & engine_identity,const std::vector<int32_t> & tokens,
                         const std::vector<int32_t> & visual_rows,const std::vector<float> & embeddings,
                         const std::vector<int32_t> & positions) {
    Sha256 hash;
    const std::string domain="m31-exact-whole-prompt-v1";
    hash.field(domain.data(),domain.size());hash.field(engine_identity.data(),engine_identity.size());
    hash.field(tokens.data(),tokens.size()*sizeof(int32_t));hash.field(visual_rows.data(),visual_rows.size()*sizeof(int32_t));
    hash.field(embeddings.data(),embeddings.size()*sizeof(float));hash.field(positions.data(),positions.size()*sizeof(int32_t));
    return hash.finish();
}
using Recipe = std::vector<std::pair<int64_t,int>>;
inline Digest entry_key(const Digest & input,const Recipe & recipe) {
    Sha256 hash;hash.field(input.data(),input.size());hash.number(recipe.size());
    for(auto [position,rows]:recipe){hash.number(uint64_t(position));hash.number(uint64_t(rows));}
    return hash.finish();
}
struct Entry {
    Digest input{}, key{};
    Engine::PrefixCheckpointPtr checkpoint;
    std::vector<float> logits;
    Recipe recipe;
    static size_t estimated_bytes(size_t prefix_bytes,size_t logit_count,size_t recipe_count) {return prefix_bytes+logit_count*sizeof(float)+recipe_count*sizeof(Recipe::value_type)+sizeof(Entry)+256;}
    size_t bytes() const {return estimated_bytes(checkpoint?checkpoint->bytes():0,logits.size(),recipe.size());}
};
class Cache {
    size_t budget_=0, used_=0;
    std::list<Entry> lru_;
public:
    explicit Cache(size_t budget=0):budget_(budget){if(budget>1024ull*1024*1024)throw std::runtime_error("prompt cache maximum is 1024MiB");}
    size_t budget()const{return budget_;}
    size_t bytes()const{return used_;}
    size_t entries()const{return lru_.size();}
    const Entry * find(const Digest & key){for(auto it=lru_.begin();it!=lru_.end();++it)if(it->input==key){lru_.splice(lru_.begin(),lru_,it);return &lru_.front();}return nullptr;}
    void erase(const Digest & key){for(auto it=lru_.begin();it!=lru_.end();)if(it->input==key){used_-=it->bytes();it=lru_.erase(it);}else ++it;}
    bool reserve(size_t bytes){if(!budget_||bytes>budget_)return false;while(used_>budget_-bytes){used_-=lru_.back().bytes();lru_.pop_back();}return true;}
    bool put(Entry entry){
        if(!entry.checkpoint||entry.logits.empty())return false;
        if (entry.recipe.size()!=1 || entry.recipe[0].first!=0) return false;
        int64_t end=0;
        for(auto [position,rows]:entry.recipe){if(position!=end||rows<=0)return false;end+=rows;}
        if(end!=entry.checkpoint->position())return false;
        entry.key=entry_key(entry.input,entry.recipe);
        const size_t count=entry.bytes();
        if(!budget_||count>budget_)return false;
        erase(entry.input);
        if(!reserve(count))return false;
        used_+=count;lru_.push_front(std::move(entry));return true;
    }
};
inline std::string engine_environment_identity() {
    std::vector<std::string> values;
#if defined(__linux__)
    for (char ** item = ::environ; item && *item; ++item) {
        const std::string value(*item);
        if (value.rfind("M31_",0)==0 || value.rfind("GGML_",0)==0) values.push_back(value);
    }
#endif
    std::sort(values.begin(),values.end());
    std::string identity;
    for (const auto & value : values) {identity += std::to_string(value.size()) + ":" + value;}
    return identity;
}
inline size_t environment_budget(){
    const char * value=std::getenv("M31_PROMPT_CACHE_MIB");if(!value)return 0;
    const std::string text(value);if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid M31_PROMPT_CACHE_MIB");
    const auto mib=std::stoull(text);if(mib>1024)throw std::runtime_error("M31_PROMPT_CACHE_MIB exceeds 1024");return size_t(mib)*1024*1024;
}
} // namespace strata::m31::prompt_cache
