// Optional Linux anonymous backing. No sysctl, hugetlb, mlock or GPU calls.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::m31 {
class PrefillThpMapping {
public:
    static constexpr size_t huge = 2ULL * 1024 * 1024;
    struct Policy {
        uint64_t free_huge_bytes = 0;
        bool buddy_known = false, force = false;
        std::string enabled, defrag;
    };
    struct HugePages { uint64_t bytes = 0; bool exact = false; };
    struct Stats {
        size_t requested_bytes = 0, mapped_bytes = 0, reservation_bytes = 0;
        uint64_t eligible_huge_bytes = 0, free_huge_bytes = 0, required_huge_free = 0;
        bool mapped = false, advised = false, forced = false;
        std::string enabled, defrag, note;
    };
    static bool requested() {
        const auto * flag = std::getenv("M31_ARENA_THP");
        return flag && std::strcmp(flag,"1") == 0;
    }
    static Policy system_policy() {
        Policy policy;
        const auto * force = std::getenv("M31_ARENA_THP_FORCE");
        policy.force = force && std::strcmp(force,"1") == 0;
#if defined(__linux__)
        auto mode=[](const char * path) {
            std::ifstream input(path);std::string line;
            std::getline(input,line);
            const auto a=line.find('['),b=line.find(']');
            return a!=std::string::npos && b>a?line.substr(a+1,b-a-1):std::string{};
        };
        policy.enabled=mode("/sys/kernel/mm/transparent_hugepage/enabled");
        policy.defrag=mode("/sys/kernel/mm/transparent_hugepage/defrag");
        const long page=sysconf(_SC_PAGESIZE);
        std::ifstream input("/proc/buddyinfo");
        std::string line;
        while(page>0 && std::getline(input,line)) {
            std::istringstream stream(line);
            std::string node,id,zone,name;
            if(!(stream>>node>>id>>zone>>name) || node!="Node" || zone!="zone")continue;
            // DMA/DMA32 reserve blocks are not counted as freely usable Normal
            // anonymous memory. This is deliberately more conservative than the
            // upstream all-zone heuristic on this 624GiB machine.
            if(name!="Normal" && name!="Movable")continue;
            uint64_t count=0;
            unsigned order=0;
            while(stream>>count) {
                if(order>=63 || uint64_t(page)>UINT64_MAX>>order)break;
                const uint64_t block=uint64_t(page)<<order;
                if(block>=huge) {
                    if(count && block>UINT64_MAX/count)throw std::runtime_error("M31 THP buddy count overflow");
                    const uint64_t bytes=block*count;
                    if(policy.free_huge_bytes>UINT64_MAX-bytes)throw std::runtime_error("M31 THP buddy total overflow");
                    policy.free_huge_bytes+=bytes;
                }
                ++order;
            }
            policy.buddy_known=policy.buddy_known || order>0;
        }
#endif
        return policy;
    }

    static std::unique_ptr<PrefillThpMapping> create(size_t bytes) {
        return create(bytes,system_policy());
    }
    static std::unique_ptr<PrefillThpMapping> create(size_t bytes,const Policy & policy) {
        auto result=std::unique_ptr<PrefillThpMapping>(new PrefillThpMapping);
        auto & s=result->stat_;
        s.requested_bytes=bytes;s.free_huge_bytes=policy.free_huge_bytes;
        s.forced=policy.force;s.enabled=policy.enabled;s.defrag=policy.defrag;
        if(!bytes || bytes>std::numeric_limits<size_t>::max()-2*huge-65536)
            throw std::runtime_error("M31 THP: invalid mapping byte count");
#if defined(__linux__)
        const long system_page=sysconf(_SC_PAGESIZE);
        if(system_page<=0 || huge%size_t(system_page)) {s.note="unsupported system page geometry; existing CPU allocator fallback";return result;}
        const size_t page=size_t(system_page);
        s.mapped_bytes=(bytes+page-1)/page*page;
        s.eligible_huge_bytes=s.mapped_bytes/huge*huge;
        // In madvise/defrag=madvise mode a fragmented 150GiB request may spend
        // minutes compacting during faults. Require free high-order blocks plus
        // a reserve BEFORE any mmap/prefault. No promise survives a concurrent
        // allocator race; actual AnonHugePages must still be measured.
        if(s.eligible_huge_bytes) {
            const uint64_t reserve=std::max<uint64_t>(64ULL<<20,s.eligible_huge_bytes/4);
            if(s.eligible_huge_bytes>UINT64_MAX-reserve)throw std::runtime_error("M31 THP guard overflow");
            s.required_huge_free=s.eligible_huge_bytes+reserve;
            if(!policy.force && (!policy.buddy_known || policy.free_huge_bytes<s.required_huge_free)) {
                s.note="fragmentation guard: free Normal/Movable high-order blocks="+std::to_string(policy.free_huge_bytes)+
                    "B, need="+std::to_string(s.required_huge_free)+"B; existing CPU allocator fallback (M31_ARENA_THP_FORCE=1 overrides with compaction risk)";
                return result;
            }
        }
        if(policy.enabled=="never") {s.note="THP mode never; existing CPU allocator fallback";return result;}
        if(policy.enabled.empty() && !policy.force) {s.note="unknown THP mode; existing CPU allocator fallback";return result;}
        // PROT_NONE guards prevent merging with adjacent anonymous VMAs, so the
        // per-mapping AnonHugePages measurement is attributable to this payload.
        // Guard reservation is virtual only; bytes are not faulted or mlocked.
        s.reservation_bytes=s.mapped_bytes+2*huge;
        void * raw=mmap(nullptr,s.reservation_bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(raw==MAP_FAILED) {s.note="anonymous mmap failed: "+std::string(std::strerror(errno))+"; existing CPU allocator fallback";return result;}
        result->reservation_=raw;
        const uintptr_t start=reinterpret_cast<uintptr_t>(raw);
        const uintptr_t aligned=(start+page+huge-1)/huge*huge;
        result->data_=reinterpret_cast<void *>(aligned);
        if(mprotect(result->data_,s.mapped_bytes,PROT_READ|PROT_WRITE)!=0) {
            s.note="anonymous mprotect failed: "+std::string(std::strerror(errno))+"; existing CPU allocator fallback";
            result->unmap();return result;
        }
        if(policy.enabled=="always") {
            // As in src/core/pinned.cu, do not add MADV_HUGEPAGE's direct reclaim
            // behavior when THP already applies automatically to every mapping.
            s.note="aligned anonymous mapping, THP always; redundant MADV_HUGEPAGE omitted";
        } else if(madvise(result->data_,s.mapped_bytes,MADV_HUGEPAGE)==0) {
            s.advised=true;
            s.note="aligned anonymous mapping, MADV_HUGEPAGE before pack; actual backing is not guaranteed";
        } else {
            s.note="MADV_HUGEPAGE failed: "+std::string(std::strerror(errno))+"; existing CPU allocator fallback";
            result->unmap();return result;
        }
        s.mapped=true;
        if(policy.force)s.note+="; forced guard override permits direct reclaim/compaction during pack";
#else
        s.note="THP backing is Linux-only; existing CPU allocator fallback";
#endif
        return result;
    }

    ~PrefillThpMapping(){unmap();}
    PrefillThpMapping(const PrefillThpMapping &)=delete;
    PrefillThpMapping & operator=(const PrefillThpMapping &)=delete;
    void * data()const{return data_;}
    const Stats & stats()const{return stat_;}

    // No page faults are introduced here. A separate THP hint is not a claim:
    // return the kernel's measured AnonHugePages for wholly contained payload VMAs.
    HugePages anon_huge_pages()const {
        HugePages out;
#if defined(__linux__)
        if(!data_)return out;
        const uintptr_t first=reinterpret_cast<uintptr_t>(data_),last=first+stat_.mapped_bytes;
        std::ifstream input("/proc/self/smaps");
        std::string line;
        bool inside=false,partial=false,found=false;
        while(std::getline(input,line)) {
            uintptr_t a=0,b=0;
            if(std::sscanf(line.c_str(),"%lx-%lx",&a,&b)==2) {
                const bool overlaps=a<last && b>first;
                inside=overlaps && a>=first && b<=last;
                if(overlaps && !inside)partial=true;
                if(inside)found=true;
            } else if(inside && line.rfind("AnonHugePages:",0)==0) {
                uint64_t kb=0;
                std::istringstream stream(line.substr(14));
                if(stream>>kb)out.bytes+=kb*1024;
            }
        }
        out.exact=found && !partial;
#endif
        return out;
    }
private:
    PrefillThpMapping()=default;
    void unmap()noexcept {
#if defined(__linux__)
        if(reservation_)munmap(reservation_,stat_.reservation_bytes);
#endif
        reservation_=nullptr;data_=nullptr;stat_.mapped=false;
    }
    void * reservation_=nullptr,*data_=nullptr;
    Stats stat_;
};
} // namespace strata::m31
