#include "cpu_pack_spans.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace strata::m31;
struct Span { void * destination; const void * source; size_t bytes; };
int main() {
    try {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        constexpr size_t mib=1024*1024, source_bytes=1024*mib, slot=25*mib+4096;
        constexpr unsigned passes=128;
        const std::array<size_t,3> offsets{0,8*mib+64,16*mib+64};
        const std::array<size_t,3> lengths{8*mib+13,8*mib-17,9*mib-91};
        std::vector<uint8_t> source(source_bytes), destination(2*slot,0);
        std::memset(source.data(),0x7b,source.size());
        const size_t sources=source.size()/slot;
        size_t bytes=0; for(auto length:lengths) bytes+=length;
        std::printf("CPU pageable copy, source=%zu bytes slots=%zu destination=%zu bytes stride=%zu role_bytes=%zu passes=%u; no GPU/pinned claim\n",
            source.size(),sources,destination.size(),slot,bytes,passes);
        for(unsigned threads : {1u,4u,8u}) {
            CpuPackSpans copier(threads);
            auto copy = [&](size_t i) {
                //17 is coprime to40 source segments, so each rotation exceeds LLC.
                const size_t input=(i*17%sources)*slot, output=(i%2)*slot;
                std::array<Span,3> spans;
                for(size_t role=0;role<spans.size();++role)
                    spans[role]={destination.data()+output+offsets[role],source.data()+input+offsets[role],lengths[role]};
                copier.copy(spans.data(),spans.size());
            };
            for(size_t i=0;i<4;++i) copy(i);
            const auto start=std::chrono::steady_clock::now();
            for(size_t i=0;i<passes;++i) copy(i+4);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            for(size_t output=0;output<2;++output) for(size_t role=0;role<offsets.size();++role) {
                const auto * data=destination.data()+output*slot+offsets[role];
                if(data[0]!=0x7b || data[lengths[role]-1]!=0x7b)
                    throw std::runtime_error("bandwidth probe output differs");
            }
            auto stats=copier.stats();
            std::printf("threads=%u bytes=%zu seconds=%.6f GBps=%.3f serial_jobs=%zu parallel_jobs=%zu worker_ranges=%zu\n",
                threads,bytes*passes,seconds,double(bytes)*passes/seconds/1e9,stats.serial_jobs,stats.parallel_jobs,stats.worker_ranges);
        }
    } catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
