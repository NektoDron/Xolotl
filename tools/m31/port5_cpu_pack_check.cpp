#include "cpu_pack_spans.hpp"
#include <cstdio>
#include <vector>
using namespace strata::m31;
struct Span { void * destination; const void * source; size_t bytes; };
void ensure(bool ok) { if (!ok) throw std::runtime_error("pack spans check failed"); }
int main() {
    try {
        const size_t mib=1024*1024;
        std::vector<uint8_t> source(25*mib), output(25*mib+128), expected(output.size());
        for(size_t i=0;i<source.size();++i) source[i]=uint8_t((i*37+i/1777)%251);
        for(unsigned threads : {1u,4u,8u}) {
            CpuPackSpans copier(threads);
            for(size_t shift : {size_t(0),size_t(17)}) {
                std::fill(output.begin(),output.end(),0xaa); expected=output;
                Span spans[]={{output.data()+shift,source.data(),7*mib+3},
                    {output.data()+8*mib+shift,source.data()+8*mib,8*mib-17},
                    {output.data()+16*mib+shift,source.data()+16*mib,9*mib-21}};
                for(auto & s:spans) std::memcpy(expected.data()+(static_cast<uint8_t*>(s.destination)-output.data()),s.source,s.bytes);
                copier.copy(spans,3);
                ensure(output==expected);
                // Reuse and mutate source immediately after the synchronous barrier.
                source[0]^=1;
                copier.copy(spans,3); ensure(output[shift]==source[0]);
            }
            copier.copy<Span>(nullptr,0);
            Span empty{nullptr,nullptr,0}; copier.copy(&empty,1);
            bool rejected=false;
            Span overlap{output.data(),output.data()+1,16};
            try{copier.copy(&overlap,1);}catch(const std::invalid_argument&){rejected=true;}
            ensure(rejected);
            std::printf("pack threads=%u allbytes/guards/reuse/empty/overlap PASS parallel_jobs=%zu\n",threads,copier.stats().parallel_jobs);
        }
    } catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
