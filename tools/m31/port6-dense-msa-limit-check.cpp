#include "msa_chunk_limit.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
using strata::m31::msa_chunk_limit;
int main(){try{
    int checks=0;auto check=[&](bool ok){if(!ok)throw std::runtime_error("MSA chunk boundary mismatch");++checks;};
    struct Case{int64_t pos;int requested,heads,tile,expected;};
    for(auto c:std::array<Case,8>{{{0,32768,4,0,23045},{57344,8192,4,0,8191},{65536,8192,4,0,7358},
        {131072,8192,4,0,3971},{1000000,8192,4,0,536},{65536,8192,4,512,8192},
        {0,32768,4,512,32768},{INT32_MAX,1,4,0,0}}})
        check(msa_chunk_limit(c.pos,c.requested,c.heads,128,c.tile)==c.expected);
    check(msa_chunk_limit(65536,8192,4,128,0,512)==8192);
    check(msa_chunk_limit(0,0,4,128)==0);
    check(msa_chunk_limit(0,32768,INT64_MAX,128)==0);
    // Compare against an independent linear scan with wide integer products.
    for(int i=0;i<200;++i){const int64_t pos=int64_t(i)*71113;const int requested=1+(i*977)%32768;
        const int heads=1+i%8,tile=i%3?0:512;int expected=0;
        for(int n=1;n<=requested;++n){const uint64_t keys=uint64_t((pos+n+255)/256*256);
            const int rows=tile?std::min(n,tile):n;
            if(__int128(keys)*heads*rows>=INT32_MAX)break;expected=n;
        }
        check(msa_chunk_limit(pos,requested,heads,128,tile)==expected);
    }
    for(auto args:std::array<std::array<int64_t,4>,4>{{{{-1,10,4,128}},{{0,-1,4,128}},{{0,10,0,128}},{{0,10,4,0}}}}){
        bool threw=false;try{msa_chunk_limit(args[0],int(args[1]),args[2],args[3]);}catch(const std::exception&){threw=true;}check(threw);
    }
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"scope\":\"pure MSA int32 chunk boundaries, no model/GPU\"}\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
