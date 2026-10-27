#include "prefill_route_history.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main() {
    using strata::m31::PrefillRouteHistory;
    auto check=[](bool value,const char * message){if(!value)throw std::runtime_error(message);};
    try {
        PrefillRouteHistory history(60,128);
        check(!history.has(4),"unknown layer became known");
        for(size_t expert=0;expert<128;++expert)check(history.predicts(4,expert),"unknown must prefetch normally");
        std::vector<uint8_t> first(128),second(128),decode(128);
        for(size_t expert: {size_t(1),size_t(5),size_t(76)})first[expert]=1;
        for(size_t expert: {size_t(2),size_t(5),size_t(100)})second[expert]=1;
        decode[7]=1;
        history.remember(4,515,first);
        history.remember(5,3863,second);
        check(history.has(4)&&history.has(5)&&!history.has(6),"layer state leaked");
        for(size_t expert=0;expert<128;++expert)check(history.predicts(4,expert)==bool(first[expert]),"wrong mask");
        history.remember(4,1,decode);
        history.remember(4,127,decode);
        for(size_t expert=0;expert<128;++expert)check(history.predicts(4,expert)==bool(first[expert]),"small batch clobbered history");
        history.remember(4,128,second);
        for(size_t expert=0;expert<128;++expert)check(history.predicts(4,expert)==bool(second[expert]),"large batch did not refresh");
        // History retains the ranking and limit imposed by the uploader; it
        // returns only an eligibility bit and never rewrites an actual ID.
        std::vector<int> ranked(128);std::iota(ranked.begin(),ranked.end(),0);
        std::reverse(ranked.begin(),ranked.end());
        std::vector<int> predicted;
        for(int expert:ranked)if(history.predicts(4,size_t(expert))&&predicted.size()<2)predicted.push_back(expert);
        check(predicted==std::vector<int>({100,5}),"rank/limit not preserved");
        int negatives=0;
        auto reject=[&](auto call){try{call();}catch(const std::exception &){++negatives;}};
        reject([&]{history.remember(60,515,first);});
        reject([&]{history.remember(4,515,std::vector<uint8_t>(127));});
        auto invalid=first;invalid[13]=2;reject([&]{history.remember(4,515,invalid);});
        reject([&]{(void)history.predicts(4,128);});
        reject([&]{PrefillRouteHistory bad(0,128);});
        check(negatives==5,"invalid history accepted");
        for(size_t expert=0;expert<128;++expert)check(history.predicts(4,expert)==bool(second[expert]),"failed update changed a valid mask");
        std::cout<<"{\"pass\":true,\"negative_cases\":5,\"decode_preserves_history\":true}\n";
        return 0;
    } catch(const std::exception & error){std::cerr<<error.what()<<'\n';return 1;}
}
