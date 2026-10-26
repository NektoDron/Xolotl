#include "prefill_route_worklist.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

using namespace strata::m31;
static void check(bool ok,const char * message) { if(!ok)throw std::runtime_error(message); }

static std::vector<int32_t> mixed_ids(int n,int used,int slots,uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> ids(size_t(n)*used);
    for(int t=0;t<n;++t) {
        for(int r=0;r<used;++r) {
            int id;
            do {
                id=int(rng()%uint64_t(slots-3));
            } while(std::find(ids.begin()+size_t(t)*used,ids.begin()+size_t(t)*used+r,id)!=ids.begin()+size_t(t)*used+r);
            ids[size_t(t)*used+r]=id;
        }
        // Rare experts straddle tiled dispatch's exact threshold in large cases.
        if(t<7)ids[size_t(t)*used]=slots-3;
        else if(t<15)ids[size_t(t)*used]=slots-2;
        else if(t<24)ids[size_t(t)*used]=slots-1;
    }
    return ids;
}

int main() {
    try {
        size_t cases=0, values=0;
        for(int n:{1,7,128,2048})for(int used:{1,4})for(uint64_t seed:{17,107,1009}) {
            const auto ids=mixed_ids(n,used,23,seed);
            const auto work=prefill_route_worklist(ids,23,used);
            std::vector<int> seen(ids.size());
            std::vector<int32_t> hist(23);
            for(auto id:ids)++hist[size_t(id)];
            check(work.histogram==hist,"histogram changed");
            for(size_t j=0;j<work.route.size();++j) {
                const int route=work.route[j];
                check(route>=0 && size_t(route)<ids.size() && !seen[size_t(route)]++,"not a bijective route permutation");
                if(j) {
                    const int prev=work.route[j-1];
                    check(ids[size_t(prev)]<=ids[size_t(route)],"worklist not expert-sorted");
                    check(ids[size_t(prev)]!=ids[size_t(route)] || prev<route,"sort not stable");
                }
                check(work.tiled[size_t(route)]==(hist[size_t(ids[size_t(route)])]>=8),"dispatch uses subset histogram");
            }
            for(bool broadcast:{false,true}) {
                constexpr int columns=13,rows=7;
                std::vector<float> activation(size_t(columns)*n*(broadcast?1:used));
                std::vector<float> weights(23*rows*columns);
                for(size_t i=0;i<activation.size();++i)activation[i]=std::sin(float(i)*0.17f);
                for(size_t i=0;i<weights.size();++i)weights[i]=std::cos(float(i)*0.41f);
                auto compute=[&](const std::vector<int32_t> & order) {
                    std::vector<float> output(ids.size()*rows);
                    for(auto route:order) {
                        const size_t input_row=broadcast?size_t(route)/size_t(used):size_t(route);
                        for(int row=0;row<rows;++row) {
                            float acc=0;
                            for(int col=0;col<columns;++col)
                                acc=std::fma(weights[(size_t(ids[size_t(route)])*rows+row)*columns+col],
                                             activation[input_row*columns+col],acc);
                            output[size_t(route)*rows+row]=acc;
                        }
                    }
                    return output;
                };
                std::vector<int32_t> original(ids.size());std::iota(original.begin(),original.end(),0);
                const auto a=compute(original),b=compute(work.route);
                check(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"work scheduling changed original output/input indexing");
                values+=a.size();++cases;
            }
        }
        const std::vector<int32_t> ids{0,1,0,1,0,1,0,1};
        const auto disabled=prefill_route_worklist(ids,2,2,false,true);
        check(std::all_of(disabled.tiled.begin(),disabled.tiled.end(),[](int v){return v==0;}),"disabled tiled override");
        const auto forced=prefill_route_worklist(ids,2,2,true,true);
        check(std::all_of(forced.tiled.begin(),forced.tiled.end(),[](int v){return v==1;}),"forced tiled override");
        int negatives=0;
        for(const auto bad:std::vector<std::vector<int32_t>>{{-1,0},{0,2},{0,1,0}}) {
            try{(void)prefill_route_worklist(bad,2,2);}catch(const std::exception &){++negatives;}
        }
        check(negatives==3,"invalid IDs/shapes accepted");
        std::cout<<"{\"cpu_worklist_pass\":true,\"layout_cases\":"<<cases
                 <<",\"compared_f32_values\":"<<values<<",\"negative_cases\":"<<negatives<<"}\n";
        return 0;
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
