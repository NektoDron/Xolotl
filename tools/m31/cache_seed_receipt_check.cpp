// CPU-only receipt/model-header check; no CUDA or payload scan.
#include "cache_seed_receipt.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <iostream>

int main(int argc,char ** argv) {
    try {
        using namespace strata::m31::cache_seed;
        require(argc==4,"usage: receipt_check SEAL SHA256 MODEL.gguf");
        const auto receipt=read_receipt(argv[1],hex_digest(argv[2]));
        const auto model=strata::GgufModel::open(argv[3]);
        verify_current_model(receipt,model);
        const auto ranking=read(receipt.rank_path,receipt.expected);
        std::cout<<"{\"verified\":true,\"count\":"<<ranking.ids.size()<<",\"slots\":"<<receipt.slots<<",\"hot\":"<<receipt.hot<<"}\n";
        return 0;
    } catch (const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
