#include "m31_arena.hpp"
#include "json.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

using namespace strata::m31;
using json=nlohmann::json;
static void check(bool condition,const char * message){if(!condition)throw std::runtime_error(message);}
static json mapping_info(const PrefillThpMapping & mapping) {
    const auto & s=mapping.stats();const auto actual=mapping.anon_huge_pages();
    return {{"mapped",s.mapped},{"advised",s.advised},{"forced",s.forced},
        {"requested_bytes",s.requested_bytes},{"mapped_bytes",s.mapped_bytes},
        {"reservation_bytes",s.reservation_bytes},{"eligible_huge_bytes",s.eligible_huge_bytes},
        {"free_huge_bytes",s.free_huge_bytes},{"required_huge_free",s.required_huge_free},
        {"enabled",s.enabled},{"defrag",s.defrag},{"note",s.note},
        {"anon_huge_bytes",actual.bytes},{"anon_huge_exact",actual.exact}};
}
#if defined(__linux__)
static void check_unmapped(void * address) {
    unsigned char page=0;errno=0;
    const int result=mincore(address,size_t(sysconf(_SC_PAGESIZE)),&page);
    const int error=errno;
    check(result==-1 && error==ENOMEM,"RAII mapping did not unmap payload after ggml wrapper destruction");
}
#endif
int main(int argc,char ** argv) {
    try {
        std::string model,out;
        for(int i=1;i<argc;++i){check(i+1<argc,"missing option argument");std::string key=argv[i],value=argv[++i];
            if(key=="--model")model=value;else if(key=="--out")out=value;else throw std::runtime_error("unknown option "+key);}
        check(!model.empty() && !out.empty() && !std::filesystem::exists(out),"need model and NEW output file");
        setenv("M31_ARENA_THP_FORCE","0",1);
        const auto file=strata::GgufModel::open(model);const auto geometry=read_geometry(file);
        // This probe is intentionally capped: never pack the production arena.
        check(geometry.expert_bytes<16ULL*1024*1024,"small test refuses production-sized model");
        std::vector<uint8_t> golden;
        size_t logical_bytes=0;
        setenv("M31_ARENA_THP","0",1);
        {
            ExpertArena baseline;baseline.init(file,geometry,.1);
            logical_bytes=baseline.bytes();
            check(!baseline.thp_mapping(),"default allocator unexpectedly used THP owner");
            check(ggml_backend_buffer_get_size(baseline.buffer())==logical_bytes,"default numBytes changed");
            const auto * data=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(baseline.buffer()));
            golden.assign(data,data+logical_bytes);
        }
        json arenas=json::array();
        for(int repeat=0;repeat<8;++repeat) {
            void * old_payload=nullptr;
            setenv("M31_ARENA_THP","1",1);
            {
                ExpertArena arena;
                // Exact logical budget must still pass: THP page/guard metadata
                // cannot silently increase ggml's visible expert buffer size.
                arena.init(file,geometry,double(logical_bytes)/1073741824.0);
                check(arena.bytes()==logical_bytes && ggml_backend_buffer_get_size(arena.buffer())==logical_bytes,"THP numBytes changed");
                const auto * data=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(arena.buffer()));
                check(!std::memcmp(golden.data(),data,logical_bytes),"THP packing changed expert bytes/tail/padding");
                for(int l=0;l<geometry.layers;++l)if(geometry.routed(l)) {
                    const auto & layer=arena.layers()[size_t(l)];
                    int r=0;
                    for(const char * role:{"gate","up","down"}) {
                        const auto name=layer_name(l,std::string("ffn_")+role+"_exps.weight");
                        const auto * slice=arena.find(name);
                        check(slice && slice->expert_stride==layer.stride,"THP slice stride changed");
                        check(slice->data==data+layer.offset+layer.role_offsets[size_t(r++)],"THP slice base changed");
                    }
                }
                check(arena.thp_mapping(),"THP attempt lacks report");
                if(repeat==0)arenas.push_back(mapping_info(*arena.thp_mapping()));
                old_payload=arena.thp_mapping()->data();
                bool repeated=false;
                try{arena.init(file,geometry,.1);}catch(const std::exception &){repeated=true;}
                check(repeated,"repeated arena initialization accepted");
            }
#if defined(__linux__)
            if(old_payload)check_unmapped(old_payload);
#endif
        }
        int bad_budgets=0;
        for(double budget:{0.,-1.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),.000000001}) {
            try{ExpertArena bad;bad.init(file,geometry,budget);}catch(const std::exception &){++bad_budgets;}
        }
        check(bad_budgets==5,"invalid arena budget accepted");
        int bad_sizes=0;
        for(size_t bytes:{size_t(0),std::numeric_limits<size_t>::max()}) {
            try{(void)PrefillThpMapping::create(bytes);}catch(const std::exception &){++bad_sizes;}
        }
        check(bad_sizes==2,"invalid mmap byte count accepted");
        auto blocked=PrefillThpMapping::system_policy();
        blocked.free_huge_bytes=0;blocked.buddy_known=true;blocked.force=false;blocked.enabled="madvise";
        const auto large=PrefillThpMapping::create(170ULL<<30,blocked);
        check(!large->data() && !large->stats().mapped && large->stats().reservation_bytes==0,"fragmentation guard reached large mmap");
        // Real small map, with the real conservative system policy. The result
        // may legitimately fall back or get zero huge pages; report the kernel.
        json small;
        void * old_payload=nullptr;
        {
            const auto mapping=PrefillThpMapping::create(16ULL<<20);
            if(mapping->data()) {
                old_payload=mapping->data();
                check(reinterpret_cast<uintptr_t>(old_payload)%PrefillThpMapping::huge==0,"payload not 2MiB aligned");
                auto wrapper=ggml_backend_cpu_buffer_from_ptr(old_payload,16ULL<<20);
                check(wrapper && ggml_backend_buffer_get_size(wrapper)==(16ULL<<20),"small borrowed buffer numBytes mismatch");
                std::memset(old_payload,0x6a,16ULL<<20);
                small=mapping_info(*mapping);
                ggml_backend_buffer_free(wrapper);
                // The ggml wrapper must not free memory owned by mmap RAII.
                auto * data=static_cast<volatile uint8_t *>(old_payload);
                check(data[0]==0x6a && data[(16ULL<<20)-1]==0x6a,"ggml wrapper incorrectly owned backing memory");
                data[0]=0x13;
                check(mapping->anon_huge_pages().bytes<=mapping->stats().mapped_bytes,"AnonHugePages attribution overflow");
            } else small=mapping_info(*mapping);
        }
#if defined(__linux__)
        if(old_payload)check_unmapped(old_payload);
#endif
        setenv("M31_ARENA_THP","0",1);
        json result={{"format","m31-thp-small-proof-v1"},{"pass",true},
            {"logical_arena_bytes",logical_bytes},{"arena_byte_parity",true},{"arena_cycles",8},
            {"slice_offsets_and_strides",true},{"borrowed_buffer_lifetime",true},
            {"raii_unmap",true},{"invalid_budgets",bad_budgets},{"invalid_mapping_sizes",bad_sizes},
            {"large_guard_prevents_mmap",true},{"large_guard",mapping_info(*large)},
            {"arena_thp",arenas},{"small_mapping",small},
            {"limits","No production arena allocation; largest touched mapping 16MiB. THP hint is not a promise; AnonHugePages is measured. No sysctl/mlock/hugetlb/GPU actions."}};
        std::ofstream stream(out);stream<<result.dump(2)<<'\n';check(bool(stream),"cannot write THP proof");
        return 0;
    } catch(const std::exception & e){std::cerr<<"THP proof: "<<e.what()<<'\n';return 1;}
}
