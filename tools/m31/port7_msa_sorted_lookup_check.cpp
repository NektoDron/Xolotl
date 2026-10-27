// Standalone proof SOURCE. Not wired to Engine/CMake and not executed yet.
#include "msa_sorted_lookup.hpp"
#include "ggml-cpu.h"
#ifdef M31_HAVE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace strata::m31;
#ifdef M31_HAVE_CUDA
// Reuse the hash-bound existing event-only TU; no new CUDA kernels or context layout.
double dense_ops_graph_us(ggml_backend_t,ggml_cgraph *,int);
#endif
namespace {
void need(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
void clear_capture(ggml_backend_t backend,bool cuda){
#ifdef M31_HAVE_CUDA
    if(cuda){ggml_backend_synchronize(backend);ggml_backend_cuda_graph_cache_clear(backend);}
#else
    (void)backend;(void)cuda;
#endif
}
struct Old {ggml_tensor *token,*row;};
Old original(ggml_context * c,ggml_tensor *idx,int64_t block,int64_t top,int64_t heads){
    auto a=ggml_reshape_3d(c,ggml_scale(c,ggml_cast(c,idx,GGML_TYPE_F32),float(block)),1,top,heads);
    auto tj=ggml_add(c,ggml_repeat_4d(c,a,block,top,heads,1),ggml_reshape_3d(c,ggml_arange(c,0,float(block),1),block,1,1));
    auto tr=ggml_add(c,ggml_scale(c,tj,float(heads)),ggml_reshape_3d(c,ggml_arange(c,0,float(heads),1),1,1,heads));
    const int64_t count=block*top*heads;
    return {ggml_cast(c,ggml_reshape_1d(c,tj,count),GGML_TYPE_I32),ggml_cast(c,ggml_reshape_1d(c,tr,count),GGML_TYPE_I32)};
}
}
int main(int argc,char ** argv){
    ggml_backend_t backend=nullptr;
    try{
        const bool cuda=argc==2&&std::string(argv[1])=="cuda";need(argc<=2&&(argc==1||cuda||std::string(argv[1])=="cpu"),"usage: check [cpu|cuda]");
        if(cuda){
            const auto session=std::getenv("M31_CARD_SESSION");need(session&&std::string(session)=="1","CUDA requires coordinator aictl/card runner");
#ifdef M31_HAVE_CUDA
            backend=ggml_backend_cuda_init(0);
#else
            throw std::runtime_error("built without CUDA");
#endif
        }else {backend=ggml_backend_cpu_init();if(backend)ggml_backend_cpu_set_n_threads(backend,2);}need(backend,"backend init failed");
        uint64_t cases=0,compared=0,head_distinct=0,strided=0;
        // Includes the maximum exact row4,194,303, tiny/padded contexts, and
        // physical lookup capacity larger than the current active block set.
        for(int64_t capacity:{256LL,4096LL,16384LL,32768LL,65536LL,1048576LL}){
            constexpr int64_t block=128,heads=4;MsaSortedLookupMaps maps(backend,capacity,block,heads,heads);
            const int64_t blocks=capacity/block;
            for(int active_variant=0;active_variant<2;++active_variant){
                const int64_t active=active_variant?std::max<int64_t>(1,blocks/2):blocks;
                for(int64_t top:{int64_t(1),std::min<int64_t>(16,active)})for(int pattern=0;pattern<4;++pattern){
                    auto c=ggml_init({256*1024,nullptr,true});need(c,"context init failed");
                    ggml_backend_buffer_t input_buffer=nullptr;ggml_gallocr_t allocator=nullptr;
                    try{
                        // nb1 includes unused tail columns: actual ggml_top_k is
                        // such a view into argsort's [activeBlocks,heads] output.
                        auto full=ggml_new_tensor_2d(c,GGML_TYPE_I32,active+7,heads);
                        input_buffer=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_get_default_buffer_type(backend));need(input_buffer,"input allocation failed");
                        auto idx=ggml_view_2d(c,full,top,heads,full->nb[1],0);auto before=original(c,idx,block,top,heads);auto after=maps.build(c,idx);
                        auto graph=ggml_new_graph_custom(c,128,false);for(auto t:{before.token,before.row,after.token,after.row}){ggml_set_output(t);ggml_build_forward_expand(graph,t);}
                        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));need(allocator&&ggml_gallocr_alloc_graph(allocator,graph),"graph allocation failed");
                        std::vector<int32_t> picks(size_t((active+7)*heads),-777),expected_token(size_t(block*top*heads)),expected_row(expected_token.size());
                        for(int64_t h=0;h<heads;++h)for(int64_t rank=0;rank<top;++rank){
                            const int64_t id=pattern==0?active-1-rank:pattern==1?(rank+7*h)%active:pattern==2?active-1:(rank*17+13*h+31)%active;
                            picks[size_t(h*(active+7)+rank)]=int32_t(id);
                            for(int64_t lane=0;lane<block;++lane){const auto o=size_t((h*top+rank)*block+lane);expected_token[o]=int32_t(id*block+lane);expected_row[o]=int32_t((id*block+lane)*heads+h);}
                        }
                        ggml_backend_tensor_set(full,picks.data(),0,picks.size()*sizeof(int32_t));need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
                        for(auto pair:{std::make_pair(before.token,&expected_token),std::make_pair(after.token,&expected_token),std::make_pair(before.row,&expected_row),std::make_pair(after.row,&expected_row)}){
                            std::vector<int32_t> actual(pair.second->size());ggml_backend_tensor_get(pair.first,actual.data(),0,actual.size()*sizeof(int32_t));need(actual==*pair.second,"lookup/original differs from closed form");compared+=actual.size();
                        }
                        if(active_variant==0&&pattern==0&&top==std::min<int64_t>(16,active)&&capacity>=4096&&capacity<=65536){
                            auto old_graph=ggml_new_graph_custom(c,128,false),new_graph=ggml_new_graph_custom(c,128,false);
                            for(auto t:{before.token,before.row})ggml_build_forward_expand(old_graph,t);
                            for(auto t:{after.token,after.row})ggml_build_forward_expand(new_graph,t);
                            std::cout<<"{\"kind\":\"geometry\",\"capacity\":"<<capacity<<",\"top\":"<<top<<",\"map_resident_bytes\":"<<maps.resident_bytes()<<",\"old_graph_nodes\":"<<ggml_graph_n_nodes(old_graph)<<",\"new_graph_nodes\":"<<ggml_graph_n_nodes(new_graph);
#ifdef M31_HAVE_CUDA
                            if(cuda){
                                for(int warm=0;warm<5;++warm){need(ggml_backend_graph_compute(backend,old_graph)==GGML_STATUS_SUCCESS,"old warm failed");need(ggml_backend_graph_compute(backend,new_graph)==GGML_STATUS_SUCCESS,"new warm failed");}
                                const double old_us=dense_ops_graph_us(backend,old_graph,200),new_us=dense_ops_graph_us(backend,new_graph,200);
                                std::cout<<",\"old_cuda_event_us\":"<<old_us<<",\"new_cuda_event_us\":"<<new_us<<",\"event_repeats\":200";
                            }
#endif
                            std::cout<<",\"scope\":\"Resident maps+idx, warmed original/new expansion only; graph nodes are not kernel launches or model TPS\"}\n";
                        }
                        ++cases;++strided;if(pattern==1||pattern==3)++head_distinct;
                    }catch(...){clear_capture(backend,cuda);if(allocator)ggml_gallocr_free(allocator);if(input_buffer)ggml_backend_buffer_free(input_buffer);ggml_free(c);throw;}
                    clear_capture(backend,cuda);
                    ggml_gallocr_free(allocator);ggml_backend_buffer_free(input_buffer);ggml_free(c);
                }
            }
        }
        std::cout<<"{\"pass\":true,\"cases\":"<<cases<<",\"compared_I32_values\":"<<compared<<",\"strided_idx_cases\":"<<strided<<",\"head_distinct_cases\":"<<head_distinct<<",\"backend\":\""<<(cuda?"cuda":"cpu")<<"\",\"scope\":\"Original n1 idx expansion vs immutable lookup and independent closed form, including1M capacity and strided sorted picks. Component only; no Engine/FA/quality/speed acceptance\"}\n";
        ggml_backend_free(backend);return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';if(backend)ggml_backend_free(backend);return 1;}
}
