#include "m31_geometry.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::m31 {
#include "m31_dspark_batched_ops.hpp"
}
using namespace strata::m31;
static int checks=0;
static void check(bool value,const char * message){if(!value)throw std::runtime_error(message);++checks;}
template<class F>static void rejected(F operation){bool threw=false;try{operation();}catch(const std::runtime_error &){threw=true;}check(threw,"invalid metadata accepted");}
int main(){ggml_context * c=nullptr;try{
    c=ggml_init({32*1024*1024,nullptr,true});check(c,"metadata allocation");
    Geometry g;g.head_dim=128;g.rope_dim=64;g.context=1048576;g.rope_base=10000;g.rms_eps=1e-5f;g.experts=128;g.used=4;g.norm_routes=true;g.route_scale=2;
    auto * norm=ggml_new_tensor_1d(c,GGML_TYPE_F32,128);auto * hidden_norm=ggml_new_tensor_1d(c,GGML_TYPE_F32,6144);auto * bias=ggml_new_tensor_1d(c,GGML_TYPE_F32,128);
    for(int count:{1,2,3,4,7,8}){
        std::vector<ggml_tensor *> rows,hidden,logits;
        for(int i=0;i<count;++i){rows.push_back(ggml_new_tensor_3d(c,GGML_TYPE_F32,128,64,1));hidden.push_back(ggml_new_tensor_2d(c,GGML_TYPE_F32,6144,1));logits.push_back(ggml_new_tensor_2d(c,GGML_TYPE_F32,128,1));}
        auto * positions=ggml_new_tensor_1d(c,GGML_TYPE_I32,count);
        const auto rope=dspark_ops_norm_rope_rows(c,rows,norm,g.rms_eps,positions,g);
        const auto rms=dspark_ops_rms_rows(c,hidden,hidden_norm,g.rms_eps);
        const auto route=dspark_ops_router_rows(c,logits,bias,g);
        check(int(rope.size())==count&&int(rms.size())==count&&int(route.ids.size())==count&&int(route.weights.size())==count,"row counts changed");
        for(int i=0;i<count;++i){
            auto * out=rope[size_t(i)];check(out->ne[0]==128&&out->ne[1]==64&&out->ne[2]==1&&ggml_is_contiguous(out),"Q shape");
            check(rms[size_t(i)]->ne[0]==6144&&rms[size_t(i)]->ne[1]==1&&rms[size_t(i)]->ne[2]==1,"hidden RMS shape");
            check(route.ids[size_t(i)]->type==GGML_TYPE_I32&&route.ids[size_t(i)]->ne[0]==4&&route.weights[size_t(i)]->ne[0]==4,"router shape/type");
            auto * operation=out->view_src;check(operation&&operation->op==GGML_OP_ROPE,"missing RoPE dependency");
            auto * pos=operation->src[1];check(pos&&pos->view_src==positions&&pos->view_offs==size_t((i/4)*4)*sizeof(int32_t),"position group offset");
            check(pos->ne[0]==std::min(4,count-(i/4)*4),"position group length");
        }
        const int k=std::min(4,count);auto * packed=ggml_new_tensor_2d(c,GGML_TYPE_F32,6144,k);auto * glu=dspark_ops_glu(c,packed);
        check(glu->ne[0]==3072&&glu->ne[1]==k&&glu->src[1]==nullptr,"packed GLU shape/mapping");
    }
    rejected([&]{dspark_ops_rms_rows(c,{},hidden_norm,g.rms_eps);});
    rejected([&]{dspark_ops_glu(c,ggml_new_tensor_2d(c,GGML_TYPE_F32,6145,4));});
    rejected([&]{dspark_ops_glu(c,ggml_new_tensor_2d(c,GGML_TYPE_F32,6144,5));});
    std::vector<ggml_tensor *> one{ggml_new_tensor_3d(c,GGML_TYPE_F32,128,64,1)};
    rejected([&]{dspark_ops_norm_rope_rows(c,one,norm,g.rms_eps,ggml_new_tensor_1d(c,GGML_TYPE_F32,1),g);});
    rejected([&]{dspark_ops_norm_rope_rows(c,one,hidden_norm,g.rms_eps,ggml_new_tensor_1d(c,GGML_TYPE_I32,1),g);});
    ggml_free(c);c=nullptr;std::printf("{\"pass\":true,\"checks\":%d,\"scope\":\"CPU metadata-only production batch-op shape/dependency tests including K7/8 groups and absolute-position offsets; GPU numerical proof is separate\"}\n",checks);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"batched ops contract: %s\n",e.what());if(c)ggml_free(c);return 1;}}
