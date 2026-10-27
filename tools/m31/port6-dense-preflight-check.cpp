#include <bits/stdc++.h>
#define private public
#include "m31_engine.hpp"
#undef private
#include "m31_engine.cpp"
using namespace strata::m31;
int main(){try{
    int checks=0;auto check=[&](bool ok,const char *why){if(!ok)throw std::runtime_error(why);++checks;};
    Engine engine;auto &p=*engine.p_;
    // No model/backend/GPU is loaded. A deliberately missing embedding tests
    // the first execution failure after the real Engine's preflight boundary.
    p.ready=true;p.cfg.context=100000;p.cfg.max_chunk=8192;
    p.g.index_heads=4;p.g.block=128;p.g.vocab=100;p.g.embd=6144;
    p.pos=65536;p.last_n=3;p.hidden={1.f,2.f,3.f};
    p.decode_chain_active=true;p.prefill_chain_active=true;
    std::vector<int32_t> tokens(8192,0);std::string error;
    check(!engine.forward(tokens.data(),8192,error),"oversized pass accepted");
    check(error.find("int32")!=std::string::npos,"wrong oversized error");
    check(!p.poisoned,"oversized rejection poisoned state");
    check(p.pos==65536&&p.last_n==3&&p.hidden==std::vector<float>({1.f,2.f,3.f}),"preflight changed hidden/position");
    check(p.decode_chain_active&&p.prefill_chain_active,"preflight changed chain flags");
    check(engine.prefill_chunk(8192,error)==7358,"nonadaptive planner failed to limit int32");
    check(!p.poisoned&&p.pos==65536,"planner changed state");
    p.pos=57344;check(engine.prefill_chunk(8192,error)==8191,"strict INT32 boundary accepted equality");
    p.pos=65536;p.msa_index_tile=512;check(engine.prefill_chunk(8192,error)==8192,"index tiling incorrectly capped");
    p.msa_index_tile=0;
    tokens[0]=100;check(!engine.forward(tokens.data(),1,error),"invalid token accepted");
    check(!p.poisoned&&p.decode_chain_active&&p.prefill_chain_active,"invalid token changed valid state");
    tokens[0]=0;check(!engine.forward(tokens.data(),1,error),"missing embedding unexpectedly executed");
    check(error.find("token_embd.weight")!=std::string::npos,"execution failure did not reach embedding lookup");
    check(p.poisoned&&p.pos==65536,"execution failure must poison without advancing position");
    check(!engine.forward(tokens.data(),1,error)&&p.poisoned,"poisoned engine allowed retry");
    engine.reset();check(!p.poisoned&&p.pos==0&&p.last_n==0&&p.hidden.empty(),"reset did not recover state");
    check(!p.decode_chain_active&&!p.prefill_chain_active,"reset kept chain flags");
    check(!engine.forward(nullptr,1,error)&&!p.poisoned,"invalid pointer preflight poisoned state");
    check(!engine.forward(tokens.data(),0,error)&&!p.poisoned,"zero-size preflight poisoned state");
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"scope\":\"actual CPU Engine preflight/poison/reset with synthetic ready state; no model/GPU inference\"}\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
