#include "cached_decode_merge_rank_outer.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
using namespace strata::m31;
using Fn=void(*)(size_t,const std::vector<uint8_t>&,const std::vector<float>&,const std::vector<float>&,std::vector<float>&);
extern "C" __attribute__((noinline,noclone,noipa)) void port7_merge_old(size_t e,const std::vector<uint8_t>&m,const std::vector<float>&g,const std::vector<float>&c,std::vector<float>&o){merge_cached_decode_terms(e,m,g,c,o);}
extern "C" __attribute__((noinline,noclone,noipa)) void port7_merge_rank_outer(size_t e,const std::vector<uint8_t>&m,const std::vector<float>&g,const std::vector<float>&c,std::vector<float>&o){merge_cached_decode_terms_rank_outer(e,m,g,c,o);}
static void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
static void same(const std::vector<float>&a,const std::vector<float>&b){need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"rank-outer F32 bits differ");}
static std::vector<float>full_terms(size_t embd,size_t used){std::vector<float>v(embd*used);for(size_t rank=0;rank<used;++rank)for(size_t row=0;row<embd;++row){const std::array<std::array<float,4>,8>edge{{{{-0.f,-0.f,-0.f,-0.f}},{{-0.f,0.f,-0.f,0.f}},{{1e20f,-1e20f,3.f,-2.f}},{{std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),std::numeric_limits<float>::min(),-std::numeric_limits<float>::min()}},{{std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min()}},{{.1f,-.2f,.3f,-.4f}},{{16777216.f,1.f,-16777216.f,1.f}},{{-1e20f,1e20f,-3.f,2.f}}}};v[rank*embd+row]=row<edge.size()?edge[row][rank%4]:float(int((row*37+rank*193)%16384)-8192)/1024.f;}return v;}
struct Inputs {std::vector<uint8_t>mask;std::vector<float>gpu,cpu;};
static Inputs inputs(size_t e,size_t used,unsigned bits){Inputs x;x.mask.resize(used);x.gpu.assign(e*used,std::numeric_limits<float>::quiet_NaN());const auto terms=full_terms(e,used);for(size_t rank=0;rank<used;++rank){x.mask[rank]=bool(bits&(1u<<(rank%4)));const auto*start=terms.data()+rank*e;if(x.mask[rank])std::copy_n(start,e,x.gpu.data()+rank*e);else x.cpu.insert(x.cpu.end(),start,start+e);}return x;}
static volatile uint32_t sink=0;
static double block_ns(Fn fn,const Inputs&x,size_t e,std::vector<float>&out,int count){const auto started=std::chrono::steady_clock::now();for(int i=0;i<count;++i)fn(e,x.mask,x.gpu,x.cpu,out);const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-started).count()/count;sink=uint32_t(sink)^std::bit_cast<uint32_t>(out[37%e]);return ns;}
int main(int argc,char**argv){try{
 need(argc==3,"usage: merge_rank_check check|bench REPORT.json");const bool bench=std::string(argv[1])=="bench";need(bench||std::string(argv[1])=="check","mode");size_t cases=0,values=0,invalid=0;
 for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){need(std::fesetround(rounding)==0,"set rounding");for(size_t embd:{size_t(1),size_t(7),size_t(31),size_t(256),size_t(6144)})for(size_t used:{size_t(1),size_t(2),size_t(4),size_t(8)})for(unsigned mask=0;mask<16;++mask){auto x=inputs(embd,used,mask);std::vector<float>a,b;port7_merge_old(embd,x.mask,x.gpu,x.cpu,a);port7_merge_rank_outer(embd,x.mask,x.gpu,x.cpu,b);same(a,b);++cases;values+=a.size();}}
 std::fesetround(FE_TONEAREST);
 {auto x=inputs(7,129,7);std::vector<float>a,b;port7_merge_old(7,x.mask,x.gpu,x.cpu,a);port7_merge_rank_outer(7,x.mask,x.gpu,x.cpu,b);same(a,b);++cases;values+=a.size();}
 for(int bad=0;bad<4;++bad){auto x=inputs(7,4,5);size_t e=7;if(bad==0)e=0;if(bad==1)x.mask[0]=2;if(bad==2)x.gpu.pop_back();if(bad==3)x.cpu.pop_back();std::vector<float>out{42.f};bool rejected=false;try{port7_merge_rank_outer(e,x.mask,x.gpu,x.cpu,out);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&out==std::vector<float>{42.f},"invalid input changed output");++invalid;}
 std::ofstream report(argv[2]);report<<"{\"pass\":true,\"cases\":"<<cases<<",\"f32_values\":"<<values<<",\"rounding_modes\":4,\"invalid_unchanged_checks\":"<<invalid<<",\"mode\":\""<<(bench?"bench":"check")<<"\",\"results\":[";bool first=true;
 if(bench)for(unsigned mask=1;mask<15;++mask){auto x=inputs(6144,4,mask);std::vector<float>a,b;for(int i=0;i<32;++i){port7_merge_old(6144,x.mask,x.gpu,x.cpu,a);port7_merge_rank_outer(6144,x.mask,x.gpu,x.cpu,b);}std::vector<double>ta,tb;for(int block=0;block<16;++block){if(block%2){tb.push_back(block_ns(port7_merge_rank_outer,x,6144,b,128));ta.push_back(block_ns(port7_merge_old,x,6144,a,128));}else{ta.push_back(block_ns(port7_merge_old,x,6144,a,128));tb.push_back(block_ns(port7_merge_rank_outer,x,6144,b,128));}}same(a,b);std::sort(ta.begin(),ta.end());std::sort(tb.begin(),tb.end());const double old=ta[ta.size()/2],candidate=tb[tb.size()/2];if(!first)report<<",";first=false;report<<"{\"mask\":"<<mask<<",\"hits\":"<<std::popcount(mask)<<",\"embd\":6144,\"old_ns_per_call\":"<<old<<",\"rank_outer_ns_per_call\":"<<candidate<<",\"speedup\":"<<old/candidate<<"}";}
 report<<"],\"scope\":\"Pure CPU full-rank GPU/compressed-CPU merge only; original rank FP add order, signedzero/cancellation/subnormal and four rounding modes; no production/Engine/TG acceptance\"}\n";need(bool(report),"report write");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
