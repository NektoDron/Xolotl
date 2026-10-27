#pragma once
// PRIVATE COMPONENT INSTRUMENTATION ONLY. Never included by production Engine.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
namespace port7_cpu_trace {
constexpr int phases=6,max_workers=512;
inline const char * names[phases]={"input_quant","gate_up","oai","down_quant","down","rank_sum"};
struct Record {uint64_t begin=0,end=0,after=0,items=0;int cpu_begin=-1,cpu_end=-1;};
struct alignas(64) Worker {std::array<Record,phases>phase;};
alignas(64) inline Worker workers[max_workers];
inline uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
inline int cpu(){
#if defined(__linux__)
 return sched_getcpu();
#else
 return -1;
#endif
}
inline size_t count(size_t rows,int ith,int nth){return rows*size_t(ith+1)/size_t(nth)-rows*size_t(ith)/size_t(nth);}
inline void start(int ith,int p,uint64_t items){auto&r=workers[ith].phase[p];r={};r.items=items;if(p==1||p==4)r.cpu_begin=cpu();r.begin=now();}
inline void finish(int ith,int p){auto&r=workers[ith].phase[p];r.end=now();if(p==1||p==4)r.cpu_end=cpu();if(p==5)r.after=r.end;}
inline void passed(int ith,int p){workers[ith].phase[p].after=now();}
struct Sample {double mean_work=0,p50_work=0,p95_work=0,max_work=0,mean_wait=0,max_wait=0,work_span=0;uint64_t min_items=0,max_items=0,active=0,migrated=0;int tail_ith=0,tail_cpu=-1;};
inline std::map<std::pair<int,int>,std::vector<Sample>> samples;
struct WorkerTotals {uint64_t calls=0,active=0,work_ns=0,wait_ns=0,items=0,max_work_ns=0,migrations=0;int last_cpu=-1;};
inline std::map<std::pair<int,int>,std::array<WorkerTotals,max_workers>> worker_totals;
inline double percentile(std::vector<double> values,double fraction){if(values.empty())return 0;std::sort(values.begin(),values.end());return values[std::min(values.size()-1,size_t(fraction*double(values.size()-1)))];}
inline void commit(int layer,int nth){
 if(nth<=0||nth>max_workers)throw std::runtime_error("trace worker count");
 for(int p=0;p<phases;++p){Sample sample;std::vector<double> work;uint64_t first=UINT64_MAX,last=0;sample.min_items=UINT64_MAX;auto&totals=worker_totals[{layer,p}];
  for(int ith=0;ith<nth;++ith){const auto&r=workers[ith].phase[p];if(!r.begin||r.end<r.begin||r.after<r.end)throw std::runtime_error("incomplete private trace record");
   const double ns=double(r.end-r.begin),wait=double(r.after-r.end);sample.mean_wait+=wait;sample.max_wait=std::max(sample.max_wait,wait);
   auto&wt=totals[ith];++wt.calls;wt.wait_ns+=r.after-r.end;wt.items+=r.items;wt.last_cpu=r.cpu_end;
   if(r.items){++wt.active;wt.work_ns+=r.end-r.begin;wt.max_work_ns=std::max(wt.max_work_ns,r.end-r.begin);wt.migrations+=r.cpu_begin>=0&&r.cpu_end>=0&&r.cpu_begin!=r.cpu_end;}
   if(r.items){work.push_back(ns);sample.mean_work+=ns;sample.min_items=std::min(sample.min_items,r.items);sample.max_items=std::max(sample.max_items,r.items);first=std::min(first,r.begin);last=std::max(last,r.end);
    if(ns>sample.max_work){sample.max_work=ns;sample.tail_ith=ith;sample.tail_cpu=r.cpu_end;}sample.migrated+=r.cpu_begin>=0&&r.cpu_end>=0&&r.cpu_begin!=r.cpu_end;}
  }
  sample.active=work.size();if(!work.empty()){sample.mean_work/=work.size();sample.p50_work=percentile(work,.5);sample.p95_work=percentile(work,.95);sample.work_span=double(last-first);}sample.mean_wait/=nth;samples[{layer,p}].push_back(sample);
 }
}
inline void report(){
 for(const auto&entry:samples){const auto[layer,p]=entry.first;const auto&values=entry.second;std::vector<double>ratios,p95ratios,means,maxima,waitmeans,waitmaxima;std::map<int,size_t>tails,cores;uint64_t migrated=0;double active=0;
  for(const auto&s:values){ratios.push_back(s.max_work/std::max(1.,s.mean_work));p95ratios.push_back(s.p95_work/std::max(1.,s.p50_work));means.push_back(s.mean_work);maxima.push_back(s.max_work);waitmeans.push_back(s.mean_wait);waitmaxima.push_back(s.max_wait);++tails[s.tail_ith];++cores[s.tail_cpu];migrated+=s.migrated;active+=s.active;}
  std::cout<<"{\"kind\":\"cpu_skew_phase\",\"layer\":"<<layer<<",\"phase\":\""<<names[p]<<"\",\"calls\":"<<values.size()<<",\"mean_active_workers\":"<<active/values.size()<<",\"median_work_max_over_mean\":"<<percentile(ratios,.5)<<",\"p95_work_max_over_mean\":"<<percentile(ratios,.95)<<",\"median_worker_mean_ns\":"<<percentile(means,.5)<<",\"median_worker_max_ns\":"<<percentile(maxima,.5)<<",\"median_worker_p95_over_p50\":"<<percentile(p95ratios,.5)<<",\"median_barrier_wait_mean_ns\":"<<percentile(waitmeans,.5)<<",\"p95_barrier_wait_max_ns\":"<<percentile(waitmaxima,.95)<<",\"migrated_active_workers\":"<<migrated<<",\"tail_worker_counts\":{";
  bool comma=false;for(auto[ith,count]:tails){std::cout<<(comma?",":"")<<'"'<<ith<<"\":"<<count;comma=true;}std::cout<<"},\"tail_cpu_counts\":{";comma=false;for(auto[id,count]:cores){std::cout<<(comma?",":"")<<'"'<<id<<"\":"<<count;comma=true;}std::cout<<"},\"min_assigned_items\":"<<values.front().min_items<<",\"max_assigned_items\":"<<values.front().max_items<<",\"scope\":\"Private timestamp instrumentation, not throughput or Engine wall-time measurement\"}\n";
  if(p==1||p==4){std::cout<<"{\"kind\":\"cpu_skew_workers\",\"layer\":"<<layer<<",\"phase\":\""<<names[p]<<"\",\"workers\":[";comma=false;
   const auto&t=worker_totals.at({layer,p});for(int ith=0;ith<max_workers;++ith)if(t[ith].calls){const auto&w=t[ith];std::cout<<(comma?",":"")<<"{\"ith\":"<<ith<<",\"calls\":"<<w.calls<<",\"active_calls\":"<<w.active<<",\"items\":"<<w.items<<",\"total_work_ns\":"<<w.work_ns<<",\"total_barrier_wait_ns\":"<<w.wait_ns<<",\"max_work_ns\":"<<w.max_work_ns<<",\"migrations\":"<<w.migrations<<",\"last_cpu\":"<<w.last_cpu<<'}';comma=true;}std::cout<<"]}\n";
  }
 }
}
} // namespace port7_cpu_trace
