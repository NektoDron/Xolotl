#include "cpu_gpu_host_executor.hpp"
#include "cpu_threadpool_config.hpp"
#include <atomic>
#include <cstdio>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
using namespace strata::m31;
void ensure(bool ok) { if (!ok) throw std::runtime_error("affinity check failed"); }
int main() {
    try {
        cpu_set_t original; ensure(!pthread_getaffinity_np(pthread_self(),sizeof(original),&original));
        std::vector<unsigned> cpus;
        for(unsigned i=0;i<CPU_SETSIZE && cpus.size()<2;++i) if(CPU_ISSET(i,&original))cpus.push_back(i);
        ensure(cpus.size()==2);
        cpu_set_t inherited; CPU_ZERO(&inherited); CPU_SET(cpus[0],&inherited);
        ensure(!pthread_setaffinity_np(pthread_self(),sizeof(inherited),&inherited));
        {
            GpuHostExecutor pool({2,1,1,cpus});
            std::atomic<unsigned> checks{0};
            pool.parallel_count(8,[&](size_t begin,size_t) {
                ensure(sched_getcpu()==int(cpus[begin==0 ? 0 : 1])); ++checks;
            });
            ensure(checks==2);
            try { pool.parallel_count(8,[&](size_t begin,size_t) { if(begin==0) throw std::runtime_error("injected"); }); }
            catch(const std::exception &) {}
            cpu_set_t current; ensure(!pthread_getaffinity_np(pthread_self(),sizeof(current),&current));
            ensure(CPU_EQUAL(&current,&inherited));
        }
        ensure(!pthread_setaffinity_np(pthread_self(),sizeof(original),&original));
        auto params=ggml_threadpool_params_default(2);
        auto mask=std::to_string(cpus[0])+","+std::to_string(cpus[1]);
        setenv("M31_CPU_AFFINITY",mask.c_str(),1); setenv("M31_CPU_POLL","0",1);
        apply_cpu_pool_environment(params);
        ensure(params.strict_cpu && params.cpumask[cpus[0]] && params.cpumask[cpus[1]] && params.poll==0);
        for(const char * invalid : {"", "0,", "2-1", "0,,1", "x", "999999"}) {
            setenv("M31_CPU_AFFINITY",invalid,1); bool rejected=false;
            try{ apply_cpu_pool_environment(params); }catch(const std::invalid_argument &){rejected=true;}
            ensure(rejected);
        }
        unsetenv("M31_CPU_AFFINITY"); unsetenv("M31_CPU_POLL");
        for (const auto & names : {std::pair{"M31_GPU_CPU_QUANTS_HOST_THREADS","M31_GPU_HOST_THREADS"},
                                   std::pair{"M31_GPU_SORT_ROUTES","M31_GPU_EXPERT_SORT_ROUTES"}}) {
            setenv(names.first,"32",1); bool rejected=false;
            try { (void)gpu_host_options_from_environment(); }
            catch(const std::invalid_argument & error) { rejected=std::string(error.what()).find(names.second)!=std::string::npos; }
            unsetenv(names.first); ensure(rejected);
        }
        setenv("M31_GPU_HOST_THREADS","32",1);
        ensure(gpu_host_options_from_environment().threads==32);
        unsetenv("M31_GPU_HOST_THREADS");
        std::puts("PASS explicit worker placement, inherited single CPU, caller restore, exceptions, pool environment parsing, runtime typo rejection");
    }catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
