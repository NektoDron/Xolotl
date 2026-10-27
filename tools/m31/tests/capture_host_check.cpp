#include "capture_host.hpp"
#include "draft_feature_view.hpp"
#include "ggml-cpu.h"
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace strata::m31;
int main() { ggml_backend_t cpu=ggml_backend_cpu_init(); try {
    int checks=0;auto check=[&](bool yes,const char * why){if(!yes)throw std::runtime_error(why);++checks;};
    check(cpu,"CPU backend");
    {
        CaptureHost capture;
        check(capture.prepare(cpu,7,5,11,false),"initial prepare did not allocate");
        check(!capture.pinned()&&capture.capacity_rows()==11,"host geometry");
        auto run=[&](size_t rows,int generation){
            capture.begin(rows);check(!rows||!capture.complete(rows),"stale slots exposed before capture");
            for(size_t slot=0;slot<5;++slot){
                for(size_t row=0;row<rows;++row)for(size_t col=0;col<7;++col)
                    capture.scratch()[row*7+col]=float(generation*10000+slot*1000+row*10+col);
                capture.scatter(slot,capture.scratch());
                check(capture.complete(rows)==(rows==0||slot==4),"incomplete slots exposed");
            }
            check(capture.complete(rows),"complete capture rejected");
            const float * values=capture.values(rows);
            for(size_t row=0;row<rows;++row)for(size_t slot=0;slot<5;++slot)for(size_t col=0;col<7;++col)
                check(values[row*35+slot*7+col]==float(generation*10000+slot*1000+row*10+col),"transpose/raw bytes changed");
            return values;
        };
        const auto * address=run(11,1);
        check(!capture.prepare(cpu,7,5,11,false),"same geometry reallocated");
        check(!capture.complete(11),"prepare retained valid metadata");
        check(run(3,2)==address,"backing address changed after reuse");
        run(0,3);check(!capture.complete(3),"zero rows exposed stale prior batch");
        capture.invalidate();check(!capture.complete(0),"invalidated rows exposed");
        bool rejected=false;try{capture.values(0);}catch(const std::exception &){rejected=true;}check(rejected,"invalidated getter accepted");
        rejected=false;try{capture.begin(12);}catch(const std::exception &){rejected=true;}check(rejected,"oversized batch accepted");
        run(1,4);check(capture.prepare(cpu,3,2,4,false),"changed geometry did not allocate");
        capture.begin(4);for(size_t slot=0;slot<2;++slot){std::fill_n(capture.scratch(),12,float(slot));capture.scatter(slot,capture.scratch());}
        check(capture.complete(4),"changed geometry unusable");
    }
    {
        auto epoch=std::make_shared<uint64_t>(0);DraftFeatureView survivor;
        {
            CaptureHost capture;capture.prepare(cpu,7,5,11,false);
            auto append=[&](size_t start,size_t rows) {
                capture.begin_window(rows,start);
                for(size_t slot=0;slot<5;++slot){
                    for(size_t row=0;row<rows;++row)for(size_t col=0;col<7;++col)
                        capture.scratch()[row*7+col]=float(slot*1000+(start+row)*10+col);
                    capture.scatter(slot,capture.scratch());
                }
            };
            append(0,5);
            DraftFeatureView early(capture.share_window(),epoch,100,5,35);
            check(early.valid()&&capture.has_live_view(),"window lease not owned/live");
            bool rejected=false;try{capture.begin_window(6,5);}catch(const std::exception &){rejected=true;}
            check(rejected,"capture overwrote a readonly live lease");
            ++*epoch;check(!early.valid(),"next epoch retained old view");
            rejected=false;try{early.data();}catch(const std::exception &){rejected=true;}check(rejected,"stale view data exposed");
            early.clear();capture.invalidate_batch();append(5,6);
            survivor=DraftFeatureView(capture.share_window(),epoch,100,11,35);
            check(capture.window_rows()==11&&survivor.valid(),"accumulated window incomplete");
            for(size_t row=0;row<11;++row)for(size_t slot=0;slot<5;++slot)for(size_t col=0;col<7;++col)
                check(survivor.data()[row*35+slot*7+col]==float(slot*1000+row*10+col),"whole accumulated window changed bytes");
            capture.invalidate_batch(); // release GPU workspace does not touch host epoch/data
            check(survivor.valid()&&survivor.data()[34]==4006.f,"workspace release broke plain-host lease");
        }
        check(survivor.valid()&&survivor.data()[34]==4006.f,"lease backing depended on scratch/backend owner");
        epoch.reset();check(!survivor.valid(),"destroyed engine epoch retained valid view");survivor.clear();
    }
    ggml_backend_free(cpu);std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"scope\":\"actual CPU ggml buffer; exact scatter/reuse/stale guards/zero window\"}\n";return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';if(cpu)ggml_backend_free(cpu);return 1;} }
