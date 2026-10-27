// Two complete expert-layer banks. Original registered bytes only; no CPU pack.
// Sources, destination tensors and every reader backend outlive this owner.
#pragma once
#include "prefill_pipeline.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

namespace strata::m31 {
class DirectLayerUploads {
public:
    struct Job {ggml_tensor *destination=nullptr;std::array<PrefillUploads::Span,3> spans{};};
    struct Stats {uint64_t batches=0,experts=0,copies=0,upload_bytes=0,padding_bytes=0;};
    DirectLayerUploads(ggml_backend_t consumer,size_t slot_bytes,PrefillUploads::Copier = {})
        :bytes_(slot_bytes) {
        require(consumer&&bytes_,"missing consumer or slot size");
        const auto device=ggml_backend_get_device(consumer);
        require(device&&ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU,"requires GPU backend");
        const char *session=std::getenv("M31_CARD_SESSION");
        require(session&&std::strcmp(session,"1")==0,"requires aictl/flock session");
        ggml_backend_dev_props properties{};ggml_backend_dev_get_props(device,&properties);
        require(properties.caps.async&&properties.caps.events,"requires async/event backend");
        try {
            zero_capacity_=std::min(bytes_,size_t(1)<<20);
            const auto host=ggml_backend_dev_host_buffer_type(device);
            require(host,"requires pinned zero-padding source");
            zeros_=ggml_backend_buft_alloc_buffer(host,zero_capacity_);
            require(zeros_,"zero-padding allocation failed");
            std::memset(ggml_backend_buffer_get_base(zeros_),0,zero_capacity_);
            for(auto &bank:banks_) {
                bank.copy=ggml_backend_dev_init(device,nullptr);
                bank.ready=ggml_backend_event_new(device);bank.readers=ggml_backend_event_new(device);
                require(bank.copy&&bank.ready&&bank.readers,"backend/event allocation failed");
            }
        }catch(...){destroy();throw;}
    }
    ~DirectLayerUploads(){destroy();}
    DirectLayerUploads(const DirectLayerUploads&)=delete;
    DirectLayerUploads&operator=(const DirectLayerUploads&)=delete;

    // Empty bank needs no reader fence. Reused bank waits the actual consumer's
    // release event on its copy backend before ANY replacement write. No host
    // or device synchronization occurs in a successful enqueue.
    void enqueue(size_t index,const std::vector<Job>&jobs,const GpuRegisteredArena&registration) {
        require(!poisoned_,"transport is poisoned after partial submission");
        auto &bank=get(index);
        require(bank.state==State::empty||bank.state==State::released,"bank still pending/acquired");
        require(!jobs.empty()&&jobs.size()<=128,"requires1..128 expert jobs");
        std::vector<std::pair<uintptr_t,uintptr_t>> destinations;
        destinations.reserve(jobs.size());
        for(const auto &job:jobs) {
            require(job.destination&&ggml_is_contiguous(job.destination)&&ggml_nbytes(job.destination)==bytes_,"invalid complete expert slot");
            const uintptr_t first=reinterpret_cast<uintptr_t>(job.destination->data);
            require(first&&bytes_<=std::numeric_limits<uintptr_t>::max()-first,"invalid destination storage");
            const auto range=std::make_pair(first,first+bytes_);
            for(auto old:destinations)require(!overlap(range,old),"aliased current-bank destinations");
            for(auto old:banks_[1-index].destinations)require(!overlap(range,old),"layer banks alias physical destinations");
            destinations.push_back(range);
            for(auto a=job.spans.begin();a!=job.spans.end();++a) {
                require(a->offset<=bytes_&&a->extent<=bytes_-a->offset&&a->bytes<=a->extent,"invalid role span");
                require(!a->bytes||registration.contains(a->data,a->bytes),"unregistered source range");
                require(a->extent-a->bytes<=zero_capacity_,"role padding exceeds bounded zero buffer");
                for(auto b=job.spans.begin();b!=a;++b)
                    require(!a->extent||!b->extent||a->offset>=b->offset+b->extent||b->offset>=a->offset+a->extent,"overlapping role spans");
            }
        }
        try {
            if(bank.state==State::released)ggml_backend_event_wait(bank.copy,bank.readers);
            for(const auto &job:jobs)for(const auto &span:job.spans) {
                size_t offset=0;
                while(offset<span.bytes) {
                    const auto *source=static_cast<const uint8_t*>(span.data)+offset;
                    const size_t count=registration.contiguous_bytes(source,span.bytes-offset);
                    require(count,"missing contiguous registered piece");
                    ggml_backend_tensor_set_async(bank.copy,job.destination,source,span.offset+offset,count);
                    offset+=count;stat_.upload_bytes+=count;++stat_.copies;
                }
                const size_t padding=span.extent-span.bytes;
                if(padding) {
                    ggml_backend_tensor_set_async(bank.copy,job.destination,ggml_backend_buffer_get_base(zeros_),span.offset+span.bytes,padding);
                    stat_.upload_bytes+=padding;stat_.padding_bytes+=padding;++stat_.copies;
                }
            }
            ggml_backend_event_record(bank.ready,bank.copy);
            bank.destinations=std::move(destinations);bank.state=State::pending;
            ++stat_.batches;stat_.experts+=jobs.size();
        }catch(...){
            // Incomplete replacement weights are never published as a valid bank.
            // Main owner also marks its model/cache state terminal on this error.
            poisoned_=true;ggml_backend_synchronize(bank.copy);bank.state=State::empty;throw;
        }
    }
    // The given backend is the real grouped-FFN reader, not an unrelated
    // bookkeeping backend. It must enqueue all slot consumers before release.
    void acquire(size_t index,ggml_backend_t actual_reader) {
        require(!poisoned_,"transport is poisoned");auto &bank=get(index);
        require(actual_reader&&bank.state==State::pending,"acquire requires pending bank/reader");
        require(ggml_backend_get_device(actual_reader)==ggml_backend_get_device(bank.copy),"reader belongs to another device");
        ggml_backend_event_wait(actual_reader,bank.ready);
        bank.reader=actual_reader;bank.state=State::acquired;
    }
    void release(size_t index) {
        auto &bank=get(index);require(bank.state==State::acquired&&bank.reader,"release without acquired reader");
        ggml_backend_event_record(bank.readers,bank.reader);bank.state=State::released;
    }
    // Error cleanup / model reset. Drains outstanding copies AND actual readers.
    // reset keeps the configured two backends/events; no ring-depth fallback.
    void reset() {
        drain();for(auto &bank:banks_){bank.state=State::empty;bank.destinations.clear();bank.reader=nullptr;}
        poisoned_=false;
    }
    void drain()noexcept {
        for(auto &bank:banks_)if(bank.copy)ggml_backend_synchronize(bank.copy);
        for(auto &bank:banks_)if(bank.reader)ggml_backend_synchronize(bank.reader);
    }
    const Stats&stats()const noexcept{return stat_;}
private:
    enum class State {empty,pending,acquired,released};
    struct Bank {
        ggml_backend_t copy=nullptr,reader=nullptr;
        ggml_backend_event_t ready=nullptr,readers=nullptr;
        State state=State::empty;
        std::vector<std::pair<uintptr_t,uintptr_t>> destinations;
    };
    static bool overlap(std::pair<uintptr_t,uintptr_t>a,std::pair<uintptr_t,uintptr_t>b){return a.first<b.second&&b.first<a.second;}
    static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(std::string("M31 layer banks: ")+message);}
    Bank &get(size_t index){require(index<2,"bank index exceeds double buffer");return banks_[index];}
    void destroy()noexcept {
        drain();
        for(auto &bank:banks_) {
            if(bank.ready)ggml_backend_event_free(bank.ready);
            if(bank.readers)ggml_backend_event_free(bank.readers);
            if(bank.copy)ggml_backend_free(bank.copy);
            bank.ready=bank.readers=nullptr;bank.copy=bank.reader=nullptr;
        }
        if(zeros_)ggml_backend_buffer_free(zeros_);zeros_=nullptr;
    }
    std::array<Bank,2> banks_{};
    ggml_backend_buffer_t zeros_=nullptr;
    size_t bytes_=0,zero_capacity_=0;
    Stats stat_{};bool poisoned_=false;
};
}
