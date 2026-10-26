// Independent route-scheduling prototype; the production CUDA TU is unchanged.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::m31 {
class PrefillRouteQuants {
public:
    PrefillRouteQuants();
    ~PrefillRouteQuants();
    PrefillRouteQuants(const PrefillRouteQuants &)=delete;
    PrefillRouteQuants & operator=(const PrefillRouteQuants &)=delete;
    std::vector<float> multiply(int type,const void * weights,size_t slot_stride,
                               int columns,int rows,int used,int n,
                               const std::vector<float> & input,const std::vector<int32_t> & ids,
                               bool sort_by_expert);
    size_t scratch_bytes()const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
