#pragma once
// Exact CPU routed FFN for a small layer-major speculative verification batch.
// Original weight formats, CPU quantizers and nrc=1 dots are retained. Several
// token assignments consume each expert row before advancing to the next row.
#include "cpu_direct_decode.hpp"
#include "cpu_batched_dot.hpp"
#include "ggml-alloc.h"
#include <array>
#include <memory>
#include <tuple>

extern "C" void ggml_barrier(struct ggml_threadpool *);
namespace strata::m31 {
namespace cpu_verify_detail {
inline std::pair<size_t,size_t> weighted_rows(size_t total_assignments,size_t prefix,size_t count,
                                            size_t rows,int ith,int nth) {
    const size_t total=total_assignments*rows;
    const size_t begin=total*size_t(ith)/size_t(nth),end=total*size_t(ith+1)/size_t(nth);
    const size_t first=prefix*rows,last=first+count*rows;
    if(end<=first||begin>=last)return {0,0};
    const size_t lo=begin>first?begin-first:0,hi=std::min(end,last)-first;
    return {(lo+count-1)/count,(hi+count-1)/count};
}
}
class CpuBatchedVerify {
    ggml_backend_t backend_;
    ggml_threadpool_t pool_;
    unsigned threads_, max_batch_;
    unsigned pair_mode_ = 0;
    bool masked_ = false;
    size_t active_assignments_ = 0;
    std::vector<uint8_t> active_tokens_;
    std::vector<size_t> selected_assignments_;
    ggml_context * context_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * output_tensor_ = nullptr;
    std::array<const ggml_tensor *,3> weight_{};
    std::array<const ggml_type_traits_cpu *,3> trait_{};
    std::array<const float *,3> scales_{};
    std::array<std::vector<uint8_t>,3> quant_;
    std::array<std::vector<size_t>,128> assignments_;
    std::vector<int32_t> experts_;
    std::vector<size_t> assignment_prefix_;
    std::vector<float> hidden_, down_;
    const float * input_ = nullptr, * route_weights_ = nullptr;
    size_t embd_ = 0, ff_ = 0, tokens_ = 0, used_ = 0;
    static std::pair<size_t,size_t> range(size_t size,int ith,int nth) {
        return {size*size_t(ith)/size_t(nth),size*size_t(ith+1)/size_t(nth)};
    }
    std::pair<size_t,size_t> expert_rows(size_t group,size_t rows,int ith,int nth) const {
        // Partition actual dot work, not the number of expert groups. A row
        // with eight token assignments costs eight times a singleton row.
        // Ceil both adjacent boundaries: every complete weight row has exactly
        // one worker, while its token assignments retain a common unpack/cache.
        const size_t count=assignments_[size_t(experts_[group])].size();
        return cpu_verify_detail::weighted_rows(active_assignments_,assignment_prefix_[group],count,rows,ith,nth);
    }
    void dot_two(int role,int columns,const void * source,const void * q0,const void * q1,float (& values)[2]) const {
        if(cpu_pair_detail::dot_pair(weight_[role]->type,columns,source,q0,q1,values[0],values[1],pair_mode_))return;
        // Unsupported types/ISA retain two independent accepted stock dots.
        trait_[role]->vec_dot(columns,&values[0],0,source,0,q0,0,1);
        trait_[role]->vec_dot(columns,&values[1],0,source,0,q1,0,1);
    }
    static void compute_masked(int ith,int nth,CpuBatchedVerify & self) {
        const auto & w=self.weight_;const auto & t=self.trait_;
        const size_t gate_stride=ggml_row_size(t[0]->vec_dot_type,int64_t(self.embd_));
        const size_t up_stride=ggml_row_size(t[1]->vec_dot_type,int64_t(self.embd_));
        const size_t down_stride=ggml_row_size(t[2]->vec_dot_type,int64_t(self.ff_));
        for(size_t token=size_t(ith);token<self.tokens_;token+=size_t(nth)) {
            if(!self.active_tokens_[token])continue;
            ggml_get_type_traits_cpu(t[0]->vec_dot_type)->from_float(self.input_+token*self.embd_,self.quant_[0].data()+token*gate_stride,int64_t(self.embd_));
            if(t[0]->vec_dot_type!=t[1]->vec_dot_type)
                ggml_get_type_traits_cpu(t[1]->vec_dot_type)->from_float(self.input_+token*self.embd_,self.quant_[1].data()+token*up_stride,int64_t(self.embd_));
        }
        ggml_barrier(self.pool_);
        const bool same_input=t[0]->vec_dot_type==t[1]->vec_dot_type;
        for(size_t group=0;group<self.experts_.size();++group) {
          const int32_t expert=self.experts_[group];
          const auto [begin,end]=self.expert_rows(group,self.ff_,ith,nth);
          for(size_t row=begin;row<end;++row) {
            const auto * gate_row=static_cast<const uint8_t *>(w[0]->data)+size_t(expert)*w[0]->nb[2]+row*w[0]->nb[1];
            const auto * up_row=static_cast<const uint8_t *>(w[1]->data)+size_t(expert)*w[1]->nb[2]+row*w[1]->nb[1];
            const auto & assignments=self.assignments_[size_t(expert)];
            if(!self.pair_mode_)for(const size_t assignment:assignments) {
                const size_t token=assignment/self.used_;
                float gate_value,up_value;
                t[0]->vec_dot(int(self.embd_),&gate_value,0,gate_row,0,self.quant_[0].data()+token*gate_stride,0,1);
                t[1]->vec_dot(int(self.embd_),&up_value,0,up_row,0,(same_input?self.quant_[0].data()+token*gate_stride:self.quant_[1].data()+token*up_stride),0,1);
                if(self.scales_[0])gate_value*=self.scales_[0][expert];
                if(self.scales_[1])up_value*=self.scales_[1][expert];
                const float a=std::min(gate_value,7.f),b=std::clamp(up_value,-7.f,7.f);
                const float glu=a/(1.f+expf(1.702f*(-a)));
                self.hidden_[assignment*self.ff_+row]=glu*(b+1.f);
            }
            else for(size_t index=0;index<assignments.size();) {
                const size_t count=index+1<assignments.size()?2:1;
                const size_t a0=assignments[index],token0=a0/self.used_;
                const auto * qg0=self.quant_[0].data()+token0*gate_stride;
                const auto * qu0=same_input?qg0:self.quant_[1].data()+token0*up_stride;
                float gates[2],ups[2];
                if(count==2) {
                    const size_t token1=assignments[index+1]/self.used_;
                    const auto * qg1=self.quant_[0].data()+token1*gate_stride;
                    const auto * qu1=same_input?qg1:self.quant_[1].data()+token1*up_stride;
                    self.dot_two(0,int(self.embd_),gate_row,qg0,qg1,gates);
                    self.dot_two(1,int(self.embd_),up_row,qu0,qu1,ups);
                } else {
                    t[0]->vec_dot(int(self.embd_),&gates[0],0,gate_row,0,qg0,0,1);
                    t[1]->vec_dot(int(self.embd_),&ups[0],0,up_row,0,qu0,0,1);
                }
                for(size_t offset=0;offset<count;++offset) {
                    float gate_value=gates[offset],up_value=ups[offset];
                    if(self.scales_[0])gate_value*=self.scales_[0][expert];
                    if(self.scales_[1])up_value*=self.scales_[1][expert];
                    const float a=std::min(gate_value,7.f),b=std::clamp(up_value,-7.f,7.f);
                    const float glu=a/(1.f+expf(1.702f*(-a)));
                    self.hidden_[assignments[index+offset]*self.ff_+row]=glu*(b+1.f);
                }
                index+=count;
            }
          }
        }
        ggml_barrier(self.pool_);
        for(size_t index=size_t(ith);index<self.selected_assignments_.size();index+=size_t(nth)) {
            const size_t assignment=self.selected_assignments_[index];
            ggml_get_type_traits_cpu(t[2]->vec_dot_type)->from_float(self.hidden_.data()+assignment*self.ff_,self.quant_[2].data()+assignment*down_stride,int64_t(self.ff_));
        }
        ggml_barrier(self.pool_);
        for(size_t group=0;group<self.experts_.size();++group) {
          const int32_t expert=self.experts_[group];
          const auto [begin,end]=self.expert_rows(group,self.embd_,ith,nth);
          for(size_t row=begin;row<end;++row) {
            const auto * down_row=static_cast<const uint8_t *>(w[2]->data)+size_t(expert)*w[2]->nb[2]+row*w[2]->nb[1];
            const auto & assignments=self.assignments_[size_t(expert)];
            if(!self.pair_mode_)for(const size_t assignment:assignments) {
                float value;t[2]->vec_dot(int(self.ff_),&value,0,down_row,0,self.quant_[2].data()+assignment*down_stride,0,1);
                if(self.scales_[2])value*=self.scales_[2][expert];
                self.down_[assignment*self.embd_+row]=value*self.route_weights_[assignment];
            }
            else for(size_t index=0;index<assignments.size();) {
                const size_t count=index+1<assignments.size()?2:1,a0=assignments[index];
                float values[2];
                if(count==2)self.dot_two(2,int(self.ff_),down_row,self.quant_[2].data()+a0*down_stride,
                    self.quant_[2].data()+assignments[index+1]*down_stride,values);
                else t[2]->vec_dot(int(self.ff_),&values[0],0,down_row,0,self.quant_[2].data()+a0*down_stride,0,1);
                for(size_t offset=0;offset<count;++offset) {
                    const size_t assignment=assignments[index+offset];float value=values[offset];
                    if(self.scales_[2])value*=self.scales_[2][expert];
                    self.down_[assignment*self.embd_+row]=value*self.route_weights_[assignment];
                }
                index+=count;
            }
          }
        }
        ggml_barrier(self.pool_);
    }
    static void compute(ggml_tensor * output,int ith,int nth,void * opaque) {
        auto & self=*static_cast<CpuBatchedVerify *>(opaque);
        if(self.masked_){compute_masked(ith,nth,self);return;}
        const auto & w=self.weight_;const auto & t=self.trait_;
        const size_t gate_stride=ggml_row_size(t[0]->vec_dot_type,int64_t(self.embd_));
        const size_t up_stride=ggml_row_size(t[1]->vec_dot_type,int64_t(self.embd_));
        const size_t down_stride=ggml_row_size(t[2]->vec_dot_type,int64_t(self.ff_));
        for(size_t token=size_t(ith);token<self.tokens_;token+=size_t(nth)) {
            ggml_get_type_traits_cpu(t[0]->vec_dot_type)->from_float(self.input_+token*self.embd_,self.quant_[0].data()+token*gate_stride,int64_t(self.embd_));
            if(t[0]->vec_dot_type!=t[1]->vec_dot_type)
                ggml_get_type_traits_cpu(t[1]->vec_dot_type)->from_float(self.input_+token*self.embd_,self.quant_[1].data()+token*up_stride,int64_t(self.embd_));
        }
        ggml_barrier(self.pool_);
        const bool same_input=t[0]->vec_dot_type==t[1]->vec_dot_type;
        for(size_t group=0;group<self.experts_.size();++group) {
          const int32_t expert=self.experts_[group];
          const auto [begin,end]=self.expert_rows(group,self.ff_,ith,nth);
          for(size_t row=begin;row<end;++row) {
            const auto * gate_row=static_cast<const uint8_t *>(w[0]->data)+size_t(expert)*w[0]->nb[2]+row*w[0]->nb[1];
            const auto * up_row=static_cast<const uint8_t *>(w[1]->data)+size_t(expert)*w[1]->nb[2]+row*w[1]->nb[1];
            const auto & assignments=self.assignments_[size_t(expert)];
            if(!self.pair_mode_)for(const size_t assignment:assignments) {
                const size_t token=assignment/self.used_;
                float gate_value,up_value;
                t[0]->vec_dot(int(self.embd_),&gate_value,0,gate_row,0,self.quant_[0].data()+token*gate_stride,0,1);
                t[1]->vec_dot(int(self.embd_),&up_value,0,up_row,0,(same_input?self.quant_[0].data()+token*gate_stride:self.quant_[1].data()+token*up_stride),0,1);
                if(self.scales_[0])gate_value*=self.scales_[0][expert];
                if(self.scales_[1])up_value*=self.scales_[1][expert];
                const float a=std::min(gate_value,7.f),b=std::clamp(up_value,-7.f,7.f);
                const float glu=a/(1.f+expf(1.702f*(-a)));
                self.hidden_[assignment*self.ff_+row]=glu*(b+1.f);
            }
            else for(size_t index=0;index<assignments.size();) {
                const size_t count=index+1<assignments.size()?2:1;
                const size_t a0=assignments[index],token0=a0/self.used_;
                const auto * qg0=self.quant_[0].data()+token0*gate_stride;
                const auto * qu0=same_input?qg0:self.quant_[1].data()+token0*up_stride;
                float gates[2],ups[2];
                if(count==2) {
                    const size_t token1=assignments[index+1]/self.used_;
                    const auto * qg1=self.quant_[0].data()+token1*gate_stride;
                    const auto * qu1=same_input?qg1:self.quant_[1].data()+token1*up_stride;
                    self.dot_two(0,int(self.embd_),gate_row,qg0,qg1,gates);
                    self.dot_two(1,int(self.embd_),up_row,qu0,qu1,ups);
                } else {
                    t[0]->vec_dot(int(self.embd_),&gates[0],0,gate_row,0,qg0,0,1);
                    t[1]->vec_dot(int(self.embd_),&ups[0],0,up_row,0,qu0,0,1);
                }
                for(size_t offset=0;offset<count;++offset) {
                    float gate_value=gates[offset],up_value=ups[offset];
                    if(self.scales_[0])gate_value*=self.scales_[0][expert];
                    if(self.scales_[1])up_value*=self.scales_[1][expert];
                    const float a=std::min(gate_value,7.f),b=std::clamp(up_value,-7.f,7.f);
                    const float glu=a/(1.f+expf(1.702f*(-a)));
                    self.hidden_[assignments[index+offset]*self.ff_+row]=glu*(b+1.f);
                }
                index+=count;
            }
          }
        }
        ggml_barrier(self.pool_);
        for(size_t assignment=size_t(ith);assignment<self.tokens_*self.used_;assignment+=size_t(nth))
            ggml_get_type_traits_cpu(t[2]->vec_dot_type)->from_float(self.hidden_.data()+assignment*self.ff_,self.quant_[2].data()+assignment*down_stride,int64_t(self.ff_));
        ggml_barrier(self.pool_);
        for(size_t group=0;group<self.experts_.size();++group) {
          const int32_t expert=self.experts_[group];
          const auto [begin,end]=self.expert_rows(group,self.embd_,ith,nth);
          for(size_t row=begin;row<end;++row) {
            const auto * down_row=static_cast<const uint8_t *>(w[2]->data)+size_t(expert)*w[2]->nb[2]+row*w[2]->nb[1];
            const auto & assignments=self.assignments_[size_t(expert)];
            if(!self.pair_mode_)for(const size_t assignment:assignments) {
                float value;t[2]->vec_dot(int(self.ff_),&value,0,down_row,0,self.quant_[2].data()+assignment*down_stride,0,1);
                if(self.scales_[2])value*=self.scales_[2][expert];
                self.down_[assignment*self.embd_+row]=value*self.route_weights_[assignment];
            }
            else for(size_t index=0;index<assignments.size();) {
                const size_t count=index+1<assignments.size()?2:1,a0=assignments[index];
                float values[2];
                if(count==2)self.dot_two(2,int(self.ff_),down_row,self.quant_[2].data()+a0*down_stride,
                    self.quant_[2].data()+assignments[index+1]*down_stride,values);
                else t[2]->vec_dot(int(self.ff_),&values[0],0,down_row,0,self.quant_[2].data()+a0*down_stride,0,1);
                for(size_t offset=0;offset<count;++offset) {
                    const size_t assignment=assignments[index+offset];float value=values[offset];
                    if(self.scales_[2])value*=self.scales_[2][expert];
                    self.down_[assignment*self.embd_+row]=value*self.route_weights_[assignment];
                }
                index+=count;
            }
          }
        }
        ggml_barrier(self.pool_);
        const auto [begin,end]=range(self.tokens_*self.embd_,ith,nth);
        auto * destination=static_cast<float *>(output->data);
        for(size_t item=begin;item<end;++item) {
            const size_t token=item/self.embd_,row=item%self.embd_;
            float sum=self.down_[token*self.used_*self.embd_+row];
            for(size_t rank=1;rank<self.used_;++rank)sum+=self.down_[(token*self.used_+rank)*self.embd_+row];
            destination[item]=sum;
        }
    }
public:
    CpuBatchedVerify(ggml_backend_t cpu,ggml_threadpool_t pool,unsigned threads,unsigned max_batch=8)
        :backend_(cpu),pool_(pool),threads_(threads),max_batch_(max_batch) {
        if(!cpu||!ggml_backend_is_cpu(cpu)||!pool||!threads||threads>GGML_MAX_N_THREADS||!max_batch||max_batch>8)
            throw std::invalid_argument("M31 batched CPU: requires CPU backend/matching persistent pool and max_batch1..8");
        if(const char * mode=std::getenv("M31_CPU_BATCH_PAIR")) {
            if(std::strcmp(mode,"0")==0)pair_mode_=0;
            else if(std::strcmp(mode,"1")==0)pair_mode_=1;
            else if(std::strcmp(mode,"2")==0)pair_mode_=2;
            else throw std::invalid_argument("M31 batched CPU: M31_CPU_BATCH_PAIR must be0,1or2");
        }
    }
    ~CpuBatchedVerify(){if(allocator_)ggml_gallocr_free(allocator_);if(context_)ggml_free(context_);}
    CpuBatchedVerify(const CpuBatchedVerify &)=delete;
    CpuBatchedVerify & operator=(const CpuBatchedVerify &)=delete;
    void forward(const ggml_tensor * wg,const ggml_tensor * wu,const ggml_tensor * wd,
                 const float * gate_scale,const float * up_scale,const float * down_scale,
                 const std::vector<float> & x,const std::vector<int32_t> & ids,
                 const std::vector<float> & weights,std::vector<float> & output) {
        forward_impl(wg,wu,wd,gate_scale,up_scale,down_scale,x,ids,weights,nullptr,output);
    }
    // cpu_mask is original [tokens,used]:1 computes a CPU miss,0 skips a GPU hit.
    // Returns weighted terms [tokens,used,embd], with +0 in skipped rank slots.
    // The caller merges GPU terms BEFORE summing in original per-token rank order.
    void forward_masked(const ggml_tensor * wg,const ggml_tensor * wu,const ggml_tensor * wd,
                 const float * gate_scale,const float * up_scale,const float * down_scale,
                 const std::vector<float> & x,const std::vector<int32_t> & ids,
                 const std::vector<float> & weights,const std::vector<uint8_t> & cpu_mask,
                 std::vector<float> & cpu_terms) {
        forward_impl(wg,wu,wd,gate_scale,up_scale,down_scale,x,ids,weights,&cpu_mask,cpu_terms);
    }
private:
    void forward_impl(const ggml_tensor * wg,const ggml_tensor * wu,const ggml_tensor * wd,
                 const float * gate_scale,const float * up_scale,const float * down_scale,
                 const std::vector<float> & x,const std::vector<int32_t> & ids,
                 const std::vector<float> & weights,const std::vector<uint8_t> * cpu_mask,
                 std::vector<float> & output) {
        if(!CpuDirectDecode::supported(wg,wu,wd)||!wg->data||!wu->data||!wd->data||wg->ne[0]<=0||wg->ne[1]<=0||
           wg->ne[2]<=0||wg->ne[2]>128||x.empty()||x.size()%size_t(wg->ne[0])||
           wu->ne[0]!=wg->ne[0]||wu->ne[1]!=wg->ne[1]||wd->ne[0]!=wg->ne[1]||wd->ne[1]!=wg->ne[0]||
           wg->ne[2]!=wu->ne[2]||wg->ne[2]!=wd->ne[2])
            throw std::invalid_argument("M31 batched CPU: invalid expert/input shape");
        const size_t tokens=x.size()/size_t(wg->ne[0]);
        if(tokens>max_batch_||ids.empty()||ids.size()%tokens||weights.size()!=ids.size()||ids.size()/tokens>128)
            throw std::invalid_argument("M31 batched CPU: invalid token/route count");
        const size_t used=ids.size()/tokens;
        if(cpu_mask&&(cpu_mask->size()!=ids.size()||std::any_of(cpu_mask->begin(),cpu_mask->end(),[](uint8_t x){return x>1;})))
            throw std::invalid_argument("M31 masked CPU: expected one0/1 mask per original assignment");
        for(size_t token=0;token<tokens;++token)for(size_t rank=0;rank<used;++rank) {
            const int32_t id=ids[token*used+rank];
            if(id<0||id>=wg->ne[2])throw std::invalid_argument("M31 batched CPU: invalid expert ID");
            for(size_t prior=0;prior<rank;++prior)if(ids[token*used+prior]==id)
                throw std::invalid_argument("M31 batched CPU: duplicate expert ID in a token");
        }
        if(graph_&&embd_!=size_t(wg->ne[0]))throw std::invalid_argument("M31 batched CPU: changed output graph width");
        embd_=size_t(wg->ne[0]);ff_=size_t(wg->ne[1]);tokens_=tokens;used_=used;
        weight_={wg,wu,wd};scales_={gate_scale,up_scale,down_scale};
        for(int role=0;role<3;++role) {
            trait_[role]=ggml_get_type_traits_cpu(weight_[role]->type);
            if(!trait_[role]||!trait_[role]->vec_dot||!ggml_get_type_traits_cpu(trait_[role]->vec_dot_type)->from_float)
                throw std::invalid_argument("M31 batched CPU: missing CPU trait");
            quant_[role].resize(ggml_row_size(trait_[role]->vec_dot_type,int64_t(role==2?ff_:embd_))*(role==2?ids.size():tokens_));
        }
        masked_=cpu_mask!=nullptr;active_assignments_=ids.size();
        active_tokens_.assign(tokens_,0);selected_assignments_.clear();
        for(auto & assignments:assignments_)assignments.clear();
        for(size_t assignment=0;assignment<ids.size();++assignment) {
            if(masked_&&!(*cpu_mask)[assignment])continue;
            assignments_[size_t(ids[assignment])].push_back(assignment);
            if(masked_){selected_assignments_.push_back(assignment);active_tokens_[assignment/used_]=1;}
        }
        if(masked_)active_assignments_=selected_assignments_.size();
        experts_.clear();assignment_prefix_.clear();size_t prefix=0;
        for(int32_t expert=0;expert<wg->ne[2];++expert)if(!assignments_[size_t(expert)].empty()) {
            experts_.push_back(expert);assignment_prefix_.push_back(prefix);prefix+=assignments_[size_t(expert)].size();
        }
        hidden_.resize(ids.size()*ff_);
        if(masked_)down_.assign(ids.size()*embd_,0.f);else down_.resize(ids.size()*embd_);
        input_=x.data();route_weights_=weights.data();
        if(masked_&&!active_assignments_){output=down_;return;}
        if(!graph_) {
            context_=ggml_init({16*1024,nullptr,true});if(!context_)throw std::runtime_error("M31 batched CPU: graph context allocation failed");
            graph_=ggml_new_graph_custom(context_,16,false);
            output_tensor_=ggml_custom_4d(context_,GGML_TYPE_F32,int64_t(embd_),int64_t(max_batch_),1,1,nullptr,0,compute,int(threads_),this);
            ggml_set_output(output_tensor_);ggml_build_forward_expand(graph_,output_tensor_);
            allocator_=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if(!allocator_||!ggml_gallocr_alloc_graph(allocator_,graph_))throw std::runtime_error("M31 batched CPU: graph allocation failed");
        }
        if(ggml_backend_graph_compute(backend_,graph_)!=GGML_STATUS_SUCCESS)throw std::runtime_error("M31 batched CPU: graph compute failed");
        if(masked_)output=down_;
        else {output.resize(tokens_*embd_);std::memcpy(output.data(),output_tensor_->data,output.size()*sizeof(float));}
    }
public:
    size_t unique_experts()const{return experts_.size();}
    const std::vector<float> & contributions()const{return down_;}
};
} // namespace strata::m31
