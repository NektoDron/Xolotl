#include "shortprefill_gpu_quant.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using nlohmann::json;
using namespace strata::m31;
namespace {
size_t checked_count(size_t columns,size_t rows){if(!columns||!rows||columns>65536||rows>65536||columns>SIZE_MAX/rows)throw std::invalid_argument("invalid dimensions");if(columns*rows>268435456)throw std::invalid_argument("input exceeds1GiB");return columns*rows;}
std::vector<float> read_f32(const std::string &path,size_t count){
    std::ifstream input(path,std::ios::binary|std::ios::ate);if(!input)throw std::runtime_error("cannot read "+path);
    if(input.tellg()!=std::streamoff(count*sizeof(float)))throw std::runtime_error("wrong input byte count for "+path);
    input.seekg(0);std::vector<float> values(count);input.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*sizeof(float)));
    if(!input)throw std::runtime_error("short input read");for(float value:values)if(!std::isfinite(value))throw std::runtime_error("nonfinite input");return values;
}
void write_bytes(const std::string &path,const void *data,size_t bytes){std::ofstream output(path,std::ios::binary);if(!output)throw std::runtime_error("cannot write "+path);output.write(static_cast<const char*>(data),std::streamsize(bytes));if(!output)throw std::runtime_error("short output write");}
uint32_t bits(float value){uint32_t result;std::memcpy(&result,&value,sizeof(value));return result;}
json check_q8(const std::vector<cuda_math_cpu::Q8Mmq32> &gpu,const std::vector<cuda_math_cpu::Q8Mmq32> &cpu,size_t columns){
    if(gpu.size()!=cpu.size())throw std::runtime_error("Q8 output sizes differ");
    size_t qs_diff=0,d_diff=0;double maximum_d=0;json examples=json::array();
    for(size_t group=0;group<gpu.size();++group){const auto &g=gpu[group],&c=cpu[group];
        if(bits(g.d)!=bits(c.d)){++d_diff;maximum_d=std::max(maximum_d,double(std::fabs(g.d-c.d)));if(examples.size()<32)examples.push_back({{"row",group/(columns/32)},{"group",group%(columns/32)},{"kind","scale"},{"gpu_bits",bits(g.d)},{"cpu_bits",bits(c.d)},{"gpu",g.d},{"cpu",c.d}});}
        for(size_t k=0;k<32;++k)if(g.qs[k]!=c.qs[k]){++qs_diff;if(examples.size()<32)examples.push_back({{"row",group/(columns/32)},{"column",(group%(columns/32))*32+k},{"kind","quant"},{"gpu",int(g.qs[k])},{"cpu",int(c.qs[k])}});}}
    return {{"groups",gpu.size()},{"values",gpu.size()*32},{"quant_differences",qs_diff},{"scale_bit_differences",d_diff},{"max_scale_abs",maximum_d},{"bitwise",qs_diff==0&&d_diff==0},{"first_differences",examples}};
}
void write_nv(const std::string &path,const std::vector<cuda_nvfp4_cpu::Activation> &rows){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot write "+path);
    for(const auto &row:rows){out.write(reinterpret_cast<const char*>(&row.row_scale),sizeof(float));out.write(reinterpret_cast<const char*>(row.scales.data()),std::streamsize(row.scales.size()));out.write(reinterpret_cast<const char*>(row.values.data()),std::streamsize(row.values.size()));}if(!out)throw std::runtime_error("short NVFP4 output write");
}
json check_nv(const std::vector<cuda_nvfp4_cpu::Activation> &gpu,const std::vector<cuda_nvfp4_cpu::Activation> &cpu){
    if(gpu.size()!=cpu.size())throw std::runtime_error("NVFP4 output sizes differ");size_t row_diff=0,scale_diff=0,value_diff=0,values=0;json examples=json::array();
    for(size_t row=0;row<gpu.size();++row){const auto &g=gpu[row],&c=cpu[row];if(g.scales.size()!=c.scales.size()||g.values.size()!=c.values.size())throw std::runtime_error("NVFP4 row dimensions differ");values+=g.values.size();
        if(bits(g.row_scale)!=bits(c.row_scale)){++row_diff;if(examples.size()<32)examples.push_back({{"row",row},{"kind","row_scale"},{"gpu_bits",bits(g.row_scale)},{"cpu_bits",bits(c.row_scale)}});}
        for(size_t i=0;i<g.scales.size();++i)if(g.scales[i]!=c.scales[i]){++scale_diff;if(examples.size()<32)examples.push_back({{"row",row},{"subblock",i},{"kind","sub_scale"},{"gpu",g.scales[i]},{"cpu",c.scales[i]}});}
        for(size_t i=0;i<g.values.size();++i)if(g.values[i]!=c.values[i]){++value_diff;if(examples.size()<32)examples.push_back({{"row",row},{"column",i},{"kind","quant"},{"gpu",int(g.values[i])},{"cpu",int(c.values[i])}});}}
    return {{"rows",gpu.size()},{"values",values},{"quant_differences",value_diff},{"scale_code_differences",scale_diff},{"row_scale_bit_differences",row_diff},{"bitwise",value_diff==0&&scale_diff==0&&row_diff==0},{"first_differences",examples}};
}
}
int main(int argc,char **argv){
    try{std::string input,up_path,type="iq3",output;size_t columns=0,rows=0;bool preallocate=false;
        for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::invalid_argument("missing argument value");const std::string key=argv[i],value=argv[i+1];
            if(key=="--input")input=value;else if(key=="--up")up_path=value;else if(key=="--type")type=value;else if(key=="--output")output=value;
            else if(key=="--reserve"){if(value!="0"&&value!="1")throw std::invalid_argument("--reserve must be0 or1");preallocate=value=="1";}
            else if(key=="--columns")columns=std::stoull(value);else if(key=="--rows")rows=std::stoull(value);else throw std::invalid_argument("unknown argument "+key);}
        if(input.empty()||output.empty()||output[0]!='/')throw std::invalid_argument("requires --input --columns --rows --type iq3|iq4|nvfp4 --output /absolute/prefix [--up up.f32]");
        if(type!="iq3"&&type!="iq4"&&type!="nvfp4")throw std::invalid_argument("unknown quantization type");
        const size_t count=checked_count(columns,rows);auto values=read_f32(input,count);std::vector<float> up,comparison=values;
        if(!up_path.empty()){up=read_f32(up_path,count);for(size_t i=0;i<count;++i){const float a=std::min(values[i],7.f),b=std::clamp(up[i],-7.f,7.f);const float glu=a/(1.f+expf(1.702f*(-a)));comparison[i]=glu*(b+1.f);}}
        if(!ShortPrefillGpuQuantizer::available())throw std::runtime_error("requires CUDA build");ShortPrefillGpuQuantizer quantizer;if(preallocate)quantizer.reserve(count,rows);
        json report={{"format","m31-shortprefill-gpu-quant-check-v1"},{"input",input},{"up",up_path},{"columns",columns},{"rows",rows},{"type",type},{"oai",!up.empty()},{"preallocated",preallocate},{"comparison","PR CUDA quantizer versus portable CPU arithmetic"}};
        if(type=="nvfp4"){
            std::vector<cuda_nvfp4_cpu::Activation> gpu,cpu(rows);if(up.empty())quantizer.quantize_nvfp4(values.data(),columns,rows,gpu);else quantizer.oai_quantize_nvfp4(values.data(),up.data(),columns,rows,gpu);
            for(size_t row=0;row<rows;++row)cuda_nvfp4_cpu::quantize(comparison.data()+row*columns,columns,cpu[row]);report["comparison_metrics"]=check_nv(gpu,cpu);
            write_nv(output+".gpu.nvfp4_rows",gpu);write_nv(output+".cpu.nvfp4_rows",cpu);
            report["artifacts"]={{"gpu",output+".gpu.nvfp4_rows"},{"cpu",output+".cpu.nvfp4_rows"},{"row_format","F32 global_scale; U8 subscale[columns/16]; I8 doubled_e2m1[columns]"}};
        }else{
            const auto quant_type=type=="iq3"?GGML_TYPE_IQ3_XXS:GGML_TYPE_IQ4_XS;
            std::vector<cuda_math_cpu::Q8Mmq32> gpu,cpu(rows*(columns/32));if(up.empty())quantizer.quantize_q8(quant_type,values.data(),columns,rows,gpu);else quantizer.oai_quantize_q8(quant_type,values.data(),up.data(),columns,rows,gpu);
            for(size_t row=0;row<rows;++row)cuda_math_cpu::quantize_mmq_d4_into(comparison.data()+row*columns,columns,cpu.data()+row*(columns/32));report["comparison_metrics"]=check_q8(gpu,cpu,columns);
            static_assert(sizeof(cuda_math_cpu::Q8Mmq32)==36);write_bytes(output+".gpu.q8_mmq32",gpu.data(),gpu.size()*sizeof(gpu[0]));write_bytes(output+".cpu.q8_mmq32",cpu.data(),cpu.size()*sizeof(cpu[0]));
            report["artifacts"]={{"gpu",output+".gpu.q8_mmq32"},{"cpu",output+".cpu.q8_mmq32"},{"group_format","F32 d; I8 qs[32], row-major groups"}};
        }
        if(!up.empty()){write_bytes(output+".cpu_oai.f32",comparison.data(),comparison.size()*sizeof(float));report["artifacts"]["cpu_oai"]=output+".cpu_oai.f32";}
        const auto &s=quantizer.stats();report["gpu_quant_stats"]={{"calls",s.calls},{"h2d_bytes",s.h2d_bytes},{"d2h_bytes",s.d2h_bytes},{"wall_seconds",s.wall_seconds},{"device_bytes",s.device_bytes},{"pinned_bytes",s.pinned_bytes},{"reserve_calls",s.reserve_calls},{"allocations",s.allocations},{"runtime_allocations",s.runtime_allocations},
            {"reserve_seconds",s.reserve_seconds},{"host_copy_seconds",s.host_copy_seconds},{"submit_seconds",s.submit_seconds},{"wait_seconds",s.wait_seconds},{"unpack_seconds",s.unpack_seconds},
            {"reserved_max_values",s.reserved_max_values},{"reserved_max_rows",s.reserved_max_rows},{"stream_priority",s.stream_priority}};
        std::ofstream receipt(output+".json");if(!receipt)throw std::runtime_error("cannot write receipt");receipt<<report.dump(2)<<'\n';std::cout<<report.dump()<<'\n';return 0;
    }catch(const std::exception &e){std::cerr<<"m31_shortprefill_gpu_quant_check: "<<e.what()<<'\n';return 2;}
}
