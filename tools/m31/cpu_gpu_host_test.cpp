#include "cpu_gpu_host_ops.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
using namespace strata::m31;
static size_t checks = 0;
static void check(bool ok, const char * reason) { ++checks; if (!ok) throw std::runtime_error(reason); }
static void equal_bytes(const void * a, const void * b, size_t bytes, const char * reason) {
    check(std::memcmp(a, b, bytes) == 0, reason);
}
template<class F> static void rejects(F && fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception &) { failed = true; }
    check(failed, "expected exception was not raised");
}
static std::vector<float> read_floats(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("missing real fixture: " + path.string());
    const auto bytes = stream.tellg();
    check(bytes >= 0 && bytes % 4 == 0, "bad raw float fixture length");
    std::vector<float> result(size_t(bytes) / 4);
    stream.seekg(0); stream.read(reinterpret_cast<char *>(result.data()), bytes);
    check(bool(stream), "short raw float fixture");
    return result;
}
static std::vector<float> make_input(size_t rows, size_t columns, const std::vector<float> * real = nullptr) {
    std::vector<float> result(rows * columns);
    if (real) {
        check(real->size() >= columns, "real fixture too short");
        for (size_t i = 0; i < result.size(); ++i) result[i] = (*real)[i % real->size()];
    } else {
        std::mt19937 random(1031);
        std::uniform_real_distribution<float> values(-10, 10);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = values(random);
            if (i % 257 == 0) result[i] = std::numeric_limits<float>::denorm_min();
            if (i % 509 == 0) result[i] = -0.f;
        }
    }
    return result;
}
// Independent preservation of the old serial loops, rather than invoking the
// new helper with one worker. Source provenance is recorded beside this test.
static std::vector<uint8_t> old_quant(ggml_type type, size_t columns, const std::vector<float> & input) {
    const auto * traits = ggml_get_type_traits_cpu(type);
    const auto * quant_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    const size_t qbytes = ggml_row_size(traits->vec_dot_type, columns);
    const size_t input_rows = input.size() / columns;
    std::vector<uint8_t> packed(qbytes * input_rows);
    for (size_t row = 0; row < input_rows; ++row)
        quant_traits->from_float(input.data()+row*columns, packed.data()+row*qbytes, columns);
    return packed;
}
static std::vector<float> old_oai(const std::vector<float> & gate, const std::vector<float> & up) {
    std::vector<float> activation(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const float a = std::min(gate[i], 7.f), b = std::clamp(up[i], -7.f, 7.f);
        const float glu = a / (1.f + expf(1.702f*(-a)));
        activation[i] = glu*(b+1.f);
    }
    return activation;
}
static void arithmetic(GpuHostExecutor & pool, const std::vector<float> & real_input,
                       const std::vector<float> & real_gate, const std::vector<float> & real_up,
                       const std::vector<float> & real_oai) {
    for (size_t rows : {size_t(1), size_t(7), size_t(128), size_t(2048)}) {
        for (bool real : {false, true}) {
            const size_t columns = real ? 6144 : 768;
            auto input = make_input(rows, columns, real ? &real_input : nullptr);
            for (auto type : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_NVFP4}) {
                const auto expected = old_quant(type, columns, input);
                const auto layout = gpu_host_quant_layout(type, columns, rows);
                check(layout.activation_type == (type == GGML_TYPE_NVFP4 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K), "unexpected activation quant type");
                std::vector<uint8_t> actual(layout.bytes, 0xa5);
                gpu_host_quantize_rows(pool, type, columns, input, actual);
                equal_bytes(expected.data(), actual.data(), expected.size(), "Q8_K/Q8_0 bytes changed with parallel rows");
            }
            const size_t width = real ? 3072 : 769; // OAI has no quant-block alignment requirement.
            auto gate = make_input(rows, width, real ? &real_gate : nullptr);
            auto up = make_input(rows, width, real ? &real_up : nullptr);
            if (!real) for (size_t i = 0; i < gate.size(); i += 4093) gate[i] = -100.f;
            const auto expected = old_oai(gate, up);
            std::vector<float> actual(gate.size(), -123.f);
            gpu_host_oai_rows(pool, width, gate, up, actual);
            equal_bytes(expected.data(), actual.data(), expected.size()*4, "OAI F32 bytes changed with parallel rows");
            if (real && rows == 128) equal_bytes(real_oai.data(), actual.data(), actual.size()*4, "OAI differs from saved ggml CPU oracle");
            if (rows <= 128) {
                auto in_place = gate;
                gpu_host_oai_rows(pool, width, in_place, up, in_place);
                equal_bytes(expected.data(), in_place.data(), expected.size()*4, "in-place OAI changed bytes");
            }
        }
    }
}
static void lifecycle() {
    GpuHostExecutor pool({16, 32, 4});
    const auto initial = pool.stats();
    check(!pool.parallel_rows(1, [](size_t a, size_t b) { check(a==0 && b==1, "bad serial range"); }), "one row was parallelized");
    check(!pool.parallel_rows(4, [](size_t, size_t) {}), "decode/down4 was parallelized");
    check(pool.stats().worker_ranges == initial.worker_ranges && pool.stats().parallel_jobs == initial.parallel_jobs, "decode woke workers");
    check(!pool.parallel_rows(0, [](size_t, size_t) { throw std::runtime_error("zero count invoked callback"); }), "empty job was parallelized");
    rejects([&] { pool.parallel_rows(128, [](size_t begin, size_t end) { if (begin <= 73 && 73 < end) throw std::runtime_error("injected"); }); });
    std::vector<int> marks(1025, 0);
    pool.parallel_rows(marks.size(), [&](size_t begin, size_t end) { for (size_t i=begin;i<end;++i) ++marks[i]; });
    check(std::all_of(marks.begin(), marks.end(), [](int n) { return n==1; }), "ranges lost/overlapped after exception");
    rejects([&] { pool.parallel_rows(128, [&](size_t, size_t) { pool.parallel_rows(1, [](size_t,size_t) {}); }); });
    rejects([&] { pool.parallel_rows(128, [&](size_t,size_t) { pool.close(); }); });
    std::atomic<int> finished{0};
    auto submit = [&] { for(int i=0;i<20;++i) pool.parallel_rows(128, [&](size_t,size_t) { finished.fetch_add(1); }); };
    std::thread first(submit), second(submit); first.join(); second.join();
    check(finished == 40*16, "concurrent submissions lost work");
    std::promise<void> entered;
    std::mutex gate_mutex; std::condition_variable gate_cv; bool release = false;
    std::thread busy([&] { pool.parallel_rows(128, [&](size_t begin,size_t) {
        if (begin == 0) { entered.set_value(); std::unique_lock lock(gate_mutex); gate_cv.wait(lock,[&]{return release;}); }
    }); });
    check(entered.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready, "job never entered");
    std::thread closer([&] { pool.close(); });
    { std::lock_guard lock(gate_mutex); release=true; } gate_cv.notify_all();
    busy.join(); closer.join(); pool.close();
    rejects([&] { pool.parallel_rows(1, [](size_t,size_t) {}); });
    rejects([] { GpuHostExecutor invalid({65,32,4}); });
    rejects([] { GpuHostExecutor invalid({0,32,4}); });
    rejects([] { GpuHostExecutor invalid({16,0,4}); });
}
static void invalid_buffers() {
    GpuHostExecutor pool({16,32,4});
    std::vector<float> input(768*128), output(input.size()), short_input(769);
    std::vector<uint8_t> packed(gpu_host_quant_layout(GGML_TYPE_IQ3_XXS,768,128).bytes);
    rejects([&] { gpu_host_quantize_rows(pool,GGML_TYPE_IQ3_XXS,768,short_input,packed); });
    rejects([&] { gpu_host_quantize_rows(pool,GGML_TYPE_F32,768,input,packed); });
    rejects([&] { gpu_host_quant_layout(GGML_TYPE_IQ3_XXS,769,128); });
    rejects([&] { gpu_host_quant_layout(GGML_TYPE_IQ3_XXS,768,SIZE_MAX); });
    rejects([&] { gpu_host_quantize_rows(pool,GGML_TYPE_IQ3_XXS,768,input.data(),input.size(),packed.data(),packed.size()-1); });
    rejects([&] { gpu_host_quantize_rows(pool,GGML_TYPE_IQ3_XXS,768,input.data(),input.size(),reinterpret_cast<uint8_t*>(input.data()),packed.size()); });
    rejects([&] { gpu_host_quantize_rows(pool,GGML_TYPE_IQ3_XXS,768,nullptr,input.size(),packed.data(),packed.size()); });
    rejects([&] { gpu_host_oai_rows(pool,768,input.data(),input.data(),input.data()+1,input.size()-768); });
    rejects([&] { gpu_host_oai_rows(pool,0,input,input,output); });
    gpu_host_quantize_rows(pool,GGML_TYPE_IQ3_XXS,768,nullptr,0,nullptr,0);
    gpu_host_oai_rows(pool,768,nullptr,nullptr,nullptr,0);
}
static void fp_environment() {
    const int initial_round = fegetround();
#if defined(__SSE__)
    const unsigned initial_mxcsr = _mm_getcsr();
#endif
    GpuHostExecutor pool({16,32,4});
    for (int mode : {FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        check(fesetround(mode)==0, "cannot set rounding mode");
#if defined(__SSE__)
        for (unsigned flush : {0U,0x8040U}) {
            _mm_setcsr((_mm_getcsr() & ~0x8040U) | flush);
#else
        {
#endif
            auto gate = make_input(128,768), up = make_input(128,768);
            gate[768*127]=-100.f; // worker-only overflow/inexact must reach caller.
            feclearexcept(FE_ALL_EXCEPT);
            const auto expected = old_oai(gate,up); const int expected_flags=fetestexcept(FE_ALL_EXCEPT);
            feclearexcept(FE_ALL_EXCEPT);
            std::vector<float> actual(gate.size()); gpu_host_oai_rows(pool,768,gate,up,actual);
            equal_bytes(expected.data(),actual.data(),actual.size()*4,"OAI FP environment changed");
            check(fegetround()==mode, "caller rounding mode not restored");
            check(fetestexcept(FE_ALL_EXCEPT)==expected_flags, "FP exception flags not merged");
#if defined(__SSE__)
            check((_mm_getcsr() & 0x8040U)==flush,"caller FTZ/DAZ changed");
#endif
            for (auto type : {GGML_TYPE_IQ3_XXS,GGML_TYPE_NVFP4}) {
                const auto quant=old_quant(type,768,gate); std::vector<uint8_t> got(quant.size());
                gpu_host_quantize_rows(pool,type,768,gate,got);
                equal_bytes(quant.data(),got.data(),got.size(),"quantizer FP environment changed");
            }
            pool.parallel_rows(128,[&](size_t,size_t) { if(fegetround()!=mode) throw std::runtime_error("worker rounding mismatch"); });
        }
    }
    fesetround(initial_round);
#if defined(__SSE__)
    _mm_setcsr(initial_mxcsr);
#endif
}
int main(int argc,char ** argv) {
    try {
        const std::filesystem::path root=argc>1 ? argv[1] : "/mnt/ai/logs/m31-port/expert-probe-raw-v1";
        const auto input=read_floats(root/"l10_n128_full_input.f32");
        const auto gate=read_floats(root/"l10_n128_full_cpu_id_gate_scaled.f32");
        const auto up=read_floats(root/"l10_n128_full_cpu_id_up_scaled.f32");
        const auto oai=read_floats(root/"l10_n128_full_cpu_id_oai.f32");
        check(input.size()==128*6144 && gate.size()==128*3072 && up.size()==gate.size() && oai.size()==gate.size(),"unexpected real fixture dimensions");
        for (unsigned threads : {1U,16U,32U,64U}) {
            GpuHostExecutor pool({threads,32,4}); arithmetic(pool,input,gate,up,oai);
            std::cout << "threads=" << threads << ": serial/parallel Q8_K,Q8_0,OAI bit-identical; n=1,7,128,2048; synthetic+real\n";
        }
        lifecycle(); invalid_buffers(); fp_environment();
        std::cout << "PASS checks=" << checks << " (correctness only; no performance claim)\n";
        return 0;
    } catch(const std::exception & error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
