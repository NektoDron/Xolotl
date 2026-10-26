cmake_minimum_required(VERSION 3.22)
project(m31_prefill_pipeline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(M31_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(M31_VENDOR "${M31_ROOT}/build-m31-cpu/_deps/strata_llamacpp-src")
option(M31_PREFILL_CUDA "Link the existing CUDA backend for the event probe" ON)
add_executable(m31_prefill_check ../prefill_check.cpp)
target_include_directories(m31_prefill_check PRIVATE "${M31_ROOT}/include" "${M31_ROOT}/tools/m31"
 "${M31_VENDOR}/ggml/include" "${M31_VENDOR}/vendor/nlohmann")
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(M31_PREFILL_CUDA)
 find_package(CUDAToolkit REQUIRED)
 set(M31_BASE "${M31_ROOT}/build-m31-cuda")
 target_compile_definitions(m31_prefill_check PRIVATE M31_PREFILL_HAVE_CUDA=1)
 target_link_libraries(m31_prefill_check PRIVATE
  "${M31_BASE}/ggml/src/libggml.a" "${M31_BASE}/ggml/src/libggml-cpu.a"
  "${M31_BASE}/ggml/src/ggml-cuda/libggml-cuda.a" "${M31_BASE}/ggml/src/libggml-base.a"
  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos CUDA::cuda_driver)
else()
 set(M31_BASE "${M31_ROOT}/build-m31-cpu")
 target_link_libraries(m31_prefill_check PRIVATE
  "${M31_BASE}/ggml/src/libggml.a" "${M31_BASE}/ggml/src/libggml-cpu.a" "${M31_BASE}/ggml/src/libggml-base.a")
endif()
target_link_libraries(m31_prefill_check PRIVATE OpenMP::OpenMP_CXX Threads::Threads m dl rt)
