cmake_minimum_required(VERSION 3.22)
project(m31_cache_calibration LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(M31_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(M31_BASE "${M31_ROOT}/build-m31-cuda" CACHE PATH "Completed CUDA vendor build")
set(M31_ENGINE_BUILD "${M31_ROOT}/build-cpu-decode" CACHE PATH "Completed CPU-decode build with route_observer ABI")
set(M31_VENDOR "${M31_ROOT}/build-m31-cpu/_deps/strata_llamacpp-src")
find_package(CUDAToolkit REQUIRED)
add_executable(m31_cache_calibrate ../cache_calibrate.cpp)
target_include_directories(m31_cache_calibrate PRIVATE "${M31_ROOT}/include" "${M31_ROOT}/tools/m31"
  "${M31_VENDOR}/ggml/include" "${M31_VENDOR}/vendor/nlohmann")
target_link_libraries(m31_cache_calibrate PRIVATE "${M31_ENGINE_BUILD}/libm31_engine.a"
  "${M31_BASE}/ggml/src/libggml.a" "${M31_BASE}/ggml/src/libggml-cpu.a"
  "${M31_BASE}/ggml/src/ggml-cuda/libggml-cuda.a" "${M31_BASE}/ggml/src/libggml-base.a"
  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos CUDA::cuda_driver m dl rt pthread)
