cmake_minimum_required(VERSION 3.22)
project(m31_route_worklist LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
get_filename_component(M31_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(M31_VENDOR "${M31_ROOT}/build-m31-cpu/_deps/strata_llamacpp-src")
set(M31_BASE "${M31_ROOT}/build-m31-cuda")
set(M31_ROUTE_SOURCE "${M31_ROOT}/tools/m31" CACHE PATH "Immutable prototype source snapshot")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
add_library(m31_route_kernels STATIC "${M31_ROUTE_SOURCE}/gpu_cpu_quants.cu" "${M31_ROUTE_SOURCE}/prefill_route_quants.cu")
target_include_directories(m31_route_kernels PUBLIC "${M31_ROUTE_SOURCE}" "${M31_VENDOR}/ggml/include" "${M31_VENDOR}/ggml/src")
target_link_libraries(m31_route_kernels PUBLIC CUDA::cudart_static)
add_executable(m31_route_worklist_test "${M31_ROUTE_SOURCE}/prefill_route_worklist_test.cpp")
target_include_directories(m31_route_worklist_test PRIVATE "${M31_ROUTE_SOURCE}")
add_executable(m31_route_probe "${M31_ROUTE_SOURCE}/prefill_route_probe.cpp")
target_include_directories(m31_route_probe PRIVATE "${M31_ROUTE_SOURCE}" "${M31_ROOT}/tools/m31" "${M31_ROOT}/include"
 "${M31_VENDOR}/ggml/include" "${M31_VENDOR}/vendor/nlohmann")
target_link_libraries(m31_route_probe PRIVATE m31_route_kernels
 "${M31_BASE}/ggml/src/libggml.a" "${M31_BASE}/ggml/src/libggml-cpu.a"
 "${M31_BASE}/ggml/src/ggml-cuda/libggml-cuda.a" "${M31_BASE}/ggml/src/libggml-base.a"
 CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos CUDA::cuda_driver
 OpenMP::OpenMP_CXX Threads::Threads m dl rt)
