cmake_minimum_required(VERSION 3.24)
project(m31_vision_adapter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(M31_PR /mnt/ai/llama.cpp-m31pr-1026 CACHE PATH "Read-only audited PR source")
set(M31_PR_BUILD "${M31_PR}/build" CACHE PATH "Read-only completed PR build")
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${M31_PR}"
  OUTPUT_VARIABLE sha OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT sha STREQUAL "6b95a9908d2c7b242817a447708aec38e937e2c0")
  message(FATAL_ERROR "Unexpected PR SHA: ${sha}")
endif()
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
add_executable(m31_vision_adapter ../vision_adapter.cpp)
target_include_directories(m31_vision_adapter PRIVATE "${M31_PR}/include" "${M31_PR}/ggml/include"
  "${M31_PR}/tools/mtmd" "${M31_PR}/common" "${M31_PR}/vendor" "${M31_PR}/vendor/nlohmann")
target_link_libraries(m31_vision_adapter PRIVATE
  "${M31_PR_BUILD}/common/libllama-common.a"
  "${M31_PR_BUILD}/tools/mtmd/libmtmd.a"
  "${M31_PR_BUILD}/common/libllama-common-base.a"
  "${M31_PR_BUILD}/src/libllama.a"
  "${M31_PR_BUILD}/ggml/src/libggml.a"
  "${M31_PR_BUILD}/ggml/src/libggml-cpu.a"
  "${M31_PR_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a"
  "${M31_PR_BUILD}/ggml/src/libggml-base.a"
  "${M31_PR_BUILD}/vendor/hash/libvendor-hash.a"
  OpenMP::OpenMP_CXX Threads::Threads CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::culibos CUDA::cuda_driver m dl rt)
