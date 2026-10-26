cmake_minimum_required(VERSION 3.22)
project(m31_thp_arena LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(M31_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(M31_VENDOR "${M31_ROOT}/build-m31-cpu/_deps/strata_llamacpp-src")
set(M31_BASE "${M31_ROOT}/build-m31-cpu")
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_executable(m31_thp_check ../prefill_thp_check.cpp)
target_include_directories(m31_thp_check PRIVATE "${M31_ROOT}/include" "${M31_ROOT}/tools/m31"
 "${M31_VENDOR}/ggml/include" "${M31_VENDOR}/vendor/nlohmann")
target_link_libraries(m31_thp_check PRIVATE "${M31_BASE}/ggml/src/libggml.a" "${M31_BASE}/ggml/src/libggml-cpu.a"
 "${M31_BASE}/ggml/src/libggml-base.a" OpenMP::OpenMP_CXX Threads::Threads m dl rt)
