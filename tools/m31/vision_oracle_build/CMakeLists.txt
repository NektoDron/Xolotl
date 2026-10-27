cmake_minimum_required(VERSION 3.24)
project(m31_vision_oracle LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
# Reuse the audited PR SHA check and static library dependency ordering.
add_subdirectory(../oracle text_oracle EXCLUDE_FROM_ALL)
add_executable(m31_vision_oracle ../vision_oracle.cpp)
target_include_directories(m31_vision_oracle PRIVATE ..
  "${M31_LLAMA_SOURCE}/include" "${M31_LLAMA_SOURCE}/ggml/include"
  "${M31_LLAMA_SOURCE}/src" "${M31_LLAMA_SOURCE}/vendor/nlohmann")
target_link_libraries(m31_vision_oracle PRIVATE m31_ref)
