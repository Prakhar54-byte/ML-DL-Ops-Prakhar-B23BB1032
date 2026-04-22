cmake_minimum_required(VERSION 3.14)
project(onnx_infer_cpp)

set(CMAKE_CXX_STANDARD 17)

# Point this to your ONNX Runtime C++ installation path
find_package(OnnxRuntime REQUIRED)

add_executable(infer_bench main.cpp)
target_link_libraries(infer_bench PRIVATE OnnxRuntime::OnnxRuntime)