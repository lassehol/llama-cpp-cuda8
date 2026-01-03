cmake_minimum_required(VERSION 3.5)
project(ggml-cuda8 LANGUAGES C CXX CUDA)

add_library(ggml-cuda8 STATIC
    cuda8_context.cpp
    cuda8_kernels.cu
    cuda8_backend.cpp
)

target_include_directories(ggml-cuda8 PRIVATE ..)
target_link_libraries(ggml-cuda8 PRIVATE ggml-base cublas cuda)
set_target_properties(ggml-cuda8 PROPERTIES CUDA_ARCHITECTURES "21")
