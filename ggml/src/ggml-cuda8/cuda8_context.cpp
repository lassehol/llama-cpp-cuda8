
#include "cuda8_context.h"
#include <iostream>

cuda8_context::cuda8_context() : handle(nullptr), device_id(0) {}

void cuda8_context::init() {
    cudaSetDevice(device_id);
    cublasCreate(&handle);
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "[CUDA8] Initialized on device " << device_id
              << " Free: " << free_mem << " Total: " << total_mem << std::endl;
}

void cuda8_context::destroy() {
    if (handle) cublasDestroy(handle);
}

