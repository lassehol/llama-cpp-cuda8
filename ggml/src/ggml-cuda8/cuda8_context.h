
#ifndef CUDA8_CONTEXT_H
#define CUDA8_CONTEXT_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct cuda8_context {
    cublasHandle_t handle;
    int device_id;
    size_t free_mem;
    size_t total_mem;

    cuda8_context();
    ~cuda8_context();

    void init();
    void destroy();
};

#endif // CUDA8_CONTEXT_H

