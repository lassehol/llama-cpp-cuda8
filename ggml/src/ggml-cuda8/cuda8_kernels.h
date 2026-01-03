
#ifndef CUDA8_KERNELS_H
#define CUDA8_KERNELS_H

#include <cuda_runtime.h>

// Basic kernels
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n);
__global__ void vec_mul_kernel(const float* a, const float* b, float* c, int n);

// Softmax kernel (simplified)
__global__ void softmax_kernel(float* data, int n);

#endif // CUDA8_KERNELS_H

