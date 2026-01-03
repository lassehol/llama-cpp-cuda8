
#include "cuda8_context.h"
#include "cuda8_kernels.h"
#include <cublas_v2.h>

void ggml_cuda8_gemm(cuda8_context& ctx, const float* A, const float* B, float* C,
                     int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha, A, M, B, K, &beta, C, M);
}

