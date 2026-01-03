#include "cuda8_context.h"
#include "cuda8_kernels.h"
#include <cublas_v2.h>
#include "ggml-backend-impl.h"

// GEMM wrapper using cuBLAS
void ggml_cuda8_gemm(cuda8_context& ctx, const float* A, const float* B, float* C,
                     int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha, A, M, B, K, &beta, C, M);
}

// Forward declaration of device registration
static ggml_backend_dev_t cuda8_device_init();

// Backend registration function
ggml_backend_reg_t ggml_backend_cuda8_reg() {
    static struct ggml_backend_reg reg = {
        .name = "CUDA8",
        .api_version = GGML_BACKEND_API_VERSION,
        .get_device_count = []() -> size_t { return 1; }, // Single GPU for now
        .get_device = size_t index -> ggml_backend_dev_t {
            return index == 0 ? cuda8_device_init() : nullptr;
        },
        .get_proc_address = nullptr, // Optional for extra functions
    };
    return &reg;
}

// Minimal device registration
static ggml_backend_dev_t cuda8_device_init() {
    static struct ggml_backend_dev dev = {
        .name = "CUDA8 Device",
        .description = "Legacy CUDA8 backend for Fermi GPUs",
        .type = GGML_BACKEND_DEVICE_TYPE_GPU,
        .device_id = "cuda8-legacy",
        .caps = { .async = false, .host_buffer = true, .buffer_from_host_ptr = true, .events = false },
        .init = const char *params -> ggml_backend_t {
            // Initialize CUDA8 context
            static cuda8_context ctx;
            ctx.init();
            return (ggml_backend_t)&ctx; // Cast to generic backend type
        },
        .buffer_type = nullptr, // Implement if needed
        .host_buffer_type = nullptr,
    };
    return &dev;
}
