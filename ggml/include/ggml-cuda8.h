#pragma once

#include "ggml-backend-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register the CUDA8 backend with ggml
ggml_backend_reg_t ggml_backend_cuda8_reg();

#ifdef __cplusplus
}
#endif
