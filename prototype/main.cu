// nvcc -std=c++11 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_20,code=compute_20 \
//      -L/usr/local/cuda-8.0/lib64 -lcublas -lcudart -Xlinker -rpath=/usr/local/cuda-8.0/lib64 \
//      main.cu -o prototype

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>

// #define GEMM_DEBUG  // uncomment to print SGEMM argument sanity

// ------------------------------
// Error checking helpers
// ------------------------------
#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t _s = (call); \
  if (_s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, int(_s)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

// ------------------------------
// Minimal bias-add kernel (Y += b per column)
// Y is MxN column-major; b is length N
// ------------------------------
__global__ void add_bias_kernel(float* __restrict__ Y,
                                const float* __restrict__ b,
                                int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  // column-major: each column is contiguous in memory stepping by M
  for (int row = 0; row < M; ++row) {
    Y[col * M + row] += b[col];
  }
}

// ------------------------------
// (Optional) reference quantized matmul stub
// ------------------------------
__global__ void qmm_kernel(const float* __restrict__ X,          // MxK, fp32
                           const unsigned char* __restrict__ Wq, // packed int4/8
                           const float* __restrict__ scales,     // per-block scales
                           float* __restrict__ Y,                // MxN, fp32
                           int M, int N, int K,
                           int blockSize, int bitsPerWeight) {
  // Stub for future work; not used here.
}

// ------------------------------
// Utility: initialize host data
// ------------------------------
static void init_random(std::vector<float>& v, float scale = 0.01f) {
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = scale * float(std::rand()) / float(RAND_MAX);
  }
}

// ------------------------------
// Main: triple-buffer pipeline
// ------------------------------
int main(int argc, char** argv) {
  // Problem sizes
  const int M = 1024; // rows of X, Y
  const int K = 2048; // cols of X, rows of W
  const int N = 1024; // cols of W, Y
  const int NUM_BUFFERS = 3;

  // Device
  CHECK_CUDA(cudaSetDevice(0));

  // Streams
  cudaStream_t streamH2D, streamGEMM, streamBias, streamD2H;
  CHECK_CUDA(cudaStreamCreate(&streamH2D));
  CHECK_CUDA(cudaStreamCreate(&streamGEMM));
  CHECK_CUDA(cudaStreamCreate(&streamBias));
  CHECK_CUDA(cudaStreamCreate(&streamD2H));

  // cuBLAS
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, streamGEMM)); // bind GEMM stream

  // Host pinned buffers (triple-buffered X & Y)
  size_t bytesX = size_t(M) * size_t(K) * sizeof(float);
  size_t bytesY = size_t(M) * size_t(N) * sizeof(float);
  float* hX[NUM_BUFFERS] = {nullptr};
  float* hY[NUM_BUFFERS] = {nullptr};
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    CHECK_CUDA(cudaHostAlloc((void**)&hX[i], bytesX, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&hY[i], bytesY, cudaHostAllocDefault));
  }

  // Host bias
  std::vector<float> hb(N);
  init_random(hb);

  // Device buffers (X and Y per-buffer)
  float *dX[NUM_BUFFERS] = {nullptr}, *dY[NUM_BUFFERS] = {nullptr};
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    CHECK_CUDA(cudaMalloc((void**)&dX[i], bytesX));
    CHECK_CUDA(cudaMalloc((void**)&dY[i], bytesY));
  }

  // Device weights W (KxN), resident
  size_t bytesW = size_t(K) * size_t(N) * sizeof(float);
  float* dW = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dW, bytesW));

  // Initialize & upload W
  std::vector<float> hW(K * N);
  init_random(hW);
  CHECK_CUDA(cudaMemcpyAsync(dW, hW.data(), bytesW, cudaMemcpyHostToDevice, streamH2D));

  // Device bias
  float* db = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&db, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpyAsync(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice, streamH2D));

  // Events per buffer
  cudaEvent_t evXReady[NUM_BUFFERS], evGemmDone[NUM_BUFFERS], evBiasDone[NUM_BUFFERS], evYReady[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    CHECK_CUDA(cudaEventCreateWithFlags(&evXReady[i], cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&evGemmDone[i], cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&evBiasDone[i], cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&evYReady[i], cudaEventDisableTiming));
  }

  // Warm up host X buffers
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    std::vector<float> tmpX(M * K);
    init_random(tmpX);
    std::memcpy(hX[i], tmpX.data(), bytesX);
  }

  // Pipeline iterations
  const int iters = 10;
  for (int it = 0; it < iters; ++it) {
    const int buf = it % NUM_BUFFERS;

    // Stage 1: H2D for X -> dX[buf]
    CHECK_CUDA(cudaMemcpyAsync(dX[buf], hX[buf], bytesX, cudaMemcpyHostToDevice, streamH2D));
    CHECK_CUDA(cudaEventRecord(evXReady[buf], streamH2D));

    // Stage 2: GEMM  Y = X(MxK) * W(KxN)
    CHECK_CUDA(cudaStreamWaitEvent(streamGEMM, evXReady[buf], 0));

    // Clear Y if using beta=0 (not strictly necessary, but harmless)
    CHECK_CUDA(cudaMemsetAsync(dY[buf], 0, bytesY, streamGEMM));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major:
    //  op(A) = X (MxK), op(B) = W (KxN), C = Y (MxN)
    //  m = M, n = N, k = K
    //  lda = M, ldb = K, ldc = M
#ifdef GEMM_DEBUG
    {
      int m = M, n = N, k = K;
      int lda = M, ldb = K, ldc = M;
      printf("SGEMM: transA=N transB=N m=%d n=%d k=%d\n", m, n, k);
      printf("SGEMM: lda=%d ldb=%d ldc=%d (need: lda>=%d, ldb>=%d, ldc>=%d)\n",
             lda, ldb, ldc, m, k, m);
    }
#endif

    CHECK_CUBLAS(
      cublasSgemm(handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  /* m */ M, /* n */ N, /* k */ K,
                  &alpha,
                  /* A */ dX[buf], /* lda */ M,
                  /* B */ dW,      /* ldb */ K,
                  &beta,
                  /* C */ dY[buf], /* ldc */ M)
    );

    CHECK_CUDA(cudaEventRecord(evGemmDone[buf], streamGEMM));

    // Stage 3: add bias on streamBias (wait GEMM)
    CHECK_CUDA(cudaStreamWaitEvent(streamBias, evGemmDone[buf], 0));
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, streamBias>>>(dY[buf], db, M, N);
    CHECK_CUDA(cudaEventRecord(evBiasDone[buf], streamBias));

    // Stage 4: D2H for Y
    CHECK_CUDA(cudaStreamWaitEvent(streamD2H, evBiasDone[buf], 0));
    CHECK_CUDA(cudaMemcpyAsync(hY[buf], dY[buf], bytesY, cudaMemcpyDeviceToHost, streamD2H));
    CHECK_CUDA(cudaEventRecord(evYReady[buf], streamD2H));

    // CPU can consume hY[buf] after evYReady[buf] completes
    CHECK_CUDA(cudaEventSynchronize(evYReady[buf]));

    // Quick checksum sample
    float checksum = 0.0f;
    for (size_t i = 0; i < size_t(M)*size_t(N); i += (M * N / 16)) checksum += hY[buf][i];
    printf("[iter %d] Y checksum (sampled): %f\n", it, checksum);

    // Prepare next hX[buf]
    std::vector<float> tmpX(M * K);
    init_random(tmpX);
    std::memcpy(hX[buf], tmpX.data(), bytesX);
  }

  // Teardown
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(streamH2D));
  CHECK_CUDA(cudaStreamDestroy(streamGEMM));
  CHECK_CUDA(cudaStreamDestroy(streamBias));
  CHECK_CUDA(cudaStreamDestroy(streamD2H));

  for (int i = 0; i < NUM_BUFFERS; ++i) {
    CHECK_CUDA(cudaEventDestroy(evXReady[i]));
    CHECK_CUDA(cudaEventDestroy(evGemmDone[i]));
    CHECK_CUDA(cudaEventDestroy(evBiasDone[i]));
    CHECK_CUDA(cudaEventDestroy(evYReady[i]));
  }
  CHECK_CUDA(cudaFree(dW));
  CHECK_CUDA(cudaFree(db));
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    CHECK_CUDA(cudaFree(dX[i]));
    CHECK_CUDA(cudaFree(dY[i]));
    CHECK_CUDA(cudaFreeHost(hX[i]));
    CHECK_CUDA(cudaFreeHost(hY[i]));
  }
  printf("Done.\n");
  return 0;
}

