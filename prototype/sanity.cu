
// sanity.cu
#include <cstdio>
#include <cuda_runtime.h>
int main(){
  cudaError_t e = cudaFree(0);  // forces context creation
  if (e != cudaSuccess) { printf("cudaFree(0): %s\n", cudaGetErrorString(e)); return 1; }
  int n=0; cudaGetDeviceCount(&n); printf("devices: %d\n", n);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("name=%s, cc=%d.%d\n", p.name, p.major, p.minor);
  return 0;
}

