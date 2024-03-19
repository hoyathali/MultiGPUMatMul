#include <cuda_runtime.h>

// MMA matrix tile dimensions.
#define M_GLOBAL 1024		    // Min 16
#define N_GLOBAL 1024		    // Min 16
#define K_GLOBAL 4096		    // Min 8
#define BAND_SIZE 256		    // Min 16


void call_cuda();

void computeMM(const float *A, const float *B, float *C, int m, int k, int n);

void custom_cudaMalloc(void** devPtr, size_t size);

void custom_cudaFree ( void* devPtr );

void custom_cudaMemcpy_d2h ( void* dst, const void* src, size_t count);
void custom_cudaMemcpy_h2d ( void* dst, const void* src, size_t count);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern cudaDeviceProp deviceProp;
