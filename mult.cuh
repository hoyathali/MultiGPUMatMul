void call_cuda();

void computeMM(const float *A, const float *B, float *C, int m, int k, int n);

void custom_cudaMalloc(void** devPtr, size_t size);

void custom_cudaFree ( void* devPtr );

void custom_cudaMemcpy_d2h ( void* dst, const void* src, size_t count);
void custom_cudaMemcpy_h2d ( void* dst, const void* src, size_t count);
