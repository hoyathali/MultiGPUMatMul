#include<iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda/pipeline>

#include "mult.cuh"

#define BLOCK_SIZE 2

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

#define WARP_SIZE 32

// M, N and K represent the wmma tile size
// These values should not be modified
#define M 16
#define N 16
#define K 8

#define M_TILES (BAND_SIZE/M)
#define N_TILES (BAND_SIZE/N)
#define K_TILES (K_GLOBAL/K)

#define C_LAYOUT wmma::mem_row_major

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that is (M = 16) * (K = 8) * 8 * (CHUNK_K = 8)
// * sizeof(float) = 32 Kb each.
// (i.e. two 8x8 arrays of tiles of 16x8 float-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the performance
// would be severely impacted. So we choose to reduce the chunk size in half,
// i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(float))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE (N * N_TILES)

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 8 four-byte "float" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_FLOAT 8

using namespace nvcuda;

cudaDeviceProp deviceProp;

void custom_cudaMalloc(void** devPtr, size_t size)
{
    cudaMallocManaged(devPtr, size);
}

void custom_cudaFree ( void* devPtr )
{
    cudaFree(devPtr);
}

void custom_cudaMemcpy_d2h ( void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

void custom_cudaMemcpy_h2d ( void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

__global__ void hello()
{
    printf("Hi from GPU %d %d\n", threadIdx.x, blockIdx.x);
}

void call_cuda()
{
	hello<<<2,32>>>();
	cudaDeviceSynchronize();
}


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void MatrixMulCUDA(float *C, const float *A,
    const float *B, int wA,
    int wB) {

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}


// Performs an MxNxK tf32 GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16, 16 and 8 respectively. 
//  3) A is row major, B is column major matrix.
// Note: This is a less performant version of the compute_tf32gemm kernel. It is designed for
//       demonstration purposes only to show the CUDA WMMA API use without relying on
//       availability of the shared memory.
__global__ void simple_wmma_tf32gemm(float *a, float *b,  float *d, int m_ld, int n_ld, int k_ld)
{
#if __CUDA_ARCH__ >= 800
   // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
   wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < k_ld; i += K) {
      int aCol = i; 
      int aRow = warpM * M;

      //int bCol = i;
      //int bRow = warpN * N;
      int bCol = warpN * N;
      int bRow = i;

      // Bounds checking
      if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
 #pragma unroll
        for (int t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
        }

 #pragma unroll
        for (int t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
        }
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cCol = warpN * N;
   int cRow = warpM * M;

   if (cRow < m_ld && cCol < n_ld) {
      //wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
   }
#endif
}

__global__ void compute_tf32gemm_async_copy(const float *A, const float *B, float *D)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ float shmem[][CHUNK_K * K + SKEW_FLOAT];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // This pointer is used to access the D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS + (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

    // This pointer is used to stream the D matrix block-wide tile to and from shared memory.
    float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Offset in shared memory from which the B matrix is stored.
    constexpr size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    constexpr int loadStride = 2; // load 4 floats, so left-shift by 2.

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the D matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Initialize the accumulator fragments to 0.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
		wmma::fill_fragment(c[i][j], 0.0f);
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const float *warp_ptr = (warpId < (WARPS_PER_BLOCK/2)) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
                                              (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2);

        constexpr int chunksPerLane = ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2;
        const int laneLoadElem = (laneId % CHUNK_COPY_LINE_LANES) << loadStride;
        const int stridePerLaneCopy = (laneId / CHUNK_COPY_LINE_LANES);
        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            // As for tf32 MMA  M == N we use M for warp 4-7 + shmem_idx_b_off.
            size_t shmem_idx =  (M * (warpId % (WARPS_PER_BLOCK/2)) * 2)  + ((warpId / (WARPS_PER_BLOCK/2)) * shmem_idx_b_off);
            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            const float *lane_ptr = (warp_ptr + tile_k * K + stridePerLaneCopy * K_GLOBAL + laneLoadElem);

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += stridePerLaneCopy;

#pragma unroll
            for(int i = 0; i < chunksPerLane; i++) {
                // Copy 16 bytes at once in each lane.
                pipe.producer_acquire();
                cuda::memcpy_async(&shmem[shmem_idx][laneLoadElem], lane_ptr, shape4, pipe);
                pipe.producer_commit();

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            cuda::pipeline_consumer_wait_prior<0>(pipe);
            __syncthreads();

            // Compute a grid of D matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / BLOCK_ROW_WARPS) * M * BLOCK_ROW_WARPS + (i * M);
                    const float *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_FLOAT);

#pragma unroll
                    for (int t = 0; t < a[i].num_elements; t++) {
                        a[i].x[t] = wmma::__float_to_tf32(a[i].x[t]);
                    }
#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
                            const float *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_FLOAT);
#pragma unroll
                            for (int t = 0; t < b[j].num_elements; t++) {
                                b[j].x[t] =  wmma::__float_to_tf32(b[j].x[t]);
                            }
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            pipe.consumer_release();
            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < N; i++) {
            *((float4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((float4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
#endif
}

void computeMM(const float *A, const float *B, float *C, int m, int k, int n)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y); 
    const int SHMEM_SZ = MAX(sizeof(float) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_FLOAT) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float));
    gpuErrchk( cudaFuncSetAttribute(compute_tf32gemm_async_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ) );
    compute_tf32gemm_async_copy<<<deviceProp.multiProcessorCount*2, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C);
    //MatrixMulCUDA<<<grid, threads>>>(C, A, B, k, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return;
}

