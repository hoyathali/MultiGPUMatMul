#include<iostream>

__global__ void hello()
{
    printf("Hi from GPU %d %d\n", threadIdx.x, blockIdx.x);
}

void call_cuda()
{
	hello<<<2,32>>>();
	cudaDeviceSynchronize();
}
	

