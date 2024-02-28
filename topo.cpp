#include "mpi.h"
#include <hwloc.h>
#include "main.cuh"

#include<iostream>

int main(int argc, char* argv[])
{
        int rank, size;
        cpu_set_t mask;
        long num;
        int proc_num(long num);

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

	call_cuda();

        printf ("Hello World, I am %d and pid: %d coreid:%d\n",rank,getpid());

        MPI_Finalize();
        return 0;
}
