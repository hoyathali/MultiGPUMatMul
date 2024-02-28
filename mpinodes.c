#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the processor name
    MPI_Get_processor_name(processor_name, &name_len);

    // Print "Hello, World!" along with the rank of the process and the processor name
    printf("Hello, World! I am process %d of %d on %s\n", rank, size, processor_name);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
