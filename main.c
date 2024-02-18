#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print rank and size
    printf("Hello from process %d of %d\n", rank, size);

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}
