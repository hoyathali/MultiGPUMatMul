#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <mpi.h>
#include <boost/mpi/datatype.hpp>
#include "mult.cuh"

#define BAND_SIZE 2
#define M 2  // Size of the matrix_B
#define K 2  // Size of the matrix_B
#define N 2  // Size of the matrix_B

template <typename T>
struct genMatrix {
    unsigned int nRows;
    unsigned int nCols;
    T counter = 0;

    genMatrix(unsigned int nRows, unsigned int nCols, T init=0): nRows(nRows), nCols(nCols), counter(init) {}

    T operator()()
    {
	return (++counter);
    }
};


template <typename T>
void matrixMult()
{
    int rank, size;
    T* column = (T*)calloc(BAND_SIZE * K, sizeof(T)); // Buffer to receive the column
    T* row = (T*)calloc(BAND_SIZE * K, sizeof(T)); // Buffer to receive the column
    T* res = (T*)calloc(BAND_SIZE * BAND_SIZE, sizeof(T)); // Buffer to receive the column
    
    T *d_column = nullptr, *d_row = nullptr, *d_res = nullptr;
    cudaMalloc(&d_column, BAND_SIZE * K * sizeof(T));
    cudaMalloc(&d_row, BAND_SIZE * K * sizeof(T));
    cudaMalloc(&d_res, BAND_SIZE * BAND_SIZE * sizeof(T));

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<T> matrix_A(M*K);
    std::vector<T> matrix_B(K*N);

    // std::generate(matrix_A.begin(), matrix_A.end(), [n = 0] () mutable { return n++; });
    std::generate(matrix_A.begin(), matrix_A.end(), genMatrix<T>(M, K));
    std::generate(matrix_B.begin(), matrix_B.end(), genMatrix<T>(K, N, 100));

    // Process 0 prints the original matrix_B
    if (rank == 0) {
	std::cout<<"Original matrix_A:"<<std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++)
                std::cout<<matrix_A[i*K + j] << "\t";
            std::cout<<std::endl;
        }
	std::cout<<std::endl;

	std::cout<<"Original matrix_B:"<<std::endl;;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++)
                std::cout<<matrix_B[i*N + j] << "\t";
            std::cout<<std::endl;
        }
    }
    std::cout<<std::endl;

    // Define the datatype for a column
    MPI_Datatype col, coltype;
    MPI_Type_vector(K, BAND_SIZE, N, boost::mpi::get_mpi_datatype<T>(), &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, BAND_SIZE*sizeof(T), &coltype);
    MPI_Type_commit(&coltype);

    for(int c=0; (c+rank)*BAND_SIZE < N; c+=size)
    {
	// Scatter the columns of the matrix_B
	MPI_Scatter(matrix_B.data() + (c+rank)*BAND_SIZE, 1, coltype, column, BAND_SIZE*K, boost::mpi::get_mpi_datatype<T>(), 0, MPI_COMM_WORLD);

	for(int r=0; r*BAND_SIZE < M; r++)
	{
	    if (rank == 0)
	    {
		std::memcpy(row, matrix_A.data() + r * K * BAND_SIZE, BAND_SIZE * K * sizeof(T));
	    }

	    // Broadcast the rows of the matrix_A
	    MPI_Bcast(row, BAND_SIZE * K, boost::mpi::get_mpi_datatype<T>(), 0, MPI_COMM_WORLD);
		
	    computeMM<float>(d_row, d_column, d_res , BAND_SIZE, K, BAND_SIZE);

	    // Each process prints the received column
	    std::cout<<"Process "<<rank<<" received row band: ";
	     for (int i = 0; i < BAND_SIZE * K; i++)
		std::cout<<row[i]<<" ";
	    std::cout<<std::endl;

	    // Each process prints the received column
	    std::cout<<"Process "<<rank<<" received column band: ";
	    for (int i = 0; i < BAND_SIZE * K; i++)
		std::cout<<column[i]<<" ";
	    std::cout<<std::endl;

	    // Each process prints the resultant matrix
	    std::cout<<"Process "<<rank<<" computed: ";
	    for (int i = 0; i < BAND_SIZE * K; i++)
	    {
		T temp;
		cudaMemcpy(&temp, d_res+i, sizeof(T), cudaMemcpyDeviceToHost);
		std::cout<<column[i]<<" ";
	    }
	    std::cout<<std::endl;
	}
    }


    MPI_Type_free(&coltype);
    MPI_Type_free(&col);

    cudaFree(d_column);
    cudaFree(d_row);
    cudaFree(d_res);
    free(column);
    free(row);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (BAND_SIZE*size > K || M % BAND_SIZE != 0 || N % BAND_SIZE != 0 || K%size != 0) {
        if (rank == 0)
            printf("Prereq issue.\n");
        MPI_Finalize();
        return 1;
    }

    matrixMult<float>();

    MPI_Finalize();
    return 0;
}

