#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <mpi.h>
#include <boost/mpi/datatype.hpp>
#include "mult.cuh"
#include <cuda_runtime.h>

#define BAND_SIZE 2
#define M 4  // Size of the matrix_B
#define K 4  // Size of the matrix_B
#define N 4  // Size of the matrix_B

struct genMatrix_A {
    unsigned int nRows;
    unsigned int nCols;
    float counter = 0;

    genMatrix_A(unsigned int nRows, unsigned int nCols, float init=0): nRows(nRows), nCols(nCols), counter(init) {}

    float operator()()
    {
	return (++counter);
    }
};

struct genMatrix_B {
    unsigned int nRows;
    unsigned int nCols;
    float counter = 0;

    genMatrix_B(unsigned int nRows, unsigned int nCols, float init=0): nRows(nRows), nCols(nCols), counter(init) {}

    float operator()()
    {
	//return 1;
	return (++counter);
    }
};


void matrixMult()
{
    int rank, size;
    float* column = (float*)calloc(BAND_SIZE * K, sizeof(float)); // Buffer to receive the column
    float* row = (float*)calloc(BAND_SIZE * K, sizeof(float)); // Buffer to receive the column
    float* res = (float*)calloc(BAND_SIZE * BAND_SIZE, sizeof(float)); // Buffer to receive the column
    
    float *d_column = nullptr, *d_row = nullptr, *d_res = nullptr;
    custom_cudaMalloc((void**)&d_column, BAND_SIZE * K * sizeof(float));
    custom_cudaMalloc((void**)&d_row, BAND_SIZE * K * sizeof(float));
    custom_cudaMalloc((void**)&d_res, BAND_SIZE * BAND_SIZE * sizeof(float));

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> matrix_A(M*K);
    std::vector<float> matrix_B(K*N);
    std::vector<float> matrix_C(M*N);

    // std::generate(matrix_A.begin(), matrix_A.end(), [n = 0] () mutable { return n++; });
    std::generate(matrix_A.begin(), matrix_A.end(), genMatrix_A(M, K));
    std::generate(matrix_B.begin(), matrix_B.end(), genMatrix_B(K, N));

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
    MPI_Type_vector(K, BAND_SIZE, N, boost::mpi::get_mpi_datatype<float>(), &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, BAND_SIZE*sizeof(float), &coltype);
    MPI_Type_commit(&coltype);

	bool forward=true;
	bool switched=false;
	int r=0;
    
    // Define the datatype for a column
    MPI_Datatype C_col, C_coltype;
    MPI_Type_vector(M, BAND_SIZE, N, boost::mpi::get_mpi_datatype<float>(), &C_col);
    MPI_Type_commit(&C_col);
    MPI_Type_create_resized(C_col, 0, BAND_SIZE*sizeof(float), &C_coltype);
    MPI_Type_commit(&C_coltype);

    for(int c=0; (c+rank)*BAND_SIZE < N; c+=size)
    {
	// Scatter the columns of the matrix_B
	MPI_Scatter(matrix_B.data() + (c+rank)*BAND_SIZE, 1, coltype, d_column, BAND_SIZE*K, boost::mpi::get_mpi_datatype<float>(), 0, MPI_COMM_WORLD);

	for(; ; forward ? r++:r--)
	{
	    if (rank == 0)
	    {
		custom_cudaMemcpy_h2d(d_row, matrix_A.data() + r * K * BAND_SIZE, BAND_SIZE * K * sizeof(float));
	    }

	    // Broadcast the rows of the matrix_A
	    if(!switched)
	    {
		MPI_Bcast(d_row, BAND_SIZE * K, boost::mpi::get_mpi_datatype<float>(), 0, MPI_COMM_WORLD);
	    }
	    switched=false;
		
	    computeMM(d_row, d_column, d_res , BAND_SIZE, K, BAND_SIZE);


	    MPI_Gather(d_res, BAND_SIZE*BAND_SIZE, boost::mpi::get_mpi_datatype<float>(), matrix_C.data() + r * N * BAND_SIZE + c * BAND_SIZE, 1, C_coltype, 0, MPI_COMM_WORLD);

	    // Each process prints the received column
	    std::cout<<"Process "<<rank<<" received row band: ";
	     for (int i = 0; i < BAND_SIZE * K; i++)
	     {
		float temp;
		custom_cudaMemcpy_d2h(&temp, d_row+i, sizeof(float));
		std::cout<<temp<<" ";
	     }
	    std::cout<<std::endl;

	    // Each process prints the received column
	    std::cout<<"Process "<<rank<<" received column band: ";
	    for (int i = 0; i < BAND_SIZE * K; i++)
	     {
		float temp;
		custom_cudaMemcpy_d2h(&temp, d_column+i, sizeof(float));
		std::cout<<temp<<" ";
	     }
	    std::cout<<std::endl;

	    // Each process prints the resultant matrix
	    std::cout<<"Process "<<rank<<" computed: ";
	    for (int i = 0; i < BAND_SIZE * BAND_SIZE; i++)
	    {
		float temp;
		custom_cudaMemcpy_d2h(&temp, d_res+i, sizeof(float));
		cudaMemcpy(&temp, d_res+i, sizeof(float), cudaMemcpyDeviceToHost);
		std::cout<<temp<<" ";
	    }
	    std::cout<<std::endl;

		if(r==0 && !forward)
		{
			forward=true;
			switched=true;
			break;

		}
		if(r==(M/BAND_SIZE) -1 && forward)
		{
			forward=false;
			switched=true;
			break;
		}

	}
    }

    // Process 0 prints the original matrix_B
    if (rank == 0) {
	std::cout<<"Computed matrix_C:"<<std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
                std::cout<<matrix_C[i*K + j] << "\t";
            std::cout<<std::endl;
        }
	std::cout<<std::endl;
    }
    std::cout<<std::endl;

    MPI_Type_free(&coltype);
    MPI_Type_free(&col);

    custom_cudaFree(d_column);
    custom_cudaFree(d_row);
    custom_cudaFree(d_res);
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

    matrixMult();

    MPI_Finalize();
    return 0;
}
