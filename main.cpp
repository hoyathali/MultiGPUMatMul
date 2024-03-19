#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>

#include <mpi.h>
#include <boost/mpi/datatype.hpp>

#include "mult.cuh"

#define verbose false //For printing matrices row received data

struct genMatrix_A {
    unsigned int nRows;
    unsigned int nCols;
    float counter = 0;

    genMatrix_A(unsigned int nRows, unsigned int nCols, float init=0): nRows(nRows), nCols(nCols), counter(init) {}

    float operator()()
    {
	return 1;
        //return (++counter);
    }
};

struct genMatrix_B {
    unsigned int nRows;
    unsigned int nCols;
    float counter = 0;

    genMatrix_B(unsigned int nRows, unsigned int nCols, float init=0): nRows(nRows), nCols(nCols), counter(init) {}

    float operator()()
    {
	return 1;
	//return (++counter);
    }
};


void matrixMult()
{
    int rank, size;
    float* column = (float*)calloc(BAND_SIZE * K_GLOBAL, sizeof(float)); // Buffer to receive the column
    float* row = (float*)calloc(BAND_SIZE * K_GLOBAL, sizeof(float)); // Buffer to receive the column
    float* res = (float*)calloc(BAND_SIZE * BAND_SIZE, sizeof(float)); // Buffer to receive the column
    
    float *d_column = nullptr, *d_row = nullptr, *d_res = nullptr;
    gpuErrchk( cudaMalloc((void**)&d_column, BAND_SIZE * K_GLOBAL * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_row, BAND_SIZE * K_GLOBAL * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_res, BAND_SIZE * BAND_SIZE * sizeof(float)) );
    gpuErrchk( cudaMemset(d_column, 0, BAND_SIZE * K_GLOBAL * sizeof(float)) );
    gpuErrchk( cudaMemset(d_row, 0, BAND_SIZE * K_GLOBAL * sizeof(float)) );

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> matrix_A;
    std::vector<float> matrix_B;
    std::vector<float> matrix_C;

    matrix_A.resize(M_GLOBAL*K_GLOBAL);
    matrix_B.resize(K_GLOBAL*N_GLOBAL);
    matrix_C.resize(M_GLOBAL*N_GLOBAL);

    // std::generate(matrix_A.begin(), matrix_A.end(), [n = 0] () mutable { return n++; });
    std::generate(matrix_A.begin(), matrix_A.end(), genMatrix_A(M_GLOBAL, K_GLOBAL));
    std::generate(matrix_B.begin(), matrix_B.end(), genMatrix_B(K_GLOBAL, N_GLOBAL));

    
    // Process 0 prints the original matrix_B
    if (rank == 0) {
	
        if(verbose){
        
        std::cout<<"Original matrix_A:"<<std::endl;
        for (int i = 0; i < M_GLOBAL; i++) {
            for (int j = 0; j < K_GLOBAL; j++)
                std::cout<<matrix_A[i*K_GLOBAL + j] << "\t";
            std::cout<<std::endl;
        }
        std::cout<<std::endl;

        std::cout<<"Original matrix_B:"<<std::endl;;
        for (int i = 0; i < K_GLOBAL; i++) {
            for (int j = 0; j < N_GLOBAL; j++)
                std::cout<<matrix_B[i*N_GLOBAL + j] << "\t";
            std::cout<<std::endl;
        }
     }
               
    std::cout<<std::endl;
    std::cout<<"Matrix A size: "<<M_GLOBAL<<" * "<<K_GLOBAL<<std::endl;
    std::cout<<"Matrix B size: "<<K_GLOBAL<<" * "<<N_GLOBAL<<std::endl;
    std::cout<<"Band size: " << BAND_SIZE<<std::endl;   
     
    }
        

    // Define the datatype for a column
    MPI_Datatype col, coltype;
    MPI_Type_vector(K_GLOBAL, BAND_SIZE, N_GLOBAL, boost::mpi::get_mpi_datatype<float>(), &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, BAND_SIZE*sizeof(float), &coltype);
    MPI_Type_commit(&coltype);

    
    // Define the datatype for a column
    MPI_Datatype C_col, C_coltype;
    MPI_Type_vector(BAND_SIZE, BAND_SIZE, N_GLOBAL, boost::mpi::get_mpi_datatype<float>(), &C_col);
    MPI_Type_commit(&C_col);
    MPI_Type_create_resized(C_col, 0, BAND_SIZE*sizeof(float), &C_coltype);
    MPI_Type_commit(&C_coltype);

    bool forward=true;
    bool switched=false;
    int r=0;

    for(int c=0; (c+rank)*BAND_SIZE < N_GLOBAL; c+=size)
    {
	// Scatter the columns of the matrix_B
	MPI_Scatter(matrix_B.data() + (c+rank)*BAND_SIZE, 1, coltype, column, BAND_SIZE*K_GLOBAL, boost::mpi::get_mpi_datatype<float>(), 0, MPI_COMM_WORLD);
	gpuErrchk( cudaMemcpy(d_column, column, BAND_SIZE * K_GLOBAL * sizeof(float), cudaMemcpyHostToDevice) );

	for(; ; forward ? r++:r--)
	{
	    // Broadcast the rows of the matrix_A
	    if(!switched)
	    {
		if (rank == 0)
		{
		    //cudaMemcpy(d_row, matrix_A.data() + r * K_GLOBAL * BAND_SIZE, BAND_SIZE * K_GLOBAL * sizeof(float), cudaMemcpyHostToDevice);
		    memcpy(row, matrix_A.data() + r * K_GLOBAL * BAND_SIZE, BAND_SIZE * K_GLOBAL * sizeof(float));
		}
		MPI_Bcast(row, BAND_SIZE * K_GLOBAL, boost::mpi::get_mpi_datatype<float>(), 0, MPI_COMM_WORLD);
		gpuErrchk( cudaMemcpy(d_row, row, BAND_SIZE * K_GLOBAL * sizeof(float), cudaMemcpyHostToDevice) );
	    }
	    switched=false;
		
	    computeMM(d_row, d_column, d_res , BAND_SIZE, K_GLOBAL, BAND_SIZE);


	    gpuErrchk( cudaMemcpy(res, d_res, BAND_SIZE * BAND_SIZE * sizeof(float), cudaMemcpyDeviceToHost) );
	    MPI_Gather(res, BAND_SIZE*BAND_SIZE, boost::mpi::get_mpi_datatype<float>(), matrix_C.data() + r * N_GLOBAL * BAND_SIZE + c * BAND_SIZE, 1, C_coltype, 0, MPI_COMM_WORLD);
        
        
	    if(verbose)
	    {
		// Each process prints the received column
		std::cout<<"Process "<<rank<<" received row band: ";
		for (int i = 0; i < BAND_SIZE * K_GLOBAL; i++)
		{
		    float temp;
		    cudaMemcpy(&temp, d_row+i, sizeof(float), cudaMemcpyDeviceToHost);
		    std::cout<<temp<<" ";
		}
		std::cout<<std::endl;

		// Each process prints the received column
		std::cout<<"Process "<<rank<<" received column band: ";
		for (int i = 0; i < BAND_SIZE * K_GLOBAL; i++)
		{
		    float temp;
		    cudaMemcpy(&temp, d_column+i, sizeof(float), cudaMemcpyDeviceToHost);
		    std::cout<<temp<<" ";
		}
		std::cout<<std::endl;

		// Each process prints the resultant matrix
		std::cout<<"Process "<<rank<<" computed: "<<r<<" "<<c<<" ";
		for (int i = 0; i < BAND_SIZE * BAND_SIZE; i++)
		{
		    float temp;
		    cudaMemcpy(&temp, d_res+i, sizeof(float), cudaMemcpyDeviceToHost);
		    std::cout<<temp<<" ";
		}
		std::cout<<std::endl;
	    }

	    //Handing iterator logic to benefit from one overlap in every iteration
	    if(r==0 && !forward)
	    {
		forward=true;
		switched=true;
		break;
	    }
	    if(r==(M_GLOBAL/BAND_SIZE) -1 && forward)
	    {
		forward=false;
		switched=true;
		break;
	    }
	}
    }

    // Process 0 prints the original matrix_B
    if (rank == 0 && false) {
        
    // Open a file in write mode.
     std::ofstream outFile("mpi_matrix_output.txt");
      if(verbose){
         std::cout<<"Computed matrix_C:"<<std::endl;
          }
        for (int i = 0; i < M_GLOBAL; i++) {
            for (int j = 0; j < N_GLOBAL; j++){
               if(verbose){
                std::cout<<matrix_C[i*N_GLOBAL + j] << "\t";
               }
                outFile<<matrix_C[i*N_GLOBAL + j] << "\t";
        }      
          outFile<<"\n";
         if(verbose){
          std::cout<<std::endl;
         }
    }
      std::cout<<std::endl<<"Matrix Multiplication Completed!"<<std::endl;
      outFile.close();

    }

    MPI_Type_free(&coltype);
    MPI_Type_free(&col);

    gpuErrchk( cudaFree(d_column) );
    gpuErrchk( cudaFree(d_row) );
    gpuErrchk( cudaFree(d_res) );
    free(column);
    free(row);
    free(res);
}

int main(int argc, char *argv[]) {
    int dev=0;
    
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM8X) architecture or higher.
    if (deviceProp.major < 8) {
        printf("tf32TensorCoreGemm requires requires SM 8.0 or higher to use Tensor Cores.  Exiting...\n");
        exit(1);
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    static_assert( M_GLOBAL % BAND_SIZE == 0 );
    static_assert( N_GLOBAL % BAND_SIZE == 0 );
    if (BAND_SIZE*size > K_GLOBAL || K_GLOBAL%size != 0) {
        if (rank == 0)
            printf("Prereq issue.\n");
        MPI_Finalize();
        return 1;
    }

    matrixMult();

    MPI_Finalize();
    return 0;
}
