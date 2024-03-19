# Multi-GPU Matrix Multiplication (MPI-CUDA)

## Abstract
This project leverages multiple Graphics Processing Units (GPUs) to accelerate matrix multiplication across distributed networks. By dividing large matrices into smaller sections (Bands) and distributing these sections across GPUs, we achieve significant computational speed improvements. This approach combines the Compute Unified Device Architecture (CUDA) for parallel GPU computing and the Message Passing Interface (MPI) for effective data coordination across the network.

## Introduction
Matrix multiplication is fundamental in numerous scientific and engineering applications, from machine learning to physics simulations. Traditional computational paradigms face limitations due to the finite memory of single GPUs. This project introduces a method that utilizes distributed computing and multiple GPUs, segmenting matrices into "Bands" for efficient parallel processing. It showcases the power of combining CUDA and MPI for scalable and high-performance matrix multiplication.

## Computational Environment

### Hardware Specifications
- **GPU Model:** NVIDIA RTX A2000
- **Memory:** 6GB GDDR6

### Software Environment
- **MPI Version:** Open MPI v4.0.3
- **CUDA Version:** 12.2
- **Driver Version:** NVIDIA-SMI 535.54.03

## Methodology
The approach involves segmenting matrices into smaller bands, optimizing memory use, and enhancing computational speed through distributed processing.

### Matrix Segmentation and Distribution
- **Dynamic Band Calculation:** Sizes of matrix bands are calculated based on input matrix dimensions and GPU memory, optimizing resource use.
- **Segmentation:** Matrices are segmented into bands (row bands from one matrix and column bands from another), facilitating distributed computation.

### Efficient Data Transfer and MPI Operations
Data transfer is optimized using MPI operations:
- **MPI Scatter:** Used to distribute column bands among GPUs, ensuring each GPU receives only the data it needs, minimizing data transfers.
- **MPI Broadcast:** Row bands are sent to all GPUs, enabling simultaneous multiplication with their respective column bands, enhancing parallelism.
- **MPI Gather:** Collects partial results from all GPUs, assembling them into the final matrix product, ensuring efficient result compilation.

### Parallel Computation across GPUs
The computation strategy leverages both MPI and CUDA for efficient, distributed matrix multiplication:
- **Initial Distribution:** Column bands are initially distributed to GPUs. Row bands are iteratively broadcasted to all GPUs for computation.
- **Computational Looping:** Each GPU multiplies its column band with every row band. To reduce communication overhead, an iterative and reverse looping strategy is applied, reusing data bands efficiently.

## Optimization Strategies
- **Minimizing Data Transfers:** Strategic data distribution and reuse minimize necessary data transfers.
- **Leveraging Parallel Computation:** The project exploits GPUs' parallel computation capabilities to maximize throughput.
- **Reducing Communication Overhead:** By optimizing MPI operations and computational strategies, communication overhead is significantly reduced.


## How to Use
make clean
salloc -n {no of nodes}
mpirun ./main

## Contributors
- Rebin Silva (PhD, CS) - rvala009@ucr.edu
- Hoyath Ali (MS, CS) - Hsing093@ucr.edu
- Danhua Zhao (MS, CS) - dzhao051@ucr.edu

