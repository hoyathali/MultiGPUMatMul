#!/bin/bash
#SBATCH --job-name=matrixmultiplication   # Job name
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of MPI tasks per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --time=00:30:00                   # Walltime (hh:mm:ss)
#SBATCH --output=matrixmultiplication.out # Standard output log file
#SBATCH --error=matrixmultiplication.err  # Standard error log file
#SBATCH --priority=TOP                    # Priority set to TOP

# Change directory to where the MPI executable is located
cd /MM/MultiGPUMatMul/

mpicc mpinodes.c -o mpinodes

# Run MPI program
mpirun -np 2 ./mpinodes
