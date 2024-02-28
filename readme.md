#Multi GPU Matrix Multiplication

##mpinodes.c Hello world program to test all the nodes in the system
Command: salloc mpinodes.c -o mpinodes 
Command: mpiexec -n 2 ./mpinodes

##Alternatively
#Created mm.slurm (To batch the jobs automatically)
Command: sbatch mm.slurm

