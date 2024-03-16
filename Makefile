# Paths and flags for remote system
REMOTE_CUDA_PATH = /usr/local/cuda
REMOTE_NVCC = $(REMOTE_CUDA_PATH)/bin/nvcc
REMOTE_CUDA_INCLUDEPATH = $(REMOTE_CUDA_PATH)/include
REMOTE_CUDA_LIBPATH = $(REMOTE_CUDA_PATH)/lib64
REMOTE_MPICC = mpicxx
REMOTE_CUDA_FLAGS = -I$(REMOTE_CUDA_INCLUDEPATH) -L$(REMOTE_CUDA_LIBPATH) -lcudart
REMOTE_MPI_FLAGS = 
REMOTE_NVCC_FLAGS = 

# Paths and flags for local system
LOCAL_CUDA_PATH = /opt/cuda
LOCAL_NVCC = nvcc
LOCAL_CUDA_LIBPATH = $(LOCAL_CUDA_PATH)/lib
LOCAL_MPICC = mpic++
LOCAL_CUDA_FLAGS = -L$(LOCAL_CUDA_LIBPATH) -lcudart
LOCAL_MPI_FLAGS = -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart

# Output binary
TARGET = main

# Object files
MPI_OBJ = mpi.o
CUDA_OBJ = mult.o

# Source files
MPI_SRC = main.cpp
CUDA_SRC = mult.cu

# Default target for remote system
all: REMOTE_FLAGS $(TARGET)

# Target for local system
local: LOCAL_FLAGS $(TARGET)

# Set flags for remote compilation
REMOTE_FLAGS:
	$(eval NVCC = $(REMOTE_NVCC))
	$(eval MPICC = $(REMOTE_MPICC))
	$(eval CUDA_FLAGS = $(REMOTE_CUDA_FLAGS))
	$(eval CUDA_LIBPATH = $(REMOTE_CUDA_LIBPATH))

# Set flags for local compilation
LOCAL_FLAGS:
	$(eval NVCC = $(LOCAL_NVCC))
	$(eval MPICC = $(LOCAL_MPICC))
	$(eval CUDA_FLAGS = $(LOCAL_CUDA_FLAGS))
	$(eval MPI_FLAGS = $(LOCAL_MPI_FLAGS))
	$(eval CUDA_LIBPATH = $(LOCAL_CUDA_LIBPATH))

# Compilation rules for MPI object
$(MPI_OBJ): $(MPI_SRC)
	$(MPICC) -c -o $(MPI_OBJ) $(MPI_SRC) $(MPI_FLAGS)

# Compilation rules for CUDA object
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) -c -o $(CUDA_OBJ) $(CUDA_SRC) $(NVCC_FLAGS)

# Linking
$(TARGET): $(MPI_OBJ) $(CUDA_OBJ)
	$(MPICC) -o $(TARGET) $(MPI_OBJ) $(CUDA_OBJ) $(CUDA_FLAGS)

# Clean
clean:
	rm -f $(TARGET) $(MPI_OBJ) $(CUDA_OBJ)

.PHONY: all local REMOTE_FLAGS LOCAL_FLAGS clean
