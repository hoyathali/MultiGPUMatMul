usr/local/cuda/bin/nvcc -c mult.cu -o mult.o
mpic++ -c topo.cpp -o topo.o
mpic++ topo.o main.o -L/usr/local/cuda/lib64 -lcudart
