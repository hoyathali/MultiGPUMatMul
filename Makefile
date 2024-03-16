ALL:
	usr/local/cuda/bin/nvcc -c mult.cu -o mult.o
	mpic++ -c topo.cpp -o topo.o
	mpic++ topo.o main.o -L/usr/local/cuda/lib64 -lcudart

LOCAL:
	nvcc -g -c mult.cu -o mult.o
	mpic++ -g -c main.cpp -L/opt/cuda/lib -lcudart -o main.o
	mpic++ -g mult.o main.o -L/opt/cuda/lib -lcudart

make L_RUN:
