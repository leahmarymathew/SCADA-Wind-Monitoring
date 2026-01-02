CC = gcc
MPICC = mpicc
CFLAGS = -O2 -fopenmp

all:
	$(CC) $(CFLAGS) src/openmp_inference.c -o openmp_inference
	$(MPICC) -O2 src/mpi_inference.c -o mpi_inference

run_openmp:
	./openmp_inference

run_mpi:
	mpirun -np 4 ./mpi_inference

clean:
	rm -f openmp_inference mpi_inference
