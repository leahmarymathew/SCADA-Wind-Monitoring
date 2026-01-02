CC = gcc
CFLAGS = -O2 -fopenmp

all:
	$(CC) $(CFLAGS) src/openmp_inference.c -o openmp_inference

run:
	./openmp_inference

clean:
	rm -f openmp_inference
