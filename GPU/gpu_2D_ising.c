#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>
#include "../functions.h"

#define DEBUG 0

__global__ void initialize_lattice_gpu(int *lattice, int size_x, int size_y){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

}

int main(int argc, char *argv[]) {

    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
        printf("Example: %s 10 10\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int lattice_size_x = atoi(argv[1]);
    int lattice_size_y = atoi(argv[2]);

    int type = 3;
    float J = 1;
    float h = 1;
    float kB = 1.0*exp(-23);
    float T = 100;
    int n_steps = 1000000;

    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("MC steps            : %d\n", n_steps);
    printf("Initialization type : %s\n",
        type == 1 ? "All up" :
        type == 2 ? "All down" : "Random");
    printf("\n");
    printf("========================================\n");

    int* d_lattice;
    size_t N = lattice_size_x * lattice_size_y;

    cudaMalloc(&d_lattice, N * sizeof(int));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;



    return 0;
}