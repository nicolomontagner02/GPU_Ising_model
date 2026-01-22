#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_X 10  
#define THREADS_PER_BLOCK_Y 10  

#define DEBUG 0

// GPU kernel to initialize lattice
__global__ void initialize_lattice_gpu_cold(void){//int *lattice, int size_x, int size_y) {
    
    printf("Hello from %i", blockIdx.x);

    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // // Bounds check
    // if (row >= size_y || col >= size_x) return;

    // lattice[row * size_x + col] = 1;
}

// Print lattice function
void print_lattice_gpu(int *lattice, int size_x, int size_y) {
    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            printf("%i ", lattice[i * size_x + j]);
        }
        printf("\n");
    }
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
    float kB = 1.0e-23;
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

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_size = N * sizeof(int);

    std::vector<int> lattice(lattice_size, 0);

    int *d_lattice;

    cudaMalloc((void**)&d_lattice, lattice_size);

    cudaMemcpy(d_lattice, lattice.data(), lattice_size, cudaMemcpyHostToDevice); 

    // Set up CUDA block/grid sizes
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y
    );

    print_lattice_gpu(lattice.data(), lattice_size_x, lattice_size_y);

    initialize_lattice_gpu_cold<<<10, 1>>>();//d_lattice, lattice_size_x, lattice_size_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(lattice.data(), d_lattice, lattice_size, cudaMemcpyDeviceToHost);

    print_lattice_gpu(lattice.data(), lattice_size_x, lattice_size_y);

    free(lattice.data());
    cudaFree(d_lattice);

    return 0;
}
