#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_X 32  
#define THREADS_PER_BLOCK_Y 32 

#define DEBUG 0

// GPU kernel to initialize lattice (cold start: all spins +1)
__global__ void initialize_lattice_gpu_cold(int *lattice, int size_x, int size_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x) {
        lattice[row * size_x + col] = 1;
    }
}

// Print lattice (host)
void print_lattice(const int *lattice, int size_x, int size_y)
{
    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            printf("%2d ", lattice[i * size_x + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
        return 1;
    }

    int lattice_size_x = atoi(argv[1]);
    int lattice_size_y = atoi(argv[2]);

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int);

    std::vector<int> lattice(N, 0);

    int *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    cudaMemcpy(d_lattice, lattice.data(), lattice_bytes, cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y
    );

    printf("Initial lattice (host):\n");
    print_lattice(lattice.data(), lattice_size_x, lattice_size_y);

    initialize_lattice_gpu_cold<<<grid, block>>>(
        d_lattice, lattice_size_x, lattice_size_y
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU initialization:\n");
    print_lattice(lattice.data(), lattice_size_x, lattice_size_y);

    cudaFree(d_lattice);
    return 0;
}