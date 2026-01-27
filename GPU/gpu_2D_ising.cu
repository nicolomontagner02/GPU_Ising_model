#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK_X 10
#define THREADS_PER_BLOCK_Y 10
#define PRINT_UP_TO 10

#define DEBUG 0

// ###############################################################
// CUDA FUNCTIONS + CALLING
// ###############################################################

__global__ void init_rng_states(curandState *states, int size_x, int size_y, unsigned long long seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;

        curand_init(seed, idx, 0, &states[idx]);
    }
}

// GPU kernel to initialize lattice (cold start: all spins +1 or -1)
__global__ void initialize_lattice_gpu_cold(int *lattice, int size_x, int size_y, int sign)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        lattice[row * size_x + col] = sign;
    }
}

// GPU kernel to initialize lattice (hot case: random spin of each site)
__global__ void initialize_lattice_gpu_hot(int *lattice, curandState *states, int size_x, int size_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;

        float r = curand_uniform(&states[idx]); // (0,1]
        lattice[idx] = (r < 0.5f) ? -1 : 1;
    }
}

// Evaluate energy
__global__ void energy_2D_kernel(const int *lattice, float *energy_partial, int size_x, int size_y, float J, float h)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {

        int idx = row * size_x + col;

        int spin = lattice[idx];

        int right = lattice[(col + 1) % size_x + row * size_x];
        int down = lattice[col + (row + 1) % size_y * size_x];

        float e = 0.0f;
        e -= J * spin * (right + down);
        e -= h * spin;

        energy_partial[idx] = e;
    }
}

float energy_2D_gpu(int *d_lattice, int size_x, int size_y, float J, float h)
{
    size_t N = size_x * size_y;
    size_t bytes = N * sizeof(float);

    float *d_energy = nullptr;
    cudaMalloc(&d_energy, bytes);

    dim3 block(8, 8);
    dim3 grid(
        (size_x + block.x - 1) / block.x,
        (size_y + block.y - 1) / block.y);

    energy_2D_kernel<<<grid, block>>>(d_lattice, d_energy, size_x, size_y, J, h);
    cudaDeviceSynchronize();

    std::vector<float> h_energy(N);
    cudaMemcpy(h_energy.data(), d_energy, bytes, cudaMemcpyDeviceToHost);

    float energy = 0.0f;
    for (size_t i = 0; i < N; i++)
        energy += h_energy[i];

    cudaFree(d_energy);
    return energy;
}

float d_energy_2D(int *lattice, int i, int j, int size_x, int size_y, float J, float h)
{

    int idx = i * size_x + j;

    int spin = lattice[idx];
    float sum_nn = 0.0f;

    sum_nn += lattice[(idx + 1) % size_x];                 // right
    sum_nn += lattice[(idx - 1) % size_x];                 // left
    sum_nn += lattice[(idx + size_x) % (size_y * size_x)]; // bottom
    sum_nn += lattice[(idx - size_x) % (size_y * size_x)]; // top

    float d_energy = 2 * spin * (J * sum_nn + h);

    return d_energy;
}

__global__ void MH_1color_checkerboard_openmp(int *lattice, curandState *states, int size_x, int size_y, float J, float h, float kB, float T, int color)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {

        int idx = row * size_x + col;

        if (idx % 2 == color)
        { // TO FIX FOR EVEN SIZE MATRICES

            float d_energy = d_energy_2D(lattice, row, col, size_x, size_y, J, h);

            if (d_energy <= 0.0f)
            {

                lattice[idx] *= -1;
            }
            else
            {
                float p = exp(-d_energy / (kB * T));
                float u = curand_uniform(&states[idx]);
                if (u < p)
                {
                    lattice[idx] *= -1;
                }
            }
        }
    }
}

// ###############################################################
// HELPER FUNCTIONS
// ###############################################################

// Print lattice (host)
void print_lattice(const int *lattice, int size_x, int size_y)
{
    for (int i = 0; i < size_y; i++)
    {
        for (int j = 0; j < size_x; j++)
        {
            printf("%2d ", lattice[i * size_x + j]);
        }
        printf("\n");
    }
}

// ###############################################################
// MAIN
// ###############################################################

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
        return 1;
    }

    int lattice_size_x = atoi(argv[1]);
    int lattice_size_y = atoi(argv[2]);

    // int sign = -1;
    float J = 1;
    float h = 1;

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int);

    int *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    curandState *d_rng_states = nullptr;
    cudaMalloc(&d_rng_states, N * sizeof(curandState));

    unsigned long long seed = (unsigned long long)time(NULL);

    // RNG initialization
    init_rng_states<<<grid, block>>>(d_rng_states, lattice_size_x, lattice_size_y, seed);
    cudaDeviceSynchronize();

    // Hot initialization of the lattice
    initialize_lattice_gpu_hot<<<grid, block>>>(d_lattice, d_rng_states, lattice_size_x, lattice_size_y);
    // initialize_lattice_gpu_cold<<<grid, block>>>(d_lattice, lattice_size_x, lattice_size_y, -1);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float energy = energy_2D_gpu(d_lattice, lattice_size_x, lattice_size_y, J, h);

    std::vector<int> lattice(N, 0);

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU initialization:\n");
    int print_x = min(PRINT_UP_TO, lattice_size_x);
    int print_y = min(PRINT_UP_TO, lattice_size_y);
    print_lattice(lattice.data(), print_x, print_y);

    printf("Energy: %f\n", energy);

    cudaFree(d_lattice);
    cudaFree(d_rng_states);
    return 0;
}