#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
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
__global__ void initialize_lattice_gpu_cold(int8_t *lattice, int size_x, int size_y, int sign)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        lattice[row * size_x + col] = sign;
    }
}

// GPU kernel to initialize lattice (hot case: random spin of each site)
__global__ void initialize_lattice_gpu_hot(int8_t *lattice, curandState *states, int size_x, int size_y)
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
__global__ void energy_2D_kernel(const int8_t *lattice, float *energy_partial, int size_x, int size_y, float J, float h)
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

float energy_2D_gpu(int8_t *d_lattice, int size_x, int size_y, float J, float h)
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

__device__ __forceinline__ float d_energy_2D(const int8_t *lattice, int i, int j, int size_x, int size_y, float J, float h)
{
    int idx = i * size_x + j;
    int spin = lattice[idx];

    int8_t sum_nn = 0;

    int right = i * size_x + (j + 1 + size_x) % size_x;
    int left = i * size_x + (j - 1 + size_x) % size_x;
    int bottom = ((i + 1 + size_y) % size_y) * size_x + j;
    int top = ((i - 1 + size_y) % size_y) * size_x + j;

    sum_nn += lattice[right];
    sum_nn += lattice[left];
    sum_nn += lattice[bottom];
    sum_nn += lattice[top];

    return 2.0f * spin * (J * sum_nn + h);
}

__global__ void MH_1color_checkerboard_gpu(int8_t *lattice, curandState *states, int size_x, int size_y, float J, float h, float beta, int color)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size_y || col >= size_x)
        return;

    if (((row + col) & 1) != color)
        return;

    int idx = row * size_x + col;

    float dE = d_energy_2D(lattice, row, col, size_x, size_y, J, h);

    if (dE <= 0.0f)
    {
        lattice[idx] *= -1;
    }
    else
    {
        float u = curand_uniform(&states[idx]);
        if (u < __expf(-dE * beta))
        {
            lattice[idx] *= -1;
        }
    }
}

void MH_checkboard_sweep_gpu(int8_t *lattice, curandState *states, int size_x, int size_y, float J, float h, float beta, dim3 grid, dim3 block)
{

    // Update black sites
    MH_1color_checkerboard_gpu<<<grid, block>>>(lattice, states, size_x, size_y, J, h, beta, 0);

    // Update white sites
    MH_1color_checkerboard_gpu<<<grid, block>>>(lattice, states, size_x, size_y, J, h, beta, 1);
}

__inline__ __device__ int warp_reduce_sum_int(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void magnetization_2D_kernel(const int8_t *lattice,
                                        int *magnetization,
                                        int size_x,
                                        int size_y)
{
    int idx = blockIdx.y * blockDim.y * size_x +
              blockIdx.x * blockDim.x +
              threadIdx.y * size_x +
              threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    // Grid-stride loop (1D over flattened lattice)
    for (int i = idx; i < size_x * size_y; i += stride)
        local_sum += lattice[i];

    // Warp-level reduction
    local_sum = warp_reduce_sum_int(local_sum);

    __shared__ int warp_sums[32]; // max 1024 threads

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    if (lane == 0)
        warp_sums[warp] = local_sum;

    __syncthreads();

    // Final reduction by first warp
    if (warp == 0)
    {
        local_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0;
        local_sum = warp_reduce_sum_int(local_sum);

        if (lane == 0)
            atomicAdd(magnetization, local_sum);
    }
}

int magnetization_2D_gpu(int8_t *d_lattice,
                         int size_x,
                         int size_y,
                         dim3 grid,
                         dim3 block)
{
    int *d_mag;
    cudaMalloc(&d_mag, sizeof(int));
    cudaMemset(d_mag, 0, sizeof(int));

    magnetization_2D_kernel<<<grid, block>>>(
        d_lattice, d_mag, size_x, size_y);

    int h_mag;
    cudaMemcpy(&h_mag, d_mag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_mag);
    return h_mag;
}

// ###############################################################
// HELPER FUNCTIONS
// ###############################################################

// Print lattice (host)
void print_lattice(const int8_t *lattice, int size_x, int size_y)
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
    float kB = 1e-23;
    float T = 100.0;

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int8_t);

    int8_t *d_lattice = nullptr;
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

    std::vector<int8_t> lattice(N, 0);

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU initialization:\n");
    int print_x = min(PRINT_UP_TO, lattice_size_x);
    int print_y = min(PRINT_UP_TO, lattice_size_y);
    print_lattice(lattice.data(), print_x, print_y);

    printf("Energy: %f\n", energy);

    int M = magnetization_2D_gpu(d_lattice,
                                 lattice_size_x,
                                 lattice_size_y,
                                 grid,
                                 block);

    printf("Magnetization: %d\n", M);

    int i = 0;

    while (i < 10)
    {
        MH_checkboard_sweep_gpu(d_lattice, d_rng_states, lattice_size_x, lattice_size_y, J, h, kB * T, grid, block);

        if (DEBUG)
        {
            printf("Sweep %i\n", i + 1);
        }

        i++;
    }

    std::vector<int8_t> lattice_f(N, 0);

    cudaMemcpy(lattice_f.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU MH evolution:\n");
    print_lattice(lattice_f.data(), print_x, print_y);

    float energy_f;
    int M_f;

    energy_f = energy_2D_gpu(d_lattice, lattice_size_x, lattice_size_y, J, h);

    printf("Energy: %f\n", energy_f);

    M_f = magnetization_2D_gpu(d_lattice,
                               lattice_size_x,
                               lattice_size_y,
                               grid,
                               block);

    printf("Magnetization: %d\n", M_f);

    cudaFree(d_lattice);
    cudaFree(d_rng_states);
    return 0;
}