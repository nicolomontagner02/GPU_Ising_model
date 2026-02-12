/*
 * GPU-Accelerated 2D Ising Model Simulation
 *
 * This code implements a parallel Monte Carlo simulation of the 2D Ising model
 * using CUDA. It includes:
 * - GPU memory management for lattice and RNG states
 * - Lattice initialization (cold/hot start)
 * - Metropolis-Hastings algorithm with checkerboard coloring for parallel updates
 * - Energy and magnetization measurements
 * - Performance timing and statistics
 *
 * The checkerboard pattern ensures no race conditions when updating neighboring spins.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../s_functions.h"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define PRINT_UP_TO 10

#define DEBUG 0

// Macro for conditional debug output
#define DEBUG_PRINT(fmt, ...) \
    if (DEBUG)                \
    printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

// ###############################################################
// CUDA FUNCTIONS + CALLING
// ###############################################################

// Initialize CUDA random number generator states for all threads
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

// Initialize RNG states for a specific tile (for memory-efficient initialization of large lattices)
__global__ void init_rng_states_tile(
    curandState *states,
    int size_x,
    int size_y,
    unsigned long long seed,
    int start_x,
    int start_y,
    int tile_x,
    int tile_y)
{
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = start_y + blockIdx.y * blockDim.y + local_row;
    int col = start_x + blockIdx.x * blockDim.x + local_col;

    if (row >= start_y + tile_y || row >= size_y || col >= start_x + tile_x || col >= size_x)
        return;

    int idx = row * size_x + col;
    curand_init(seed, idx, 0, &states[idx]);
}

// Tiled RNG initialization wrapper for processing large lattices in chunks
void init_rng_states_gpu_tiled(
    curandState *d_states,
    int size_x,
    int size_y,
    unsigned long long seed,
    dim3 block,
    int tile_x,
    int tile_y)
{
    dim3 grid;

    for (int start_y = 0; start_y < size_y; start_y += tile_y)
    {
        int ty = min(tile_y, size_y - start_y);

        for (int start_x = 0; start_x < size_x; start_x += tile_x)
        {
            int tx = min(tile_x, size_x - start_x);

            grid.x = (tx + block.x - 1) / block.x;
            grid.y = (ty + block.y - 1) / block.y;

            init_rng_states_tile<<<grid, block>>>(
                d_states,
                size_x,
                size_y,
                seed,
                start_x,
                start_y,
                tx,
                ty);

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error during RNG init tile (%d,%d): %s\n",
                       start_x, start_y,
                       cudaGetErrorString(err));
                exit(1);
            }
        }
    }
}

// Initialize lattice with all spins pointing in the same direction (cold start)
__global__ void initialize_lattice_gpu_cold(int *lattice, int size_x, int size_y, int sign)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        lattice[row * size_x + col] = sign;
    }
}

// Initialize lattice with random spins (hot start)
__global__ void initialize_lattice_gpu_hot(int *lattice, curandState *states, int size_x, int size_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;

        float r = curand_uniform(&states[idx]); // Random value in (0,1]
        lattice[idx] = (r < 0.5f) ? -1 : 1;
    }
}

// Compute energy contribution for each lattice site
__global__ void energy_2D_kernel_gpu(const int *lattice, float *energy_partial, int size_x, int size_y, float J, float h)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;
        int spin = lattice[idx];

        // Periodic boundary conditions
        int right = lattice[(col + 1) % size_x + row * size_x];
        int down = lattice[col + (row + 1) % size_y * size_x];

        float e = 0.0f;
        e -= J * spin * (right + down); // Interaction energy
        e -= h * spin;                  // External field energy

        energy_partial[idx] = e;
    }
}

// Calculate total energy of the system using parallel reduction
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

    energy_2D_kernel_gpu<<<grid, block>>>(d_lattice, d_energy, size_x, size_y, J, h);
    cudaDeviceSynchronize();

    DEBUG_PRINT("Energy kernel computed partial energies for %zu sites", N);

    std::vector<float> h_energy(N);
    cudaMemcpy(h_energy.data(), d_energy, bytes, cudaMemcpyDeviceToHost);

    float energy = 0.0f;
    for (size_t i = 0; i < N; i++)
        energy += h_energy[i];

    cudaFree(d_energy);
    return energy;
}

// Device function: compute energy difference if spin (i,j) is flipped
__device__ float d_energy_2D_gpu(
    const int *lattice,
    int i, int j,
    int size_x, int size_y,
    float J, float h)
{
    int idx = i * size_x + j;
    int spin = lattice[idx];

    float sum_nn = 0.0f;

    // Get indices of 4 nearest neighbors with periodic boundary conditions
    int right = i * size_x + (j + 1 + size_x) % size_x;
    int left = i * size_x + (j - 1 + size_x) % size_x;
    int bottom = ((i + 1 + size_y) % size_y) * size_x + j;
    int top = ((i - 1 + size_y) % size_y) * size_x + j;

    sum_nn += lattice[right];
    sum_nn += lattice[left];
    sum_nn += lattice[bottom];
    sum_nn += lattice[top];

    // Energy difference when flipping spin: 2 * E_current
    return 2.0f * spin * (J * sum_nn + h);
}

// Metropolis-Hastings update for a single color in checkerboard pattern
__global__ void MH_1color_checkerboard_gpu(int *lattice, curandState *states, int size_x, int size_y, float J, float h, float kB, float T, int color)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size_y || col >= size_x)
        return;

    // Only update sites matching current color (checkerboard pattern)
    if (((row + col) & 1) != color)
        return;

    int idx = row * size_x + col;

    // Compute energy change for spin flip
    float dE = d_energy_2D_gpu(lattice, row, col, size_x, size_y, J, h);

    // Metropolis acceptance criterion
    if (dE <= 0.0f)
    {
        // Always accept if energy decreases
        lattice[idx] *= -1;
    }
    else
    {
        // Accept with Boltzmann probability if energy increases
        float u = curand_uniform(&states[idx]);
        if (u < expf(-dE / (kB * T)))
        {
            lattice[idx] *= -1;
        }
    }
}

// Perform one complete MH sweep using checkerboard coloring (no synchronization needed between colors)
void MH_checkboard_sweep_gpu(int *lattice, curandState *states, int size_x, int size_y, float J, float h, float kB, float T, dim3 grid, dim3 block)
{
    // Update black sites (color 0)
    MH_1color_checkerboard_gpu<<<grid, block>>>(lattice, states, size_x, size_y, J, h, kB, T, 0);

    // Update white sites (color 1)
    MH_1color_checkerboard_gpu<<<grid, block>>>(lattice, states, size_x, size_y, J, h, kB, T, 1);
}

// Compute magnetization using parallel reduction with atomic operations
__global__ void magnetization_2D_kernel(
    const int *lattice,
    int *magnetization,
    int N)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = global_tid; i < N; i += stride)
        local_sum += lattice[i];

    // Store partial sum in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Block-level reduction (tree reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block result using atomicAdd for thread-safe accumulation
    if (tid == 0)
        atomicAdd(magnetization, sdata[0]);
}

// Calculate total magnetization of the system
int magnetization_2D_gpu(int *d_lattice,
                         int size_x,
                         int size_y)
{
    int N = size_x * size_y;

    int *d_mag;
    cudaMalloc(&d_mag, sizeof(int));
    cudaMemset(d_mag, 0, sizeof(int));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    magnetization_2D_kernel<<<blocks, threads, threads * sizeof(int)>>>(
        d_lattice, d_mag, N);

    DEBUG_PRINT("Magnetization kernel launched with %d blocks and %d threads", blocks, threads);

    int h_mag;
    cudaMemcpy(&h_mag, d_mag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_mag);
    return h_mag;
}

// ###############################################################
// HELPER FUNCTIONS
// ###############################################################

// Print a subset of the lattice to console (host function)
void print_lattice_gpu(const int *lattice, int size_x, int size_y)
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

// Main simulation function: initializes, evolves, and measures the 2D Ising model
extern "C" Observables run_ising_simulation_gpu(
    int lattice_size_x,
    int lattice_size_y,
    int type,
    float J, float h, float kB, float T,
    int n_steps)
{
    DEBUG_PRINT("Starting Ising simulation: %dx%d lattice, type=%d, T=%.2f",
                lattice_size_x, lattice_size_y, type, T);

    clock_t t0, t1;

    const int N = lattice_size_x * lattice_size_y;
    const size_t lattice_bytes = N * sizeof(int);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    DEBUG_PRINT("Grid: %dx%d, Block: %dx%d", grid.x, grid.y, block.x, block.y);

    /* -------------------------------------------------- */
    /* GPU memory allocation                              */
    /* -------------------------------------------------- */
    t0 = clock();

    int *d_lattice = nullptr;
    curandState *d_rng_states = nullptr;

    cudaMalloc(&d_lattice, lattice_bytes);
    cudaMalloc(&d_rng_states, N * sizeof(curandState));

    DEBUG_PRINT("Allocated %.2f MB for lattice and %.2f MB for RNG states",
                lattice_bytes / 1e6, N * sizeof(curandState) / 1e6);

    unsigned long long seed = 1234;

    /* RNG initialization */
    init_rng_states<<<grid, block>>>(
        d_rng_states,
        lattice_size_x,
        lattice_size_y,
        seed);
    cudaDeviceSynchronize();

    DEBUG_PRINT("RNG states initialized with seed %llu", seed);

    /* Lattice initialization */
    if (type == 0)
    {
        /* cold start: all spins +1 */
        initialize_lattice_gpu_cold<<<grid, block>>>(
            d_lattice,
            lattice_size_x,
            lattice_size_y,
            +1);
        DEBUG_PRINT("Cold start (all +1)");
    }
    else if (type == 1)
    {
        /* cold start: all spins -1 */
        initialize_lattice_gpu_cold<<<grid, block>>>(
            d_lattice,
            lattice_size_x,
            lattice_size_y,
            -1);
        DEBUG_PRINT("Cold start (all -1)");
    }
    else
    {
        /* hot start */
        initialize_lattice_gpu_hot<<<grid, block>>>(
            d_lattice,
            d_rng_states,
            lattice_size_x,
            lattice_size_y);
        DEBUG_PRINT("Hot start (random spins)");
    }

    cudaDeviceSynchronize();

    t1 = clock();
    float initialization_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    DEBUG_PRINT("Initialization completed in %.4f seconds", initialization_time);

    /* -------------------------------------------------- */
    /* Metropolis–Hastings evolution                      */
    /* -------------------------------------------------- */
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    DEBUG_PRINT("Starting MH evolution: %d sweeps", n_sweeps);

    t0 = clock();

    for (int step = 0; step < n_sweeps; step++)
    {
        MH_checkboard_sweep_gpu(
            d_lattice,
            d_rng_states,
            lattice_size_x,
            lattice_size_y,
            J, h, kB, T,
            grid, block);

        if (DEBUG && (step + 1) % 100 == 0)
            DEBUG_PRINT("Completed %d/%d sweeps", step + 1, n_sweeps);
    }

    cudaDeviceSynchronize();

    t1 = clock();
    float MH_evolution_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    DEBUG_PRINT("MH evolution completed in %.4f seconds", MH_evolution_time);

    /* -------------------------------------------------- */
    /* Measurements (GPU)                                 */
    /* -------------------------------------------------- */
    float E = energy_2D_gpu(
        d_lattice,
        lattice_size_x,
        lattice_size_y,
        J, h);

    DEBUG_PRINT("Total energy: %.2f", E);

    float e_density =
        E / (float)(lattice_size_x * lattice_size_y);

    int m = magnetization_2D_gpu(
        d_lattice,
        lattice_size_x,
        lattice_size_y);

    DEBUG_PRINT("Total magnetization: %d", m);

    float m_density =
        (float)m / (float)(lattice_size_x * lattice_size_y);

    /* -------------------------------------------------- */
    /* Cleanup                                            */
    /* -------------------------------------------------- */
    cudaFree(d_lattice);
    cudaFree(d_rng_states);

    DEBUG_PRINT("GPU memory freed");

    /* -------------------------------------------------- */
    /* Output                                             */
    /* -------------------------------------------------- */
    Observables out;
    out.E = E;
    out.e_density = e_density;
    out.m = (float)m;
    out.m_density = m_density;
    out.initialization_time = initialization_time;
    out.MH_evolution_time = MH_evolution_time;
    out.MH_evolution_time_over_steps =
        MH_evolution_time /
        ((float)n_sweeps * lattice_size_x * lattice_size_y);

    DEBUG_PRINT("Simulation completed successfully");

    return out;
}

// ###############################################################
// MAIN
// ###############################################################
#ifdef STANDALONE_BUILD
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
        return 1;
    }

    int lattice_size_x = atoi(argv[1]);
    int lattice_size_y = atoi(argv[2]);

    DEBUG_PRINT("Main: lattice size %dx%d", lattice_size_x, lattice_size_y);

    float J = 1;
    float h = 1;
    float kB = 1e-23;
    float T = 100.0;

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int);

    int *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    DEBUG_PRINT("Grid config: %dx%d blocks, %dx%d threads", grid.x, grid.y, block.x, block.y);

    curandState *d_rng_states = nullptr;
    cudaMalloc(&d_rng_states, N * sizeof(curandState));

    unsigned long long seed = (unsigned long long)time(NULL);

    DEBUG_PRINT("RNG seed: %llu", seed);

    // RNG initialization
    init_rng_states<<<grid, block>>>(d_rng_states, lattice_size_x, lattice_size_y, seed);
    cudaDeviceSynchronize();

    DEBUG_PRINT("RNG states initialized");

    // Hot initialization of the lattice
    initialize_lattice_gpu_hot<<<grid, block>>>(d_lattice, d_rng_states, lattice_size_x, lattice_size_y);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    DEBUG_PRINT("Lattice initialized");

    float energy = energy_2D_gpu(d_lattice, lattice_size_x, lattice_size_y, J, h);

    std::vector<int> lattice(N, 0);

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    DEBUG_PRINT("Lattice copied to host");

    printf("\nLattice after GPU initialization:\n");
    int print_x = min(PRINT_UP_TO, lattice_size_x);
    int print_y = min(PRINT_UP_TO, lattice_size_y);
    print_lattice_gpu(lattice.data(), print_x, print_y);

    printf("Energy: %f\n", energy);

    int i = 0;

    DEBUG_PRINT("Starting MH sweeps loop");

    while (i < 10)
    {
        MH_checkboard_sweep_gpu(d_lattice, d_rng_states, lattice_size_x, lattice_size_y, J, h, kB, T, grid, block);
        i++;
        DEBUG_PRINT("Completed sweep %d/10", i);
    }

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    DEBUG_PRINT("Final lattice copied to host");

    printf("\nLattice after GPU initialization:\n");
    print_lattice_gpu(lattice.data(), print_x, print_y);

    energy = energy_2D_gpu(d_lattice, lattice_size_x, lattice_size_y, J, h);

    printf("Energy: %f\n", energy);

    cudaFree(d_lattice);
    cudaFree(d_rng_states);

    DEBUG_PRINT("GPU memory freed, program ending");

    return 0;
}
#endif