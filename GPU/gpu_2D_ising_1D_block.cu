/*
 * GPU-Accelerated 2D Ising Model Simulator
 *
 * This CUDA implementation simulates the 2D Ising model using the Metropolis-Hastings
 * algorithm with checkerboard decomposition for efficient parallel updates.
 *
 * Key Features:
 * - Cold and hot lattice initialization
 * - Energy and magnetization calculations with GPU reduction
 * - Metropolis-Hastings evolution with checkerboard coloring
 * - Stateless RNG using Philox counter-based generator
 * - Memory-efficient 1D block indexing
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../s_functions.h"
#include "../g_functions.h"
#include "../ge_functions.h"

#define THREADS_PER_BLOCK 256
#define PRINT_UP_TO 10
#define DEBUG 1

// ============================================================
// GPU KERNEL: Lattice Initialization (Cold Start)
// ============================================================
// Initializes lattice with all spins to the same value (+1 or -1)
// Uses 1D block indexing for memory efficiency
__global__ void initialize_lattice_gpu_cold_1D_block(int8_t *lattice, int size_x, int size_y, int sign)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = size_x * size_y;

    if (idx < N)
    {
        lattice[idx] = sign;
    }
}

// ============================================================
// GPU DEVICE FUNCTION: Stateless Random Number Generation
// ============================================================
// Generates uniform random number using Philox counter-based RNG
// seed: unique seed per simulation
// counter: unique counter per thread/call
__device__ __forceinline__ float rng_uniform_stateless_1D_block(
    unsigned long long seed,
    unsigned long long counter)
{
    curandStatePhilox4_32_10 state;
    curand_init(seed, counter, 0, &state);
    return curand_uniform(&state);
}

// ============================================================
// GPU KERNEL: Lattice Initialization (Hot Start)
// ============================================================
// Initializes lattice with random spins using stateless RNG
// Uses 1D block indexing; spins are randomly ±1 with 50/50 probability
__global__ void initialize_lattice_gpu_hot_1D_block(int8_t *lattice, unsigned long long seed, int size_x, int size_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = size_x * size_y;

    if (idx < N)
    {
        unsigned long long counter = (unsigned long long)N + idx;
        float r = rng_uniform_stateless_1D_block(seed, counter);
        lattice[idx] = (r < 0.5f) ? -1 : 1;
    }
}

// ============================================================
// ENERGY & MAGNETIZATION CALCULATIONS
// ============================================================

// GPU KERNEL: Calculate Energy (per-site contributions)
// Computes energy for each lattice site using periodic boundary conditions
// Energy = -J * sum(spin * neighbors) - h * spin
__global__ void energy_2D_kernel_gpu_1D_block(const int8_t *lattice, float *energy_partial, int size_x, int size_y, float J, float h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = size_x * size_y;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        int spin = lattice[i];
        int row = i / size_x;
        int col = i % size_x;

        // Periodic boundary conditions
        int right = row * size_x + (col + 1) % size_x;
        int down = ((row + 1) % size_y) * size_x + col;

        float e = 0.0f;
        e -= J * spin * (lattice[right] + lattice[down]);
        e -= h * spin;

        energy_partial[i] = e;
    }
}

// HOST FUNCTION: Calculate Total Energy
// Launches kernel to compute per-site energies, then sums them on host
float energy_2D_gpu_1D_block(int8_t *d_lattice, int size_x, int size_y, float J, float h)
{
    size_t N = size_x * size_y;
    size_t bytes = N * sizeof(float);

    float *d_energy = nullptr;
    cudaMalloc(&d_energy, bytes);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    energy_2D_kernel_gpu_1D_block<<<blocks, threads>>>(d_lattice, d_energy, size_x, size_y, J, h);
    cudaDeviceSynchronize();

    std::vector<float> h_energy(N);
    cudaMemcpy(h_energy.data(), d_energy, bytes, cudaMemcpyDeviceToHost);

    float energy = 0.0f;
    for (size_t i = 0; i < N; i++)
        energy += h_energy[i];

    cudaFree(d_energy);

    if (DEBUG)
        printf("[CHECKPOINT] Energy calculation completed: E = %f\n", energy);

    return energy;
}

// GPU DEVICE FUNCTION: Calculate Energy Change for Spin Flip
// Computes ΔE = 2 * spin * (J * sum_neighbors + h)
// Uses coalesced memory reads for all four neighbors
__device__ __forceinline__ float d_energy_2D_gpu_1D_block(const int8_t *lattice, int idx, int size_x, int size_y, float J, float h)
{
    int spin = lattice[idx];
    int row = idx / size_x;
    int col = idx % size_x;

    // Precompute wrapped coordinates with periodic boundaries
    int right = row * size_x + ((col + 1) % size_x);
    int left = row * size_x + ((col - 1 + size_x) % size_x);
    int bottom = (((row + 1) % size_y) * size_x) + col;
    int top = (((row - 1 + size_y) % size_y) * size_x) + col;

    // Coalesced memory reads
    int8_t nn_right = lattice[right];
    int8_t nn_left = lattice[left];
    int8_t nn_bottom = lattice[bottom];
    int8_t nn_top = lattice[top];

    int8_t sum_nn = nn_right + nn_left + nn_bottom + nn_top;

    return 2.0f * spin * (J * sum_nn + h);
}

// ============================================================
// METROPOLIS-HASTINGS EVOLUTION
// ============================================================

// GPU KERNEL: Single Color Checkerboard Update
// Updates spins of one color (black or white) in the checkerboard pattern
// Uses Metropolis acceptance criterion: accept if dE <= 0 or with probability exp(-beta*dE)
__global__ void MH_1color_checkerboard_gpu_1D_block(int8_t *lattice, unsigned long long seed, int size_x, int size_y, float J, float h, float beta, int color, int sweep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = size_x * size_y;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        int row = i / size_x;
        int col = i % size_x;

        // Skip sites not matching current color
        if (((row + col) & 1) != color)
            continue;

        int8_t spin = lattice[i];

        // Calculate energy change for spin flip
        float dE = d_energy_2D_gpu_1D_block(lattice, i, size_x, size_y, J, h);

        if (dE <= 0.0f)
        {
            // Always accept energy-lowering moves
            lattice[i] = -spin;
        }
        else
        {
            // Accept higher energy moves with Boltzmann probability
            unsigned long long counter = sweep * (unsigned long long)(size_x * size_y) * 2 + color * (unsigned long long)(size_x * size_y) + i;
            float u = rng_uniform_stateless_1D_block(seed, counter);
            if (u < __expf(-dE * beta))
            {
                lattice[i] = -spin;
            }
        }
    }
}

// HOST FUNCTION: Perform One Checkerboard Sweep
// Updates both black and white sublattices sequentially
void MH_checkboard_sweep_gpu_1D_block(int8_t *lattice, unsigned long long seed, int size_x, int size_y, float J, float h, float beta, int sweep)
{
    int N = size_x * size_y;
    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    // Update black sites (checkerboard color 0)
    MH_1color_checkerboard_gpu_1D_block<<<blocks, threads>>>(lattice, seed, size_x, size_y, J, h, beta, 0, sweep);

    // Update white sites (checkerboard color 1)
    MH_1color_checkerboard_gpu_1D_block<<<blocks, threads>>>(lattice, seed, size_x, size_y, J, h, beta, 1, sweep);

    if (DEBUG)
        printf("[CHECKPOINT] Sweep %d completed\n", sweep);
}

// ============================================================
// MAGNETIZATION CALCULATION
// ============================================================

// GPU KERNEL: Calculate Magnetization with Block Reduction
// Uses shared memory and warp-level reduction for efficient GPU reduction
__global__ void magnetization_2D_kernel_gpu_1D_block(
    const int8_t *lattice,
    int *magnetization,
    int N)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    // Grid-stride loop with coalesced memory access
    for (int i = global_tid; i < N; i += stride)
        local_sum += lattice[i];

    // Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction: reduce to 32 threads
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Final warp reduction (32 threads to 1)
    // Use volatile to prevent compiler reordering
    if (tid < 32)
    {
        volatile int *smem = sdata;
        if (blockDim.x >= 64)
            smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32)
            smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16)
            smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)
            smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)
            smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)
            smem[tid] += smem[tid + 1];
    }

    // One atomic operation per block to accumulate results
    if (tid == 0)
        atomicAdd(magnetization, sdata[0]);
}

// HOST FUNCTION: Calculate Total Magnetization
// Launches reduction kernel and returns total magnetization
int magnetization_2D_gpu_1D_block(int8_t *d_lattice,
                                  int size_x,
                                  int size_y)
{
    int N = size_x * size_y;

    int *d_mag;
    cudaMalloc(&d_mag, sizeof(int));
    cudaMemset(d_mag, 0, sizeof(int));

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    magnetization_2D_kernel_gpu_1D_block<<<blocks, threads, threads * sizeof(int)>>>(
        d_lattice, d_mag, N);

    int h_mag;
    cudaMemcpy(&h_mag, d_mag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_mag);

    if (DEBUG)
        printf("[CHECKPOINT] Magnetization calculation completed: M = %d\n", h_mag);

    return h_mag;
}

// ============================================================
// HOT LATTICE SANITY CHECK
// ============================================================

// GPU KERNEL: Check Randomness of Hot-Start Lattice
// Verifies: magnetization near 0, spin balance, nearest-neighbor correlations weak
__global__ void hot_lattice_sanity_kernel_1D_block(
    const int8_t *lattice,
    int size_x,
    int size_y,
    int *magnetization,
    int *spin_sum,
    int *nn_corr_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = size_x * size_y;

    if (idx >= N)
        return;

    int s = lattice[idx];

    // Accumulate magnetization and spin balance
    atomicAdd(magnetization, s);
    atomicAdd(spin_sum, s);

    // Nearest neighbor correlation (right + down, periodic)
    int row = idx / size_x;
    int col = idx % size_x;

    int right = row * size_x + (col + 1) % size_x;
    int down = ((row + 1) % size_y) * size_x + col;

    int corr = s * (lattice[right] + lattice[down]);
    atomicAdd(nn_corr_sum, corr);
}

// HOST FUNCTION: Validate Hot Lattice Randomness
// Returns true if lattice passes randomness criteria
bool check_hot_lattice_randomness_1D_block(
    const int8_t *d_lattice,
    int size_x,
    int size_y)
{
    int N = size_x * size_y;

    int *d_M, *d_S, *d_C;
    cudaMalloc(&d_M, sizeof(int));
    cudaMalloc(&d_S, sizeof(int));
    cudaMalloc(&d_C, sizeof(int));

    cudaMemset(d_M, 0, sizeof(int));
    cudaMemset(d_S, 0, sizeof(int));
    cudaMemset(d_C, 0, sizeof(int));

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    hot_lattice_sanity_kernel_1D_block<<<blocks, threads>>>(
        d_lattice, size_x, size_y,
        d_M, d_S, d_C);

    int M, S, C;
    cudaMemcpy(&M, d_M, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&S, d_S, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&C, d_C, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_S);
    cudaFree(d_C);

    double sqrtN = sqrt((double)N);

    // Acceptance thresholds for randomness
    double max_M = 2.0 * sqrtN;
    double max_S = 2.0 * sqrtN;
    double max_C = 0.05 * N;

    bool ok_M = fabs((double)M) < max_M;
    bool ok_S = fabs((double)S) < max_S;
    bool ok_C = fabs((double)C) < max_C;

    if (DEBUG)
    {
        printf("[CHECKPOINT] Hot lattice sanity check:\n");
        printf("  Magnetization: %d (threshold: %.1f) - %s\n", M, max_M, ok_M ? "PASS" : "FAIL");
        printf("  Spin sum: %d (threshold: %.1f) - %s\n", S, max_S, ok_S ? "PASS" : "FAIL");
        printf("  NN correlation: %d (threshold: %.1f) - %s\n", C, max_C, ok_C ? "PASS" : "FAIL");
    }

    return ok_M && ok_S && ok_C;
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

// Print lattice to console (host memory)
void print_lattice_gpu_1D_block(const int8_t *lattice, int size_x, int size_y)
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

// Save lattice to file with descriptive filename
void save_lattice_1D_block(const char *folder, int8_t *lattice, int type, int size_x, int size_y, float J, float h, float T, int mc_steps)
{
    char filename[512];

    snprintf(filename, sizeof(filename),
             "%s/ising_J%.3f_h%.3f_T%.3f_Lx%d_Ly%d_MC%d_type%d.dat",
             folder, J, h, T, size_x, size_y, mc_steps, type);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        perror("Error opening file for lattice save");
        return;
    }

    for (int i = 0; i < size_y; i++)
    {
        for (int j = 0; j < size_x; j++)
        {
            fprintf(fp, "%d ", lattice[i * size_x + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    if (DEBUG)
        printf("[CHECKPOINT] Lattice saved to %s\n", filename);
}

// ============================================================
// MAIN SIMULATION FUNCTION
// ============================================================

// Run Ising simulation and return observables
extern "C" Observables run_ising_simulation_1D_block_gpu(
    int lattice_size_x,
    int lattice_size_y,
    int type,
    float J, float h,
    float kB, float T,
    int n_steps)
{
    Observables out;

    const int N = lattice_size_x * lattice_size_y;
    const size_t lattice_bytes = N * sizeof(int8_t);

    const float beta = 1.0f / (kB * T);
    const unsigned long long seed = (unsigned long long)time(NULL);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    if (DEBUG)
        printf("[CHECKPOINT] Simulation initialized: N=%d, T=%.3f, seed=%llu\n", N, T, seed);

    // ============================================================
    // Device Memory Allocation
    // ============================================================
    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    cudaDeviceSynchronize();

    // ============================================================
    // Lattice Initialization
    // ============================================================
    clock_t t0 = clock();

    if (type == 0)
    {
        // Cold start: all spins +1
        initialize_lattice_gpu_cold_1D_block<<<blocks, threads>>>(
            d_lattice, lattice_size_x, lattice_size_y, +1);
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization with spin +1\n");
    }
    else if (type == 1)
    {
        // Cold start: all spins -1
        initialize_lattice_gpu_cold_1D_block<<<blocks, threads>>>(
            d_lattice, lattice_size_x, lattice_size_y, -1);
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization with spin -1\n");
    }
    else
    {
        // Hot start: random spins
        initialize_lattice_gpu_hot_1D_block<<<blocks, threads>>>(
            d_lattice, seed, lattice_size_x, lattice_size_y);

        cudaError_t err_i = cudaGetLastError();
        if (err_i != cudaSuccess)
        {
            printf("[ERROR] Init kernel launch failed: %s\n", cudaGetErrorString(err_i));
            cudaFree(d_lattice);
            memset(&out, 0, sizeof(Observables));
            return out;
        }
        if (DEBUG)
            printf("[CHECKPOINT] Hot initialization completed\n");
    }

    cudaDeviceSynchronize();

    clock_t t1 = clock();
    out.initialization_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    // ============================================================
    // Hot Lattice Sanity Check
    // ============================================================
    if (type == 3)
    {
        bool ok = check_hot_lattice_randomness_1D_block(
            d_lattice, lattice_size_x, lattice_size_y);

        if (!ok)
        {
            printf("[WARNING] Hot lattice failed randomness check\n");
        }
    }

    // ============================================================
    // Metropolis-Hastings Evolution
    // ============================================================
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    if (DEBUG)
        printf("[CHECKPOINT] Starting evolution: %d sweeps\n", n_sweeps);

    t0 = clock();

    for (int sweep = 0; sweep < n_sweeps; ++sweep)
    {
        MH_checkboard_sweep_gpu_1D_block(
            d_lattice,
            seed,
            lattice_size_x,
            lattice_size_y,
            J, h, beta,
            sweep);
    }

    cudaDeviceSynchronize();

    t1 = clock();
    out.MH_evolution_time = (float)(t1 - t0) / CLOCKS_PER_SEC;
    out.MH_evolution_time_over_steps = out.MH_evolution_time / (double)(n_sweeps * lattice_size_x * lattice_size_y);

    if (DEBUG)
        printf("[CHECKPOINT] Evolution completed in %.4f seconds\n", out.MH_evolution_time);

    // ============================================================
    // Observable Measurements
    // ============================================================
    float E = energy_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y, J, h);
    int M = magnetization_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y);

    out.E = E;
    out.e_density = E / (float)N;
    out.m = (float)M;
    out.m_density = (float)M / (float)N;

    if (DEBUG)
        printf("[CHECKPOINT] Measurements: E=%.4f, M=%d\n", E, M);

    // ============================================================
    // Cleanup
    // ============================================================
    cudaFree(d_lattice);

    cudaDeviceSynchronize();
    cudaError_t err_o = cudaGetLastError();
    if (err_o != cudaSuccess)
    {
        printf("[WARNING] Cleanup error: %s\n", cudaGetErrorString(err_o));
    }

    if (DEBUG)
        printf("[CHECKPOINT] Simulation completed successfully\n");

    return out;
}

// ============================================================
// MAIN SIMULATION FUNCTION WITH LATTICE SAVING
// ============================================================

// Run Ising simulation with optional lattice saving
extern "C" Observables run_ising_simulation_1D_block_gpu_save(
    int lattice_size_x,
    int lattice_size_y,
    int type,
    float J, float h,
    float kB, float T,
    int n_steps, int save_lattice_flag, const char *save_folder)
{
    Observables out;

    const int N = lattice_size_x * lattice_size_y;
    const size_t lattice_bytes = N * sizeof(int8_t);

    const float beta = 1.0f / (kB * T);
    const unsigned long long seed = (unsigned long long)time(NULL);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    if (DEBUG)
        printf("[CHECKPOINT] Simulation with save initialized: N=%d, T=%.3f\n", N, T);

    // ============================================================
    // Device Memory Allocation
    // ============================================================
    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    cudaDeviceSynchronize();

    // ============================================================
    // Lattice Initialization
    // ============================================================
    clock_t t0 = clock();

    if (type == 0)
    {
        initialize_lattice_gpu_cold_1D_block<<<blocks, threads>>>(
            d_lattice, lattice_size_x, lattice_size_y, +1);
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization with spin +1\n");
    }
    else if (type == 1)
    {
        initialize_lattice_gpu_cold_1D_block<<<blocks, threads>>>(
            d_lattice, lattice_size_x, lattice_size_y, -1);
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization with spin -1\n");
    }
    else
    {
        initialize_lattice_gpu_hot_1D_block<<<blocks, threads>>>(
            d_lattice, seed, lattice_size_x, lattice_size_y);

        cudaError_t err_i = cudaGetLastError();
        if (err_i != cudaSuccess)
        {
            printf("[ERROR] Init kernel launch failed: %s\n", cudaGetErrorString(err_i));
            cudaFree(d_lattice);
            memset(&out, 0, sizeof(Observables));
            return out;
        }
        if (DEBUG)
            printf("[CHECKPOINT] Hot initialization completed\n");
    }

    cudaDeviceSynchronize();

    clock_t t1 = clock();
    out.initialization_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    // ============================================================
    // Save Initial Lattice
    // ============================================================
    if (save_lattice_flag == 1)
    {
        std::vector<int8_t> h_lattice(N);
        cudaMemcpy(h_lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);
        save_lattice_1D_block(save_folder, h_lattice.data(), type, lattice_size_x, lattice_size_y, J, h, T, 0);
        if (DEBUG)
            printf("[CHECKPOINT] Initial lattice saved\n");
    }

    // ============================================================
    // Hot Lattice Sanity Check
    // ============================================================
    if (type == 3)
    {
        bool ok = check_hot_lattice_randomness_1D_block(
            d_lattice, lattice_size_x, lattice_size_y);

        if (!ok)
        {
            printf("[WARNING] Hot lattice failed randomness check\n");
        }
    }

    // ============================================================
    // Metropolis-Hastings Evolution
    // ============================================================
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    if (DEBUG)
        printf("[CHECKPOINT] Starting evolution: %d sweeps\n", n_sweeps);

    t0 = clock();

    for (int sweep = 0; sweep < n_sweeps; ++sweep)
    {
        MH_checkboard_sweep_gpu_1D_block(
            d_lattice,
            seed,
            lattice_size_x,
            lattice_size_y,
            J, h, beta,
            sweep);
    }

    cudaDeviceSynchronize();

    t1 = clock();
    out.MH_evolution_time = (float)(t1 - t0) / CLOCKS_PER_SEC;
    out.MH_evolution_time_over_steps = out.MH_evolution_time / (double)(n_sweeps * lattice_size_x * lattice_size_y);

    if (DEBUG)
        printf("[CHECKPOINT] Evolution completed in %.4f seconds\n", out.MH_evolution_time);

    // ============================================================
    // Observable Measurements
    // ============================================================
    float E = energy_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y, J, h);
    int M = magnetization_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y);

    out.E = E;
    out.e_density = E / (float)N;
    out.m = (float)M;
    out.m_density = (float)M / (float)N;

    if (DEBUG)
        printf("[CHECKPOINT] Measurements: E=%.4f, M=%d\n", E, M);

    // ============================================================
    // Save Final Lattice
    // ============================================================
    if (save_lattice_flag == 1)
    {
        std::vector<int8_t> h_lattice(N);
        cudaMemcpy(h_lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);
        save_lattice_1D_block(save_folder, h_lattice.data(), type, lattice_size_x, lattice_size_y, J, h, T, n_steps);
        if (DEBUG)
            printf("[CHECKPOINT] Final lattice saved\n");
    }

    // ============================================================
    // Cleanup
    // ============================================================
    cudaFree(d_lattice);

    cudaDeviceSynchronize();
    cudaError_t err_o = cudaGetLastError();
    if (err_o != cudaSuccess)
    {
        printf("[WARNING] Cleanup error: %s\n", cudaGetErrorString(err_o));
    }

    if (DEBUG)
        printf("[CHECKPOINT] Simulation completed successfully\n");

    return out;
}

// ============================================================
// STANDALONE EXECUTABLE CODE
// ============================================================

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

    float J = 1;
    float h = 1;
    float kB = 1.0;
    float T = 1.0;
    float beta = 1 / T / kB;

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int8_t);

    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    unsigned long long seed = (unsigned long long)time(NULL);

    cudaError_t err;
    int initialization = 1; // 0 for cold, 1 for hot

    printf("[MAIN] Starting 2D Ising Model Simulation\n");
    printf("[MAIN] Lattice size: %dx%d (N=%zu)\n", lattice_size_x, lattice_size_y, N);

    if (initialization == 0)
    {
        printf("[MAIN] Cold initialization\n");
        initialize_lattice_gpu_cold_1D_block<<<blocks, threads>>>(d_lattice, lattice_size_x, lattice_size_y, -1);
    }
    else if (initialization == 1)
    {
        printf("[MAIN] Hot initialization\n");
        initialize_lattice_gpu_hot_1D_block<<<blocks, threads>>>(d_lattice, seed, lattice_size_x, lattice_size_y);

        bool is_random = check_hot_lattice_randomness_1D_block(d_lattice, lattice_size_x, lattice_size_y);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[ERROR] CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if (is_random)
        {
            printf("[MAIN] Hot lattice is effectively random\n");
        }
        else
        {
            printf("[ERROR] Hot lattice randomness check failed\n");
            return 1;
        }
    }
    else
    {
        printf("[ERROR] Invalid initialization choice (0 or 1)\n");
        return 1;
    }

    float energy = energy_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y, J, h);

    std::vector<int8_t> lattice(N, 0);
    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\n[MAIN] Lattice after GPU initialization:\n");
    int print_x = min(PRINT_UP_TO, (int)lattice_size_x);
    int print_y = min(PRINT_UP_TO, (int)lattice_size_y);
    print_lattice_gpu_1D_block(lattice.data(), print_x, print_y);

    printf("[MAIN] Energy: %f\n", energy);

    int M = magnetization_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y);
    printf("[MAIN] Magnetization: %d\n", M);

    printf("\n[MAIN] Starting 10 MH sweeps\n");
    for (int i = 0; i < 10; i++)
    {
        MH_checkboard_sweep_gpu_1D_block(d_lattice, seed, lattice_size_x, lattice_size_y, J, h, beta, i);
        printf("[MAIN] Sweep %d completed\n", i + 1);
    }

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\n[MAIN] Lattice after MH evolution:\n");
    print_lattice_gpu_1D_block(lattice.data(), print_x, print_y);

    energy = energy_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y, J, h);
    printf("[MAIN] Energy: %f\n", energy);

    M = magnetization_2D_gpu_1D_block(d_lattice, lattice_size_x, lattice_size_y);
    printf("[MAIN] Magnetization: %d\n", M);

    cudaFree(d_lattice);

    printf("[MAIN] Simulation completed successfully\n");
    return 0;
}
#endif