// This CUDA program implements an efficient 2D Ising model simulator
// using the Metropolis-Hastings algorithm with checkerboard updates.
//
// Key features:
//  - Cold and hot lattice initialization
//  - Stateless RNG for memory efficiency
//  - Checkerboard domain decomposition for better parallelism
//  - Energy and magnetization measurements
//  - Optional lattice saving and randomness checks
// ====================================================================

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

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define PRINT_UP_TO 10

#define DEBUG 1

// ###############################################################
// CUDA FUNCTIONS + CALLING
// ###############################################################

// GPU kernel to initialize lattice (cold start: all spins +1 or -1)
__global__ void initialize_lattice_gpu_cold_2Dblock(int8_t *lattice, int size_x, int size_y, int sign)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        lattice[row * size_x + col] = sign;
    }
}

// Stateless uniform RNG using Philox counter-based generator
__device__ __forceinline__ float rng_uniform_stateless(
    unsigned long long seed,
    unsigned long long counter)
{
    curandStatePhilox4_32_10 state;
    curand_init(seed, counter, 0, &state);
    return curand_uniform(&state);
}

// GPU kernel to initialize lattice with random spins (hot start)
__global__ void initialize_lattice_gpu_hot_2Dblock(int8_t *lattice, unsigned long long seed, int size_x, int size_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;
        unsigned long counter = (unsigned long long)(size_x * size_y) + idx;
        float r = rng_uniform_stateless(seed, counter);
        lattice[idx] = (r < 0.5f) ? -1 : 1;
    }
}

// #######################################################################################################
// ENERGY & MAGNETIZATION
// #######################################################################################################

// Kernel: Compute energy contribution for each site (periodic boundary conditions)
__global__ void energy_2D_kernel_gpu_2Dblock(const int8_t *lattice, float *energy_partial, int size_x, int size_y, float J, float h)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y && col < size_x)
    {
        int idx = row * size_x + col;
        int spin = lattice[idx];

        // Periodic boundary conditions for right and down neighbors
        int right = lattice[(col + 1) % size_x + row * size_x];
        int down = lattice[col + (row + 1) % size_y * size_x];

        float e = 0.0f;
        e -= J * spin * (right + down);
        e -= h * spin;

        energy_partial[idx] = e;
    }
}

// Host function: Calculate total energy of the lattice
float energy_2D_gpu_2Dblock(int8_t *d_lattice, int size_x, int size_y, float J, float h)
{
    size_t N = size_x * size_y;
    size_t bytes = N * sizeof(float);

    float *d_energy = nullptr;
    cudaMalloc(&d_energy, bytes);

    dim3 block(8, 8);
    dim3 grid(
        (size_x + block.x - 1) / block.x,
        (size_y + block.y - 1) / block.y);

    energy_2D_kernel_gpu_2Dblock<<<grid, block>>>(d_lattice, d_energy, size_x, size_y, J, h);
    cudaDeviceSynchronize();

    if (DEBUG)
        printf("[CHECKPOINT] Energy kernel executed\n");

    std::vector<float> h_energy(N);
    cudaMemcpy(h_energy.data(), d_energy, bytes, cudaMemcpyDeviceToHost);

    float energy = 0.0f;
    for (size_t i = 0; i < N; i++)
        energy += h_energy[i];

    cudaFree(d_energy);
    return energy;
}

// Device function: Compute energy change for spin flip at position (i, j)
__device__ __forceinline__ float d_energy_2D_gpu_2Dblock(const int8_t *lattice, int i, int j, int size_x, int size_y, float J, float h)
{
    int idx = i * size_x + j;
    int spin = lattice[idx];

    int8_t sum_nn = 0;

    // Periodic boundary conditions for all four neighbors
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

// Kernel: Metropolis-Hastings update for one checkerboard color
__global__ void MH_1color_checkerboard_gpu_2Dblock(int8_t *lattice, unsigned long long seed, int size_x, int size_y, float J, float h, float beta, int color, int sweep)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size_y || col >= size_x)
        return;

    // Skip sites that don't match the current color (checkerboard pattern)
    if (((row + col) & 1) != color)
        return;

    int idx = row * size_x + col;

    // Compute energy change for spin flip
    float dE = d_energy_2D_gpu_2Dblock(lattice, row, col, size_x, size_y, J, h);

    // Accept or reject flip based on Metropolis criterion
    if (dE <= 0.0f)
    {
        lattice[idx] *= -1;
    }
    else
    {
        unsigned long long counter = sweep * (unsigned long long)(size_x * size_y) * 2 + color * (unsigned long long)(size_x * size_y) + idx;
        float u = rng_uniform_stateless(seed, counter);
        if (u < __expf(-dE * beta))
        {
            lattice[idx] *= -1;
        }
    }
}

// Host function: Perform one full Monte Carlo sweep using checkerboard algorithm
void MH_checkboard_sweep_gpu_2Dblock(int8_t *lattice, unsigned long long seed, int size_x, int size_y, float J, float h, float beta, dim3 grid, dim3 block, int sweep)
{
    if (DEBUG)
        printf("[CHECKPOINT] Starting sweep %d\n", sweep);

    // Update black sites (color 0)
    MH_1color_checkerboard_gpu_2Dblock<<<grid, block>>>(lattice, seed, size_x, size_y, J, h, beta, 0, sweep);

    // Update white sites (color 1)
    MH_1color_checkerboard_gpu_2Dblock<<<grid, block>>>(lattice, seed, size_x, size_y, J, h, beta, 1, sweep);

    if (DEBUG)
        printf("[CHECKPOINT] Sweep %d completed\n", sweep);
}

// Kernel: Compute magnetization using shared memory reduction
__global__ void magnetization_2D_kernel_gpu_2Dblock(
    const int8_t *lattice,
    int *magnetization,
    int N)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    // Grid-stride loop over lattice
    for (int i = global_tid; i < N; i += stride)
        local_sum += lattice[i];

    // Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // One atomic per block
    if (tid == 0)
        atomicAdd(magnetization, sdata[0]);
}

// Host function: Calculate total magnetization of the lattice
int magnetization_2D_gpu_2Dblock(int8_t *d_lattice,
                                 int size_x,
                                 int size_y)
{
    int N = size_x * size_y;

    int *d_mag;
    cudaMalloc(&d_mag, sizeof(int));
    cudaMemset(d_mag, 0, sizeof(int));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    magnetization_2D_kernel_gpu_2Dblock<<<blocks, threads, threads * sizeof(int)>>>(
        d_lattice, d_mag, N);

    int h_mag;
    cudaMemcpy(&h_mag, d_mag, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_mag);

    if (DEBUG)
        printf("[CHECKPOINT] Magnetization calculated: %d\n", h_mag);

    return h_mag;
}

// ###############################################################
// Sanity check of the hot initialized lattice
// ###############################################################

// Kernel: Compute statistics for hot lattice randomness validation
__global__ void hot_lattice_sanity_kernel(
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

    // Magnetization & spin balance
    atomicAdd(magnetization, s);
    atomicAdd(spin_sum, s);

    // Nearest neighbors (right + down, periodic)
    int row = idx / size_x;
    int col = idx % size_x;

    int right = row * size_x + (col + 1) % size_x;
    int down = ((row + 1) % size_y) * size_x + col;

    int corr = s * (lattice[right] + lattice[down]);
    atomicAdd(nn_corr_sum, corr);
}

// Host function: Verify that hot lattice initialization produces random configuration
bool check_hot_lattice_randomness(
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

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hot_lattice_sanity_kernel<<<blocks, threads>>>(
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

    // Thresholds for randomness validation
    double max_M = 2.0 * sqrtN;
    double max_S = 2.0 * sqrtN;
    double max_C = 0.05 * N;

    bool ok_M = fabs((double)M) < max_M;
    bool ok_S = fabs((double)S) < max_S;
    bool ok_C = fabs((double)C) < max_C;

    if (DEBUG)
    {
        printf("[CHECKPOINT] Hot lattice sanity check:\n");
        printf("  Magnetization: %d (threshold: %.2f) - %s\n", M, max_M, ok_M ? "PASS" : "FAIL");
        printf("  Spin sum: %d (threshold: %.2f) - %s\n", S, max_S, ok_S ? "PASS" : "FAIL");
        printf("  NN correlation: %d (threshold: %.2f) - %s\n", C, max_C, ok_C ? "PASS" : "FAIL");
    }

    return ok_M && ok_S && ok_C;
}

// ###############################################################
// HELPER FUNCTIONS
// ###############################################################

// Print lattice (host)
void print_lattice_gpu_2Dblock(const int8_t *lattice, int size_x, int size_y)
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

// Save lattice configuration to file
void save_lattice(const char *folder, int8_t *lattice, int type, int size_x, int size_y, float J, float h, float T, int mc_steps)
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

// ###############################################################
// Main simulation function (without lattice saving)
// ###############################################################

extern "C" Observables run_ising_simulation_2Dblock_gpu(
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

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    if (DEBUG)
        printf("[CHECKPOINT] Simulation started: %dx%d lattice, type=%d, T=%.3f\n",
               lattice_size_x, lattice_size_y, type, T);

    // ============================================================
    // Device allocation
    // ============================================================
    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    cudaDeviceSynchronize();

    // ============================================================
    // Initialization
    // ============================================================
    clock_t t0 = clock();

    if (type == 0)
    {
        // Cold start: all spins +1
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization (all spins +1)\n");
        initialize_lattice_gpu_cold_2Dblock<<<grid, block>>>(
            d_lattice, lattice_size_x, lattice_size_y, +1);
    }
    else if (type == 1)
    {
        // Cold start: all spins -1
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization (all spins -1)\n");
        initialize_lattice_gpu_cold_2Dblock<<<grid, block>>>(
            d_lattice, lattice_size_x, lattice_size_y, -1);
    }
    else
    {
        // Hot start (stateless RNG)
        if (DEBUG)
            printf("[CHECKPOINT] Hot initialization (random spins)\n");
        initialize_lattice_gpu_hot_2Dblock<<<grid, block>>>(
            d_lattice, seed, lattice_size_x, lattice_size_y);

        cudaError_t err_i = cudaGetLastError();
        if (err_i != cudaSuccess)
        {
            printf("[GPU ERROR] Init kernel launch failed: %s\n", cudaGetErrorString(err_i));
            cudaFree(d_lattice);
            memset(&out, 0, sizeof(Observables));
            return out;
        }
    }

    cudaDeviceSynchronize();

    clock_t t1 = clock();
    out.initialization_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    if (DEBUG)
        printf("[CHECKPOINT] Initialization completed in %.6f seconds\n", out.initialization_time);

    // ============================================================
    // Optional sanity check (only for hot start)
    // ============================================================
    if (type == 2 || type == 3)
    {
        bool ok = check_hot_lattice_randomness(
            d_lattice, lattice_size_x, lattice_size_y);

        if (!ok)
        {
            printf("[WARNING] Hot lattice failed randomness check\n");
        }
    }

    // ============================================================
    // Metropolis–Hastings evolution
    // ============================================================
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    if (DEBUG)
        printf("[CHECKPOINT] Starting MC evolution: %d sweeps\n", n_sweeps);

    t0 = clock();

    for (int sweep = 0; sweep < n_sweeps; ++sweep)
    {
        MH_checkboard_sweep_gpu_2Dblock(
            d_lattice,
            seed,
            lattice_size_x,
            lattice_size_y,
            J, h, beta,
            grid, block,
            sweep);
    }

    cudaDeviceSynchronize();

    t1 = clock();
    out.MH_evolution_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    out.MH_evolution_time_over_steps =
        out.MH_evolution_time /
        (double)(n_sweeps * lattice_size_x * lattice_size_y);

    if (DEBUG)
        printf("[CHECKPOINT] MC evolution completed in %.6f seconds\n", out.MH_evolution_time);

    // ============================================================
    // Measurements
    // ============================================================
    if (DEBUG)
        printf("[CHECKPOINT] Starting measurements\n");

    float E = energy_2D_gpu_2Dblock(
        d_lattice, lattice_size_x, lattice_size_y, J, h);

    int M = magnetization_2D_gpu_2Dblock(
        d_lattice, lattice_size_x, lattice_size_y);

    out.E = E;
    out.e_density = E / (float)N;
    out.m = (float)M;
    out.m_density = (float)M / (float)N;

    if (DEBUG)
        printf("[CHECKPOINT] Measurements: E=%.6f, M=%d\n", E, M);

    // ============================================================
    // Cleanup
    // ============================================================
    cudaFree(d_lattice);

    cudaDeviceSynchronize();
    cudaError_t err_o = cudaGetLastError();
    if (err_o != cudaSuccess)
    {
        printf("[GPU WARNING] During cleanup: %s\n", cudaGetErrorString(err_o));
    }

    if (DEBUG)
        printf("[CHECKPOINT] Simulation completed successfully\n");

    return out;
}

// ###############################################################
// Main simulation function (with lattice saving)
// ###############################################################

extern "C" Observables run_ising_simulation_2Dblock_gpu_save(
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

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    if (DEBUG)
        printf("[CHECKPOINT] Simulation with saving started: %dx%d lattice, type=%d, T=%.3f\n",
               lattice_size_x, lattice_size_y, type, T);

    // ============================================================
    // Device allocation
    // ============================================================
    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    cudaDeviceSynchronize();

    // ============================================================
    // Initialization
    // ============================================================
    clock_t t0 = clock();

    if (type == 0)
    {
        // Cold start: all spins +1
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization (all spins +1)\n");
        initialize_lattice_gpu_cold_2Dblock<<<grid, block>>>(
            d_lattice, lattice_size_x, lattice_size_y, +1);
    }
    else if (type == 1)
    {
        // Cold start: all spins -1
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization (all spins -1)\n");
        initialize_lattice_gpu_cold_2Dblock<<<grid, block>>>(
            d_lattice, lattice_size_x, lattice_size_y, -1);
    }
    else
    {
        // Hot start (stateless RNG)
        if (DEBUG)
            printf("[CHECKPOINT] Hot initialization (random spins)\n");
        initialize_lattice_gpu_hot_2Dblock<<<grid, block>>>(
            d_lattice, seed, lattice_size_x, lattice_size_y);

        cudaError_t err_i = cudaGetLastError();
        if (err_i != cudaSuccess)
        {
            printf("[GPU ERROR] Init kernel launch failed: %s\n", cudaGetErrorString(err_i));
            cudaFree(d_lattice);
            memset(&out, 0, sizeof(Observables));
            return out;
        }
    }

    cudaDeviceSynchronize();

    clock_t t1 = clock();
    out.initialization_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    if (DEBUG)
        printf("[CHECKPOINT] Initialization completed in %.6f seconds\n", out.initialization_time);

    // Save initial lattice if requested
    if (save_lattice_flag == 1)
    {
        std::vector<int8_t> h_lattice(N);
        cudaMemcpy(h_lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);
        save_lattice(save_folder, h_lattice.data(), type, lattice_size_x, lattice_size_y, J, h, T, 0);
    }

    // ============================================================
    // Optional sanity check (only for hot start)
    // ============================================================
    if (type == 2 || type == 3)
    {
        bool ok = check_hot_lattice_randomness(
            d_lattice, lattice_size_x, lattice_size_y);

        if (!ok)
        {
            printf("[WARNING] Hot lattice failed randomness check\n");
        }
    }

    // ============================================================
    // Metropolis–Hastings evolution
    // ============================================================
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    if (DEBUG)
        printf("[CHECKPOINT] Starting MC evolution: %d sweeps\n", n_sweeps);

    t0 = clock();

    for (int sweep = 0; sweep < n_sweeps; ++sweep)
    {
        MH_checkboard_sweep_gpu_2Dblock(
            d_lattice,
            seed,
            lattice_size_x,
            lattice_size_y,
            J, h, beta,
            grid, block,
            sweep);
    }

    cudaDeviceSynchronize();

    t1 = clock();
    out.MH_evolution_time = (float)(t1 - t0) / CLOCKS_PER_SEC;

    out.MH_evolution_time_over_steps =
        out.MH_evolution_time /
        (double)(n_sweeps * lattice_size_x * lattice_size_y);

    if (DEBUG)
        printf("[CHECKPOINT] MC evolution completed in %.6f seconds\n", out.MH_evolution_time);

    // ============================================================
    // Measurements
    // ============================================================
    if (DEBUG)
        printf("[CHECKPOINT] Starting measurements\n");

    float E = energy_2D_gpu_2Dblock(
        d_lattice, lattice_size_x, lattice_size_y, J, h);

    int M = magnetization_2D_gpu_2Dblock(
        d_lattice, lattice_size_x, lattice_size_y);

    out.E = E;
    out.e_density = E / (float)N;
    out.m = (float)M;
    out.m_density = (float)M / (float)N;

    if (DEBUG)
        printf("[CHECKPOINT] Measurements: E=%.6f, M=%d\n", E, M);

    // Save final lattice if requested
    if (save_lattice_flag == 1)
    {
        std::vector<int8_t> h_lattice(N);
        cudaMemcpy(h_lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);
        save_lattice(save_folder, h_lattice.data(), type, lattice_size_x, lattice_size_y, J, h, T, n_steps);
    }

    // ============================================================
    // Cleanup
    // ============================================================
    cudaFree(d_lattice);

    cudaDeviceSynchronize();
    cudaError_t err_o = cudaGetLastError();
    if (err_o != cudaSuccess)
    {
        printf("[GPU WARNING] During cleanup: %s\n", cudaGetErrorString(err_o));
    }

    if (DEBUG)
        printf("[CHECKPOINT] Simulation completed successfully\n");

    return out;
}

// ###############################################################
// MAIN (Standalone testing)
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

    float J = 1;
    float h = 1;
    float kB = 1.0;
    float T = 1.0;
    float beta = 1 / T / kB;

    size_t N = lattice_size_x * lattice_size_y;
    size_t lattice_bytes = N * sizeof(int8_t);

    int8_t *d_lattice = nullptr;
    cudaMalloc(&d_lattice, lattice_bytes);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (lattice_size_x + block.x - 1) / block.x,
        (lattice_size_y + block.y - 1) / block.y);

    unsigned long long seed = (unsigned long long)time(NULL);

    printf("Lattice size: %d x %d\n", lattice_size_x, lattice_size_y);

    cudaError_t err;
    int initialization = 1; // 0 for cold, 1 for hot

    if (initialization == 0)
    {
        if (DEBUG)
            printf("[CHECKPOINT] Cold initialization\n");
        initialize_lattice_gpu_cold_2Dblock<<<grid, block>>>(d_lattice, lattice_size_x, lattice_size_y, -1);
    }
    else if (initialization == 1)
    {
        if (DEBUG)
            printf("[CHECKPOINT] Hot initialization\n");
        initialize_lattice_gpu_hot_2Dblock<<<grid, block>>>(d_lattice, seed, lattice_size_x, lattice_size_y);

        bool is_random = check_hot_lattice_randomness(d_lattice, lattice_size_x, lattice_size_y);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if (is_random)
        {
            printf("The randomly generated lattice is effectively random!\n");
        }
        else
        {
            printf("The hot generated lattice is not random.\n");
            return 1;
        }
    }
    else
    {
        printf("Wrong initialization choice (0 or 1).\n");
        return 1;
    }

    float energy = energy_2D_gpu_2Dblock(d_lattice, lattice_size_x, lattice_size_y, J, h);

    std::vector<int8_t> lattice(N, 0);

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU initialization:\n");
    int print_x = min(PRINT_UP_TO, lattice_size_x);
    int print_y = min(PRINT_UP_TO, lattice_size_y);
    print_lattice_gpu_2Dblock(lattice.data(), print_x, print_y);

    printf("Energy: %f\n", energy);

    int M = magnetization_2D_gpu_2Dblock(d_lattice, lattice_size_x, lattice_size_y);

    printf("Magnetization: %d\n", M);

    // MC evolution loop
    if (DEBUG)
        printf("[CHECKPOINT] Starting 10 MC sweeps\n");

    for (int i = 0; i < 10; ++i)
    {
        MH_checkboard_sweep_gpu_2Dblock(d_lattice, seed, lattice_size_x, lattice_size_y, J, h, beta, grid, block, i);
    }

    cudaMemcpy(lattice.data(), d_lattice, lattice_bytes, cudaMemcpyDeviceToHost);

    printf("\nLattice after GPU MH evolution:\n");
    print_lattice_gpu_2Dblock(lattice.data(), print_x, print_y);

    energy = energy_2D_gpu_2Dblock(d_lattice, lattice_size_x, lattice_size_y, J, h);

    printf("Energy: %f\n", energy);

    M = magnetization_2D_gpu_2Dblock(d_lattice, lattice_size_x, lattice_size_y);

    printf("Magnetization: %d\n", M);

    cudaFree(d_lattice);
    return 0;
}
#endif
