/*
 * 2D Ising Model Simulation using OpenMP
 *
 * This program implements a 2D Ising model simulation using the Metropolis-Hastings
 * algorithm with a checkerboard update scheme. The simulation uses OpenMP for
 * parallelization across multiple CPU cores.
 *
 * Key features:
 * - Parallel lattice initialization
 * - Energy and magnetization calculations with reduction
 * - Checkerboard Metropolis-Hastings sweeps for efficient updates
 * - Performance timing and measurements
 * - Debug mode for checkpoint verification
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../s_functions.h"

#define DEBUG 0

// Initialize a 2D lattice with spins
// type: 1 (all up), 2 (all down), 3 (random)
int **initialize_lattice_openmp(int lattice_size_x, int lattice_size_y, int type)
{
    // Allocate memory for lattice pointers
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++)
    {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

#pragma omp parallel
    {
        unsigned int seed = 1234 + omp_get_thread_num();

#pragma omp for collapse(2)
        for (int i = 0; i < lattice_size_x; i++)
        {
            for (int j = 0; j < lattice_size_y; j++)
            {
                // Initialize spins based on type
                if (type == 1)
                {
                    lattice[i][j] = 1;
                }
                else if (type == 2)
                {
                    lattice[i][j] = -1;
                }
                else if (type == 3)
                {
                    lattice[i][j] = (rand_r(&seed) & 1) ? 1 : -1;
                }
            }
        }
    }

    if (DEBUG)
        printf("[CHECKPOINT] Lattice initialization complete: %d x %d\n",
               lattice_size_x, lattice_size_y);

    return lattice;
}

// Calculate total energy of the 2D Ising model with periodic boundary conditions
// Energy formula: E = -J * sum(S_i * S_j) - h * sum(S_i)
float energy_2D_openmp(int **lattice, int size_x, int size_y, float J, float h)
{
    float energy = 0.0;

#pragma omp parallel for reduction(+ : energy) collapse(2)
    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            int spin = lattice[i][j];
            // Each spin interacts with the right one and the bottom one (periodic BC)
            int right_neighbor = lattice[i][(j + 1) % size_y];
            int bottom_neighbor = lattice[(i + 1) % size_x][j];

            if (DEBUG && i < 2 && j < 2)
            {
                printf("[CHECKPOINT] Energy calc at (%d,%d): spin=%d, right=%d, bottom=%d\n",
                       i, j, spin, right_neighbor, bottom_neighbor);
            }

            energy -= J * spin * (right_neighbor + bottom_neighbor);
            energy -= h * spin;
        }
    }

    if (DEBUG)
        printf("[CHECKPOINT] Total energy calculated: %f\n", energy);

    return energy;
}

// Calculate total magnetization (sum of all spins)
int magnetisation_2D_openmp(int **lattice, int size_x, int size_y)
{
    int magnetisation = 0;

#pragma omp parallel for reduction(+ : magnetisation) collapse(2)
    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            magnetisation += lattice[i][j];
        }
    }

    if (DEBUG)
        printf("[CHECKPOINT] Total magnetization calculated: %d\n", magnetisation);

    return magnetisation;
}

// Perform one Metropolis-Hastings sweep using checkerboard decomposition
// This ensures no race conditions in parallel updates
void MH_sweep_checkerboard_openmp(int **lattice,
                                  int size_x, int size_y,
                                  float J, float h,
                                  float kB, float T)
{
    // Process two colors sequentially (checkerboard pattern)
    for (int color = 0; color < 2; color++)
    {
        if (DEBUG)
            printf("[CHECKPOINT] Starting color %d sweep\n", color);

#pragma omp parallel
        {
            unsigned int seed = 5678 + omp_get_thread_num();

#pragma omp for collapse(2)
            for (int i = 0; i < size_x; i++)
            {
                for (int j = 0; j < size_y; j++)
                {
                    // Skip sites not matching current color
                    if ((i + j) % 2 != color)
                        continue;

                    // Calculate energy change if spin is flipped
                    float dE = d_energy_2D(lattice, i, j,
                                           size_x, size_y, J, h);

                    int accept = 0;

                    // Metropolis acceptance criterion
                    if (dE <= 0.0f)
                    {
                        accept = 1;
                    }
                    else
                    {
                        float u = (float)rand_r(&seed) / (float)RAND_MAX;
                        if (u < expf(-dE / (kB * T)))
                            accept = 1;
                    }

                    // Update spin if move is accepted
                    if (accept)
                    {
                        lattice[i][j] *= -1;
                        if (DEBUG && i < 2 && j < 2)
                            printf("[CHECKPOINT] Spin flipped at (%d,%d), dE=%f\n",
                                   i, j, dE);
                    }
                }
            }
        } /* implicit barrier between colors */
    }
}

// Main simulation function: initialize and run Ising simulation
Observables run_ising_simulation_openmp(int lattice_size_x, int lattice_size_y,
                                        int type,
                                        float J, float h, float kB, float T,
                                        int n_steps)
{
    clock_t t0, t1;

    if (DEBUG)
        printf("[CHECKPOINT] Starting simulation with lattice %d x %d\n",
               lattice_size_x, lattice_size_y);

    /* --- Initialization timing --- */
    t0 = clock();
    int **lattice = initialize_lattice_openmp(lattice_size_x, lattice_size_y, type);
    t1 = clock();

    float initialization_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    /* --- Metropolis evolution timing --- */
    int n_sweeps = (int)n_steps / lattice_size_x / lattice_size_y;
    n_sweeps = fmax(1, n_sweeps);

    if (DEBUG)
        printf("[CHECKPOINT] Beginning %d MH sweeps\n", n_sweeps);

    t0 = clock();
    for (int step = 0; step < n_sweeps; step++)
    {
        MH_sweep_checkerboard_openmp(lattice,
                                     lattice_size_x, lattice_size_y,
                                     J, h, kB, T);
        if (DEBUG && step % 100 == 0)
            printf("[CHECKPOINT] Completed sweep %d/%d\n", step, n_sweeps);
    }

    t1 = clock();

    float MH_evolution_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    if (DEBUG)
        printf("[CHECKPOINT] MH evolution complete in %f seconds\n", MH_evolution_time);

    /* --- Measurements --- */
    float E = energy_2D_openmp(lattice, lattice_size_x, lattice_size_y, J, h);
    float e_density = energy_density_2D(E, lattice_size_x, lattice_size_y);
    float m = magnetisation_2D_openmp(lattice, lattice_size_x, lattice_size_y);
    float m_density = m / lattice_size_x / lattice_size_y;

    /* --- Cleanup --- */
    for (int i = 0; i < lattice_size_x; i++)
    {
        free(lattice[i]);
    }
    free(lattice);

    if (DEBUG)
        printf("[CHECKPOINT] Simulation cleanup complete\n");

    /* --- Output struct --- */
    Observables out;
    out.E = E;
    out.e_density = e_density;
    out.m = m;
    out.m_density = m_density;
    out.initialization_time = initialization_time;
    out.MH_evolution_time = MH_evolution_time;
    out.MH_evolution_time_over_steps =
        MH_evolution_time / (float)n_sweeps / lattice_size_x / lattice_size_y;

    return out;
}

#ifdef STANDALONE_BUILD
int main(int argc, char *argv[])
{
    // Check command line arguments
    if (argc != 3)
    {
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
    float kB = 1.0 * exp(-23);
    float T = 100;
    int n_steps = 1000000;
    int n_sweeps = (int)(n_steps / lattice_size_x / lattice_size_y);

    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("MC steps            : %d\n", n_sweeps);
    printf("Initialization type : %s\n",
           type == 1 ? "All up" : type == 2 ? "All down"
                                            : "Random");
    printf("\n");
    printf("========================================\n");

    Observables out = run_ising_simulation_openmp(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_sweeps);

    printf("Energy                : %f\n", out.E);
    printf("Energy density        : %f\n", out.e_density);
    printf("Magnetization         : %f\n", out.m);
    printf("Magnetization density : %f\n", out.m_density);
    printf("Initialization time (s)        : %f\n", out.initialization_time);
    printf("MH evolution time (s)          : %f\n", out.MH_evolution_time);
    printf("MH time per step (s)           : %e\n", out.MH_evolution_time_over_steps);

    return 0;
}
#endif