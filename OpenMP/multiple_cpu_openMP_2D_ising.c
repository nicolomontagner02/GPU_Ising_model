#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../functions.h"

#define DEBUG 0

// void print_lattice(int **lattice, int size_x, int size_y){

//     printf("2D Ising Lattice:\n");
//         for (int i = 0; i < size_x; i++) {
//             for (int j = 0; j < size_y; j++) {
//                 printf("%d ", lattice[i][j]);
//             }
//             printf("\n");
//         }

// }

int **initialize_lattice_openmp(int lattice_size_x, int lattice_size_y, int type)
{
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++) {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

    #pragma omp parallel
    {
        unsigned int seed = 1234 + omp_get_thread_num();

        #pragma omp for collapse(2)
        for (int i = 0; i < lattice_size_x; i++) {
            for (int j = 0; j < lattice_size_y; j++) {

                if (type == 1) {
                    lattice[i][j] = 1;
                }
                else if (type == 2) {
                    lattice[i][j] = -1;
                }
                else if (type == 3) {
                    lattice[i][j] = (rand_r(&seed) & 1) ? 1 : -1;
                }
            }
        }
    }

    return lattice;
}

float energy_2D_openmp(int **lattice, int size_x, int size_y, float J, float h) {
    float energy = 0.0;

    #pragma omp parallel for reduction(+:energy) collapse(2)

    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            
            int spin = lattice[i][j];
            // each spin interacts with the right one and the bottom one (boundary conditions % size for edges sites)
            int right_neighbor = lattice[i][(j+1) % size_y];
            int bottom_neighbor = lattice[(i+1) % size_x][j];

            if (DEBUG){
                printf("Right neighbor spin of (%i,%i): %i\n",i,j, right_neighbor);
                printf("Bottom neighbor spin of (%i,%i): %i\n",i,j, bottom_neighbor);
            }

            energy -= J * spin * (right_neighbor + bottom_neighbor);
            energy -= h * spin;
        }
    }
    return energy;
}

int magnetisation_2D_openmp(int **lattice, int size_x, int size_y){

    int magnetisation = 0;

    #pragma omp parallel for reduction(+:magnetisation) collapse(2)

    for (int i = 0; i < size_x; i++){
        for (int j = 0; j < size_y; j++){
            magnetisation += lattice[i][j];
        }
    }

    return magnetisation ;
}

// float d_energy_2D(int **lattice, int i, int j, int size_x, int size_y,float J, float h){
//     int spin = lattice[i][j];
//     float sum_nn = 0.0f;

//     if (i > 0)           sum_nn += lattice[i-1][j];
//     if (i < size_x - 1)  sum_nn += lattice[i+1][j];
//     if (j > 0)           sum_nn += lattice[i][j-1];
//     if (j < size_y - 1)  sum_nn += lattice[i][j+1];

//     return 2.0f * spin * (J * sum_nn + h);
// }

// float energy_density_2D(float energy, int size_x, int size_y){

//     int N = size_x*size_y;

//     float e_density = energy / N;

//     return e_density;
// }

void MH_sweep_checkerboard_openmp(int **lattice,
                           int size_x, int size_y,
                           float J, float h,
                           float kB, float T)
{
    for (int color = 0; color < 2; color++) {

        #pragma omp parallel
        {
            unsigned int seed = 5678 + omp_get_thread_num();

            #pragma omp for collapse(2)
            for (int i = 0; i < size_x; i++) {
                for (int j = 0; j < size_y; j++) {

                    if ((i + j) % 2 != color) continue;

                    float dE = d_energy_2D(lattice, i, j,
                                           size_x, size_y, J, h);

                    int accept = 0;

                    if (dE <= 0.0f) {
                        accept = 1;
                    } else {
                        float u = (float)rand_r(&seed) / (float)RAND_MAX;
                        if (u < expf(-dE / (kB * T)))
                            accept = 1;
                    }

                    if (accept) {
                        lattice[i][j] *= -1;
                    }
                }
            }
        } /* implicit barrier between colors */
    }
}

// void save_lattice(const char *folder, int **lattice, int size_x, int size_y, float J, float h, float T, int mc_steps){
    
//     char filename[512];

//     snprintf(filename, sizeof(filename),
//              "%s/ising_J%.3f_h%.3f_T%.3f_Lx%d_Ly%d_MC%d.dat",
//              folder, J, h, T, size_x, size_y, mc_steps);

//     FILE *fp = fopen(filename, "w");
//     if (fp == NULL) {
//         perror("Error opening file for lattice save");
//         return;
//     }

//     for (int i = 0; i < size_x; i++) {
//         for (int j = 0; j < size_y; j++) {
//             fprintf(fp, "%d ", lattice[i][j]);
//         }
//         fprintf(fp, "\n");
//     }

//     fclose(fp);

//     printf("Lattice saved to %s\n", filename);
// }

// void report_state(const char *label, int **lattice, int size_x, int size_y, float J, float h){

//     float E = energy_2D(lattice, size_x, size_y, J, h);
//     float e_density = energy_density_2D(E, size_x, size_y);
//     int m = magnetisation_2D(lattice, size_x, size_y);

//     int N = size_x * size_y;

//     printf("----------------------------------------\n");
//     printf("%s\n", label);
//     printf("----------------------------------------\n");
//     printf("Total energy        : %f\n", E);
//     printf("Energy density      : %f\n", e_density);
//     printf("Magnetisation       : %f\n", m);
//     printf("Magnetisation/spin  : %f\n",(float) m /(float) N);
    
//     if (size_x <= 4 && size_y <= 4) {
//         print_lattice(lattice, size_x, size_y);
//     }
// }

// int test(){

//     // Parse command line arguments
//     int lattice_size_x = 10;
//     int lattice_size_y = 10;

//     int type = 3; // 1 for all spin up, 2 for all spin down, 3 for random initialization of the lattice

//     float J = 1.0; // Interaction strength
//     float h = 1.0 ; // External magnetic field
//     float kB = 1.0*exp(-23);
//     float T = 100.0;
//     int n_steps = 1000000;

//     // simulation header
//     printf("========================================\n");
//     printf("2D Ising Model — Metropolis Simulation\n");
//     printf("========================================\n");
//     printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
//     printf("Interaction J       : %.3f\n", J);
//     printf("External field h    : %.3f\n", h);
//     printf("Temperature T       : %.3f\n", T);
//     printf("MC steps            : %d\n", n_steps);
//     printf("Initialization type : %s\n",
//         type == 1 ? "All up" :
//         type == 2 ? "All down" : "Random");
//     printf("\n");

//     int **lattice = initialize_lattice_openmp(lattice_size_x, lattice_size_y, type);

//     report_state("Initial state",lattice, lattice_size_x, lattice_size_y,J, h);
//     save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, 0);

//     for (int step = 0; step < n_steps; step++) {
//     MH_sweep_checkerboard_openmp(lattice,
//                           lattice_size_x, lattice_size_y,
//                           J, h, kB, T);
//     }

//     report_state("Final state", lattice, lattice_size_x, lattice_size_y, J, h);
//     save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, n_steps);

//     // free the memory
//     for (int i = 0; i < lattice_size_x; i++) {
//         free(lattice[i]);
//     }
//     free(lattice);

//     return 0;

// }

// typedef struct {
//     float E;
//     float e_density;
//     float m;
//     float m_density;
//     float initialization_time;
//     float MH_evolution_time;
//     float MH_evolution_time_over_steps;
// } Observables;

Observables run_ising_simulation_openmp(int lattice_size_x, int lattice_size_y,
                                 int type,
                                 float J, float h, float kB, float T,
                                 int n_steps)
{
    clock_t t0, t1;

    /* --- Initialization timing --- */
    t0 = clock();
    int **lattice = initialize_lattice_openmp(lattice_size_x, lattice_size_y, type);
    t1 = clock();

    float initialization_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    /* --- Metropolis evolution timing --- */
    t0 = clock();
    for (int step = 0; step < n_steps; step++) {
    MH_sweep_checkerboard_openmp(lattice,
                          lattice_size_x, lattice_size_y,
                          J, h, kB, T);
    }

    t1 = clock();

    float MH_evolution_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    /* --- Measurements --- */
    float E = energy_2D_openmp(lattice, lattice_size_x, lattice_size_y, J, h);
    float e_density = energy_density_2D(E, lattice_size_x, lattice_size_y);
    float m = magnetisation_2D_openmp(lattice, lattice_size_x, lattice_size_y);
    float m_density = m / lattice_size_x / lattice_size_y;

    /* --- Cleanup --- */
    for (int i = 0; i < lattice_size_x; i++) {
        free(lattice[i]);
    }
    free(lattice);

    /* --- Output struct --- */
    Observables out;
    out.E = E;
    out.e_density = e_density;
    out.m = m;
    out.m_density = m_density;
    out.initialization_time = initialization_time;
    out.MH_evolution_time = MH_evolution_time;
    out.MH_evolution_time_over_steps =
        MH_evolution_time / (float)n_steps / lattice_size_x / lattice_size_y;

    return out;
}

// int main(int argc, char *argv[]) {

//     // Check command line arguments
//     if (argc != 3) {
//         printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
//         printf("Example: %s 10 10\n", argv[0]);
//         return 1;
//     }

//     // Parse command line arguments
//     int lattice_size_x = atoi(argv[1]);
//     int lattice_size_y = atoi(argv[2]);

//     int type = 3;
//     float J = 1;
//     float h = 1;
//     float kB = 1.0*exp(-23);
//     float T = 100;
//     int n_steps = 1000000;
//     int n_sweeps = (int) (n_steps / lattice_size_x / lattice_size_y);

//     printf("========================================\n");
//     printf("2D Ising Model — Metropolis Simulation\n");
//     printf("========================================\n");
//     printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
//     printf("Interaction J       : %.3f\n", J);
//     printf("External field h    : %.3f\n", h);
//     printf("Temperature T       : %.3f\n", T);
//     printf("MC steps            : %d\n", n_sweeps);
//     printf("Initialization type : %s\n",
//         type == 1 ? "All up" :
//         type == 2 ? "All down" : "Random");
//     printf("\n");
//     printf("========================================\n");

//     Observables out = run_ising_simulation(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_sweeps);

//     printf("Energy                : %f\n", out.E);
//     printf("Energy density        : %f\n", out.e_density);
//     printf("Magnetization         : %f\n", out.m);
//     printf("Magnetization density : %f\n", out.m_density);
//     printf("Initialization time (s)        : %f\n", out.initialization_time);
//     printf("MH evolution time (s)          : %f\n", out.MH_evolution_time);
//     printf("MH time per step (s)           : %e\n", out.MH_evolution_time_over_steps);

//     return 0;
// }