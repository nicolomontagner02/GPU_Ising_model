#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "s_functions.h"
#include "m_functions.h"
#include "g_functions.h"
// #include "ge_functions.h"

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
    float J = 0;
    float h = 1;
    float kB = 1.0 * exp(-23);
    float T = 100;
    int n_steps = 100000;
    int n_sweeps = (int)(n_steps / lattice_size_x / lattice_size_y);
    n_sweeps = fmax(1, n_sweeps);

    printf("========================================\n");
    printf("2D Ising Model — 1 CPU\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("MC steps            : %d\n", n_steps);
    printf("Initialization type : %s\n",
           type == 1 ? "All up" : type == 2 ? "All down"
                                            : "Random");
    printf("\n");
    printf("========================================\n");

    Observables out = run_ising_simulation(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_steps);

    printf("Energy                : %f\n", out.E);
    printf("Energy density        : %f\n", out.e_density);
    printf("Magnetization         : %f\n", out.m);
    printf("Magnetization density : %f\n", out.m_density);
    printf("Initialization time (s)        : %f\n", out.initialization_time);
    printf("MH evolution time (s)          : %f\n", out.MH_evolution_time);
    printf("MH time per step (s)           : %e\n", out.MH_evolution_time_over_steps);
    printf("\n");

    // OpenMP part

    printf("========================================\n");
    printf("2D Ising Model — 4 CPU\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("# sweeps            : %d\n", n_sweeps);
    printf("Initialization type : %s\n",
           type == 1 ? "All up" : type == 2 ? "All down"
                                            : "Random");
    printf("\n");
    printf("========================================\n");

    Observables out1 = run_ising_simulation_openmp(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_sweeps);

    printf("Energy                : %f\n", out1.E);
    printf("Energy density        : %f\n", out1.e_density);
    printf("Magnetization         : %f\n", out1.m);
    printf("Magnetization density : %f\n", out1.m_density);
    printf("Initialization time (s)        : %f\n", out1.initialization_time);
    printf("MH evolution time (s)          : %f\n", out1.MH_evolution_time);
    printf("MH time per step (s)           : %e\n", out1.MH_evolution_time_over_steps);

    // cuda part 1st

    printf("========================================\n");
    printf("2D Ising Model — GPU\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("# sweeps            : %d\n", n_sweeps);
    printf("Initialization type : %s\n",
           type == 1 ? "All up" : type == 2 ? "All down"
                                            : "Random");
    printf("\n");
    printf("========================================\n");

    Observables out2 = run_ising_simulation_gpu(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_sweeps);

    printf("Energy                : %f\n", out2.E);
    printf("Energy density        : %f\n", out2.e_density);
    printf("Magnetization         : %f\n", out2.m);
    printf("Magnetization density : %f\n", out2.m_density);
    printf("Initialization time (s)        : %f\n", out2.initialization_time);
    printf("MH evolution time (s)          : %f\n", out2.MH_evolution_time);
    printf("MH time per step (s)           : %e\n", out2.MH_evolution_time_over_steps);

    return 0;
}