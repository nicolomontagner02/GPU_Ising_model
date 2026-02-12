#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "s_functions.h"
#include "ge_functions.h"

void print_header(const char *title)
{
    printf("\n");
    printf("========================================\n");
    printf("%s\n", title);
    printf("========================================\n");
}

int main(int argc, char *argv[])
{
    // Parse command line arguments
    int run_cpu = 1;
    int run_gpu = 1;

    // Simulation parameters
    int lattice_size_x = 1000;
    int lattice_size_y = 1000;
    int type = 3;
    int type_gpu = 2; // random initialization

    float J = 1.0;
    float h = 0.0; // no external field for annealing
    float kB = 1.0;

    float T_initial = 10.0;
    float T_final = 0.5;
    float tau = 5000.0; // decay time constant

    int steps_per_temp_cpu = 100;      // CPU: MC steps at each temperature
    int sweeps_per_temp_gpu = 11;      // GPU: MC sweeps at each temperature
    int temp_update_interval = 100;    // Update temperature interval
    int temp_update_interval_gpu = 10; // GPU: Update temperature interval
    int save_trajectory = 1;

    printf("========================================\n");
    printf("Simulated Annealing Comparison\n");
    printf("========================================\n");
    printf("Lattice size    : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("T_initial       : %.3f\n", T_initial);
    printf("T_final         : %.3f\n", T_final);
    printf("Decay tau       : %.3f\n", tau);
    printf("Initialization  : %s\n",
           type == 0 ? "All +1" : type == 1 ? "All -1"
                                            : "Random");
    printf("\n");

    // ============================================================
    // CPU Annealing
    // ============================================================
    if (run_cpu)
    {
        print_header("Running CPU Annealing");

        clock_t cpu_start = clock();

        int result_cpu = annealing(
            lattice_size_x, lattice_size_y, type,
            J, h, kB,
            T_initial, T_final, tau,
            steps_per_temp_cpu, temp_update_interval,
            save_trajectory);

        clock_t cpu_end = clock();
        double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

        if (result_cpu == 0)
        {
            printf("\nCPU annealing completed successfully\n");
            printf("Total CPU time: %.3f seconds\n", cpu_time);
        }
        else
        {
            printf("\nCPU annealing failed with code %d\n", result_cpu);
        }
    }

    // ============================================================
    // GPU Annealing (simple)
    // ============================================================
    if (run_gpu)
    {
        print_header("Running GPU Annealing");

        clock_t gpu_start = clock();

        int result_gpu = annealing_gpu(
            lattice_size_x, lattice_size_y, type_gpu,
            J, h, kB,
            T_initial, T_final, tau,
            sweeps_per_temp_gpu, temp_update_interval_gpu,
            save_trajectory);

        clock_t gpu_end = clock();
        double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

        if (result_gpu == 0)
        {
            printf("\nGPU annealing completed successfully\n");
            printf("Total GPU time: %.3f seconds\n", gpu_time);
        }
        else
        {
            printf("\nGPU annealing failed with code %d\n", result_gpu);
        }
    }

    printf("\n========================================\n");
    printf("All simulations completed\n");
    printf("========================================\n");

    return 0;
}