#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "s_functions.h"
#include "m_functions.h"
#include "g_functions.h"
#include "ge_functions.h"

/* ------------------------------------------------------------
   Backend wrapper
------------------------------------------------------------ */

typedef Observables (*ising_fn)(
    int, int, int,
    float, float, float, float,
    int);

typedef struct
{
    const char *name;
    ising_fn fn;
} Backend;

/* ------------------------------------------------------------
   Main
------------------------------------------------------------ */

int main(void)
{
    /* ---------------- Parameters ---------------- */

    const int n_repetition = 5;

    int lattice_sizes[] = {
        16, 64, 128, 192, 256, 320, 384, 448, 512,
        640, 768, 896, 1024, 1536, 2048, 2560,
        3072, 3548, 4096};
    int n_L = sizeof(lattice_sizes) / sizeof(int);

    float J_values[] = {1.0f};
    float h_values[] = {0.5f, 1.0f, 2.0f};
    float T_values[] = {0.5f, 2.0f, 10.0f};

    int init_types[] = {1, 3};
    const char *init_names[] = {"all_up", "random"};

    const float kB = 1.0f;
    const int n_steps = 100000;

    /* ---------------- Backends ---------------- */

    Backend backends[] = {
        {"cpu_openmp", run_ising_simulation_openmp},
        {"gpu_efficient", run_ising_simulation_efficient_gpu}};
    int n_backends = sizeof(backends) / sizeof(Backend);

    /* ---------------- CSV file ---------------- */

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", t);

    char csv_name[256];
    snprintf(csv_name, sizeof(csv_name),
             "results/ising_results_%snew.csv", timestamp);

    FILE *csv = fopen(csv_name, "w");
    if (!csv)
    {
        perror("Error opening CSV file");
        return 1;
    }

    fprintf(csv,
            "backend,L,init_type,J,h,T,n_steps,"
            "E,e_density,m,m_density,"
            "init_time,mh_time,mh_time_per_step\n");

    /* ---------------- Sweep ---------------- */

    for (int iL = 0; iL < n_L; iL++)
    {
        int L = lattice_sizes[iL];

        for (int it = 0; it < 2; it++)
        {
            int type_id = init_types[it];
            const char *type_name = init_names[it];

            for (int iJ = 0; iJ < 1; iJ++)
            {
                float J = J_values[iJ];

                for (int ih = 0; ih < 3; ih++)
                {
                    float h = h_values[ih];

                    for (int iT = 0; iT < 3; iT++)
                    {
                        float T = T_values[iT];

                        for (int ib = 0; ib < n_backends; ib++)
                        {
                            Backend *b = &backends[ib];

                            printf("\n=== Running backend: %s ===\n", b->name);
                            printf(
                                "Backend=%-14s L=%4d init=%-9s "
                                "J=%.2f h=%.2f T=%.2f\n",
                                b->name, L, type_name, J, h, T);

                            for (int r = 0; r < n_repetition; r++)
                            {

                                Observables obs =
                                    b->fn(L, L, type_id,
                                          J, h, kB, T,
                                          n_steps);

                                fprintf(csv,
                                        "%s,%d,%s,%.3f,%.3f,%.3f,%d,"
                                        "%f,%f,%f,%f,%f,%f,%e\n",
                                        b->name,
                                        L,
                                        type_name,
                                        J,
                                        h,
                                        T,
                                        n_steps,
                                        obs.E,
                                        obs.e_density,
                                        obs.m,
                                        obs.m_density,
                                        obs.initialization_time,
                                        obs.MH_evolution_time,
                                        obs.MH_evolution_time_over_steps);
                            }
                        }
                    }
                }
            }
        }
    }

    fclose(csv);
    printf("\nResults written to %s\n", csv_name);

    return 0;
}