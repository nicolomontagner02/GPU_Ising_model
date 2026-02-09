#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "s_functions.h"
#include "ge_functions.h"

#define DEBUG 0

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

    int save = 0;
    int count = 0;

    float kB = 1.0;
    int n_steps = 10000000;
    int n_sweeps = (int)(n_steps / lattice_size_x / lattice_size_y);
    n_sweeps = fmax(1, n_sweeps);

    int T_span[] = {1, 100};
    float J_s = 0.0;
    float J_e = 3.0;
    int J_n = 50;
    float J_span[J_n];

    float dJ = (J_e - J_s) / (J_n - 1);
    for (int i = 0; i < J_n; i++)
    {
        J_span[i] = J_s + i * dJ;
    }

    float h_s = 0.0;
    float h_e = 3.0;
    int h_n = 50;
    float h_span[h_n];

    float dh = (h_e - h_s) / (h_n - 1);
    for (int i = 0; i < h_n; i++)
    {
        h_span[i] = h_s + i * dh;
    }

    for (int ty = 1; ty < 4; ty += 2)
    {
        int type = ty; // 1 for all spin up, 2 for all spin down, 3 for random initialization of the lattice

        for (int t = 0; t < sizeof(T_span) / sizeof(T_span[0]); t++)
        {
            float T = T_span[t];
            float magnetization[J_n][h_n];
            printf("Magnetization initialized for T=%.3f, type=%d\n", T, type);
            for (int j = 0; j < J_n; j++)
            {
                float J = J_span[j];

                for (int h_i = 0; h_i < h_n; h_i++)
                {
                    float h = h_span[h_i];
                    printf("Running simulation with T=%.3f, J=%.3f, h=%.3f, type=%d\n", T, J, h, type);

		    if (count %7 == 0){
			save = 1;
                        count = 0;
		    }

                    Observables out = run_ising_simulation_efficient_gpu_save(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_steps, save, "data");
                    magnetization[j][h_i] = out.m_density;

		    count++;
		    save = 0;

                }
            }

            if (DEBUG == 1)
            {
                // Print magnetization data to console
                printf("Magnetization density data for T=%.3f, type=%d:\n", T, type);
                for (int j = 0; j < J_n; j++)
                {
                    for (int h_i = 0; h_i < h_n; h_i++)
                    {
                        printf("J=%.3f, h=%.3f, magnetization_density   : %.6f\n", J_span[j], h_span[h_i], magnetization[j][h_i]);
                    }
                }
            }

            // Store magnetization data to file
            char filename[256];
            snprintf(filename, sizeof(filename), "data/magnetization%i_%i_T%.3f_type%d.csv", lattice_size_x, lattice_size_y, T, type);
            FILE *file = fopen(filename, "w");
            if (file == NULL)
            {
                fprintf(stderr, "Error opening file %s for writing\n", filename);
                return 1;
            }
            fprintf(file, "J,h,magnetization_density\n");
            for (int j = 0; j < J_n; j++)
            {
                for (int h_i = 0; h_i < h_n; h_i++)
                {
                    fprintf(file, "%.3f,%.3f,%.6f\n", J_span[j], h_span[h_i], magnetization[j][h_i]);
                }
            }
            fclose(file);
            printf("Magnetization data saved to %s\n", filename);
        }
    }
    return 0;
}
