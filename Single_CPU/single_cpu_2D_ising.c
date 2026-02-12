/*
 * 2D Ising Model Metropolis Simulation
 *
 * This code implements a Monte Carlo simulation of the 2D Ising model using the
 * Metropolis Hastings algorithm. The Ising model is a mathematical model of ferromagnetism
 * in statistical mechanics, where spins on a 2D lattice interact with their nearest
 * neighbors and an external magnetic field.
 *
 * Key features:
 * - Lattice initialization (all up, all down, or random)
 * - Energy and magnetization calculations
 * - Metropolis-Hastings spin flip dynamics
 * - Performance timing and observable measurements
 * - Optional DEBUG mode for detailed step-by-step tracking
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DEBUG 0

/* Print the current state of the 2D lattice */
void print_lattice(int **lattice, int size_x, int size_y)
{
    printf("2D Ising Lattice:\n");
    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            printf("%d ", lattice[i][j]);
        }
        printf("\n");
    }
}

/*
 * Initialize the lattice with spins based on the initialization type
 * type=1: all spins up (+1)
 * type=2: all spins down (-1)
 * type=3: random spins (+1 or -1)
 */
int **initialize_lattice(int lattice_size_x, int lattice_size_y, int type)
{
    if (DEBUG)
        printf("[CHECKPOINT] Initializing lattice (%d x %d) with type %d\n",
               lattice_size_x, lattice_size_y, type);

    // Allocate memory for the lattice
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++)
    {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

    switch (type)
    {
    // Switch for different initializations cases
    case 1:
        // Initialize to all spins up (+1)
        for (int i = 0; i < lattice_size_x; i++)
        {
            for (int j = 0; j < lattice_size_y; j++)
            {
                lattice[i][j] = 1;
            }
        }
        if (DEBUG)
            printf("[CHECKPOINT] All spins initialized to +1\n");
        break;

    case 2:
        // Initialize to all spins down (-1)
        for (int i = 0; i < lattice_size_x; i++)
        {
            for (int j = 0; j < lattice_size_y; j++)
            {
                lattice[i][j] = -1;
            }
        }
        if (DEBUG)
            printf("[CHECKPOINT] All spins initialized to -1\n");
        break;

    case 3:
        // Initialize to random spins (+1 or -1)
        srand(2); // Seed for reproducibility
        for (int i = 0; i < lattice_size_x; i++)
        {
            for (int j = 0; j < lattice_size_y; j++)
            {
                lattice[i][j] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
            }
        }
        if (DEBUG)
            printf("[CHECKPOINT] Random spins initialized\n");
        break;

    default:
        printf("Wrong initialization case submitted!\n");
        break;
    }

    // print the lattice (only small size)
    if (lattice_size_x <= 4 && lattice_size_y <= 4)
    {
        print_lattice(lattice, lattice_size_x, lattice_size_y);
    }

    return lattice;
}

/*
 * Calculate the total energy of the 2D lattice
 * Uses periodic boundary conditions
 */
float energy_2D(int **lattice, int size_x, int size_y, float J, float h)
{
    if (DEBUG)
        printf("[CHECKPOINT] Computing total energy\n");

    float energy = 0.0;
    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            int spin = lattice[i][j];
            // Each spin interacts with the right and bottom neighbors (periodic boundary conditions)
            int right_neighbor = lattice[i][(j + 1) % size_y];
            int bottom_neighbor = lattice[(i + 1) % size_x][j];

            if (DEBUG)
            {
                printf("Right neighbor spin of (%i,%i): %i\n", i, j, right_neighbor);
                printf("Bottom neighbor spin of (%i,%i): %i\n", i, j, bottom_neighbor);
            }

            energy -= J * spin * (right_neighbor + bottom_neighbor);
            energy -= h * spin;
        }
    }
    if (DEBUG)
        printf("[CHECKPOINT] Total energy computed: %f\n", energy);
    return energy;
}

/* Calculate the total magnetization of the lattice */
int magnetisation_2D(int **lattice, int size_x, int size_y)
{
    if (DEBUG)
        printf("[CHECKPOINT] Computing magnetization\n");

    int magnetisation = 0;

    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            magnetisation += lattice[i][j];
        }
    }

    if (DEBUG)
        printf("[CHECKPOINT] Magnetization computed: %d\n", magnetisation);
    return magnetisation;
}

/*
 * Calculate the energy change if spin at position (i,j) is flipped
 * Used in Metropolis algorithm to determine acceptance probability
 */
float d_energy_2D(int **lattice, int i, int j, int size_x, int size_y, float J, float h)
{
    int spin = lattice[i][j];
    float sum_nn = 0.0f;

    // Calculate indices with periodic boundary conditions
    int ip = (i + 1) % size_x;
    int im = (i - 1 + size_x) % size_x;
    int jp = (j + 1) % size_y;
    int jm = (j - 1 + size_y) % size_y;

    // Sum all four nearest neighbors
    sum_nn += lattice[im][j];
    sum_nn += lattice[ip][j];
    sum_nn += lattice[i][jm];
    sum_nn += lattice[i][jp];

    return 2.0f * spin * (J * sum_nn + h);
}

/* Calculate the energy density (energy per spin) */
float energy_density_2D(float energy, int size_x, int size_y)
{
    int N = size_x * size_y;
    float e_density = energy / N;
    return e_density;
}

/*
 * Perform one Metropolis-Hastings step: propose a spin flip and accept/reject
 * based on the energy change and temperature
 */
void MH_step(int **lattice, int size_x, int size_y, float J, float h, float kB, float T)
{
    // Randomly select a spin to flip
    int i_s = rand() % size_x;
    int j_s = rand() % size_y;

    if (DEBUG)
    {
        printf("[CHECKPOINT] Proposed flip at (%d, %d)\n", i_s, j_s);
    }

    // Calculate energy change for the proposed flip
    float d_energy = d_energy_2D(lattice, i_s, j_s,
                                 size_x, size_y, J, h);

    int accept = 0;

    // Accept if energy decreases, or with probability exp(-dE/kT) if it increases
    if (d_energy <= 0.0f)
    {
        accept = 1;
    }
    else
    {
        float p = exp(-d_energy / (kB * T));
        float u = (float)rand() / (float)RAND_MAX;
        if (u < p)
        {
            accept = 1;
        }
    }

    // Apply the flip if accepted
    if (accept)
    {
        lattice[i_s][j_s] *= -1;
        if (DEBUG)
        {
            printf("[CHECKPOINT] Move accepted (ΔE = %.4f)\n", d_energy);
        }
    }
    else if (DEBUG)
    {
        printf("[CHECKPOINT] Move rejected (ΔE = %.4f)\n", d_energy);
    }
}

/* Print current observables (energy, magnetization) */
void report_state(const char *label, int **lattice, int size_x, int size_y, float J, float h)
{
    if (DEBUG)
        printf("[CHECKPOINT] Reporting state: %s\n", label);

    float E = energy_2D(lattice, size_x, size_y, J, h);
    float e_density = energy_density_2D(E, size_x, size_y);
    int m = magnetisation_2D(lattice, size_x, size_y);

    int N = size_x * size_y;

    printf("----------------------------------------\n");
    printf("%s\n", label);
    printf("----------------------------------------\n");
    printf("Total energy        : %f\n", E);
    printf("Energy density      : %f\n", e_density);
    printf("Magnetisation       : %i\n", m);
    printf("Magnetisation/spin  : %f\n", (float)m / (float)N);

    if (size_x <= 4 && size_y <= 4)
    {
        print_lattice(lattice, size_x, size_y);
    }
}

/* Save the final lattice configuration to a file */
void save_lattice(const char *folder, int **lattice, int size_x, int size_y, float J, float h, float T, int mc_steps)
{
    if (DEBUG)
        printf("[CHECKPOINT] Saving lattice to file\n");

    char filename[512];

    snprintf(filename, sizeof(filename),
             "%s/ising_J%.3f_h%.3f_T%.3f_Lx%d_Ly%d_MC%d.dat",
             folder, J, h, T, size_x, size_y, mc_steps);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        perror("Error opening file for lattice save");
        return;
    }

    // Write lattice to file
    for (int i = 0; i < size_x; i++)
    {
        for (int j = 0; j < size_y; j++)
        {
            fprintf(fp, "%d ", lattice[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Lattice saved to %s\n", filename);
    if (DEBUG)
        printf("[CHECKPOINT] Lattice save complete\n");
}

int test()
{
    if (DEBUG)
        printf("[CHECKPOINT] Starting test() function\n");

    // Parse command line arguments
    int lattice_size_x = 10;
    int lattice_size_y = 10;

    int type = 3; // 1 for all spin up, 2 for all spin down, 3 for random initialization

    float J = 1.0; // Interaction strength
    float h = 1.0; // External magnetic field
    float kB = 1.0 * exp(-23);
    float T = 100.0;       // Temperature
    int n_steps = 1000000; // Number of Monte Carlo steps

    // Simulation header
    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
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

    if (DEBUG)
        printf("[CHECKPOINT] Parameters set, initializing lattice\n");

    int **lattice = initialize_lattice(lattice_size_x, lattice_size_y, type);

    report_state("Initial state", lattice, lattice_size_x, lattice_size_y, J, h);
    save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, 0);

    if (DEBUG)
        printf("[CHECKPOINT] Starting %d MC steps\n", n_steps);

    for (int i = 0; i < n_steps; i++)
    {
        MH_step(lattice, lattice_size_x, lattice_size_y, J, h, kB, T);
        if (DEBUG && i % 100000 == 0)
        {
            printf("[CHECKPOINT] MH step %i / %i executed\n", i, n_steps);
        }
    }

    if (DEBUG)
        printf("[CHECKPOINT] MC evolution complete\n");

    report_state("Final state", lattice, lattice_size_x, lattice_size_y, J, h);
    save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, n_steps);

    // Free the memory
    if (DEBUG)
        printf("[CHECKPOINT] Freeing lattice memory\n");
    for (int i = 0; i < lattice_size_x; i++)
    {
        free(lattice[i]);
    }
    free(lattice);

    if (DEBUG)
        printf("[CHECKPOINT] test() function complete\n");
    return 0;
}

/* Structure to hold simulation observables and timing results */
typedef struct
{
    float E;                            // Total energy
    float e_density;                    // Energy density (per spin)
    float m;                            // Total magnetization
    float m_density;                    // Magnetization density (per spin)
    float initialization_time;          // Time for initialization (seconds)
    float MH_evolution_time;            // Total time for MC evolution (seconds)
    float MH_evolution_time_over_steps; // Time per MC step (seconds)
} Observables;

/*
 * Run a complete Ising model simulation with timing
 * Returns structure containing final observables and performance metrics
 */
Observables run_ising_simulation(int lattice_size_x, int lattice_size_y,
                                 int type,
                                 float J, float h, float kB, float T,
                                 int n_steps)
{
    if (DEBUG)
        printf("[CHECKPOINT] Starting run_ising_simulation()\n");

    clock_t t0, t1;

    /* --- Initialization timing --- */
    if (DEBUG)
        printf("[CHECKPOINT] Starting initialization phase\n");
    t0 = clock();
    int **lattice = initialize_lattice(lattice_size_x, lattice_size_y, type);
    t1 = clock();

    float initialization_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;
    if (DEBUG)
        printf("[CHECKPOINT] Initialization complete (%.6f s)\n", initialization_time);

    /* --- Metropolis evolution timing --- */
    if (DEBUG)
        printf("[CHECKPOINT] Starting Metropolis evolution phase (%d steps)\n", n_steps);
    t0 = clock();
    for (int i = 0; i < n_steps; i++)
    {
        MH_step(lattice, lattice_size_x, lattice_size_y, J, h, kB, T);
        if (DEBUG && i % 100000 == 0)
        {
            printf("[CHECKPOINT] Evolution: %d / %d steps completed\n", i, n_steps);
        }
    }
    t1 = clock();

    float MH_evolution_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;
    if (DEBUG)
        printf("[CHECKPOINT] Metropolis evolution complete (%.6f s)\n", MH_evolution_time);

    /* --- Measurements --- */
    if (DEBUG)
        printf("[CHECKPOINT] Computing final observables\n");
    float E = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
    float e_density = energy_density_2D(E, lattice_size_x, lattice_size_y);
    float m = magnetisation_2D(lattice, lattice_size_x, lattice_size_y);
    float m_density = m / lattice_size_x / lattice_size_y;

    /* --- Cleanup --- */
    if (DEBUG)
        printf("[CHECKPOINT] Freeing lattice memory\n");
    for (int i = 0; i < lattice_size_x; i++)
    {
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
        MH_evolution_time / (float)n_steps;

    if (DEBUG)
        printf("[CHECKPOINT] run_ising_simulation() complete\n");
    return out;
}

// Simulated annealing function

typedef struct
{
    float *temperatures;
    float *energies;
    float *energy_densities;
    float *magnetizations;
    float *magnetization_densities;
    int n_temp_points;
} AnnealingResult;

/**
 * Simulated annealing with exponential temperature decay
 *
 * T(t) = T_initial * exp(-t/tau)
 */

AnnealingResult simulated_annealing(int lattice_size_x, int lattice_size_y,
                                    int type,
                                    float J, float h, float kB,
                                    float T_initial, float T_final, float tau,
                                    int steps_per_temp, int temp_update_interval,
                                    int save_trajectory)
{
    printf("========================================\n");
    printf("Simulated Annealing — 2D Ising Model\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("T_initial           : %.3f\n", T_initial);
    printf("T_final             : %.3f\n", T_final);
    printf("Decay constant tau  : %.3f\n", tau);
    printf("Steps per temp      : %d\n", steps_per_temp);
    printf("Temp update interval: %d\n", temp_update_interval);
    printf("\n");

    // Initialize lattice
    int **lattice = initialize_lattice(lattice_size_x, lattice_size_y, type);

    // Calculate number of temperature points
    int max_steps = (int)(-tau * log(T_final / T_initial));
    int n_temp_points = max_steps / temp_update_interval + 1;

    // Allocate arrays for trajectory if needed
    float *temperatures = NULL;
    float *energies = NULL;
    float *energy_densities = NULL;
    float *magnetizations = NULL;
    float *magnetization_densities = NULL;

    if (save_trajectory)
    {
        temperatures = (float *)malloc(n_temp_points * sizeof(float));
        energies = (float *)malloc(n_temp_points * sizeof(float));
        energy_densities = (float *)malloc(n_temp_points * sizeof(float));
        magnetizations = (float *)malloc(n_temp_points * sizeof(float));
        magnetization_densities = (float *)malloc(n_temp_points * sizeof(float));
    }

    float T = T_initial;
    int total_steps = 0;
    int trajectory_index = 0;

    printf("Starting annealing...\n");

    // Main annealing loop
    while (T > T_final)
    {
        // Perform MC steps at current temperature
        for (int i = 0; i < steps_per_temp; i++)
        {
            MH_step(lattice, lattice_size_x, lattice_size_y, J, h, kB, T);
        }

        total_steps += steps_per_temp;

        // Save current state if tracking trajectory
        if (save_trajectory && trajectory_index < n_temp_points)
        {
            float E = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
            float e_density = energy_density_2D(E, lattice_size_x, lattice_size_y);
            float m = magnetisation_2D(lattice, lattice_size_x, lattice_size_y);
            int N = lattice_size_x * lattice_size_y;

            temperatures[trajectory_index] = T;
            energies[trajectory_index] = E;
            energy_densities[trajectory_index] = e_density;
            magnetizations[trajectory_index] = m;
            magnetization_densities[trajectory_index] = (float)m / (float)N;

            trajectory_index++;
        }

        // Update temperature using exponential decay
        T = T_initial * exp(-total_steps / tau);

        // Print progress every 10 temperature updates
        if (trajectory_index % 10 == 0)
        {
            printf("Step %d: T = %.4f\n", total_steps, T);
        }
    }

    printf("\nAnnealing complete after %d total MC steps\n", total_steps);

    // Final measurements
    float E_final = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
    float e_density_final = energy_density_2D(E_final, lattice_size_x, lattice_size_y);
    float m_final = magnetisation_2D(lattice, lattice_size_x, lattice_size_y);
    int N = lattice_size_x * lattice_size_y;

    printf("\nFinal State:\n");
    printf("Temperature         : %.4f\n", T);
    printf("Energy              : %.4f\n", E_final);
    printf("Energy density      : %.4f\n", e_density_final);
    printf("Magnetization       : %.4f\n", m_final);
    printf("Magnetization/spin  : %.4f\n", (float)m_final / (float)N);

    // Cleanup lattice
    for (int i = 0; i < lattice_size_x; i++)
    {
        free(lattice[i]);
    }
    free(lattice);

    // Prepare result
    AnnealingResult result;
    result.temperatures = temperatures;
    result.energies = energies;
    result.energy_densities = energy_densities;
    result.magnetizations = magnetizations;
    result.magnetization_densities = magnetization_densities;
    result.n_temp_points = trajectory_index;

    return result;
}

void save_annealing_trajectory(const char *filename, AnnealingResult *result)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        perror("Error opening file for annealing trajectory save");
        return;
    }

    fprintf(fp, "# Temperature Energy EnergyDensity Magnetization MagnetizationDensity\n");

    for (int i = 0; i < result->n_temp_points; i++)
    {
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f\n",
                result->temperatures[i],
                result->energies[i],
                result->energy_densities[i],
                result->magnetizations[i],
                result->magnetization_densities[i]);
    }

    fclose(fp);
    printf("Annealing trajectory saved to %s\n", filename);
}

void free_annealing_result(AnnealingResult *result)
{
    if (result->temperatures != NULL)
        free(result->temperatures);
    if (result->energies != NULL)
        free(result->energies);
    if (result->energy_densities != NULL)
        free(result->energy_densities);
    if (result->magnetizations != NULL)
        free(result->magnetizations);
    if (result->magnetization_densities != NULL)
        free(result->magnetization_densities);
}

int annealing(int lattice_size_x, int lattice_size_y, int type, float J, float h, float kB, float T_initial, float T_final, float tau, int steps_per_temp, int temp_update_interval, int save_trajectory)
{

    AnnealingResult result = simulated_annealing(
        lattice_size_x, lattice_size_y, type,
        J, h, kB,
        T_initial, T_final, tau,
        steps_per_temp, temp_update_interval,
        save_trajectory);

    // Save trajectory to file
    save_annealing_trajectory("annealing_trajectory.dat", &result);

    // Free memory
    free_annealing_result(&result);

    return 0;
}

#ifdef STANDALONE_BUILD
int main(int argc, char *argv[])
{
    if (DEBUG)
        printf("[CHECKPOINT] main() started\n");

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

    if (DEBUG)
        printf("[CHECKPOINT] Parameters parsed: Lattice %d x %d, type %d\n",
               lattice_size_x, lattice_size_y, type);

    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
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

    if (DEBUG)
        printf("[CHECKPOINT] Calling run_ising_simulation()\n");
    Observables out = run_ising_simulation(lattice_size_x, lattice_size_y, type, J, h, kB, T, n_steps);

    printf("Energy                : %f\n", out.E);
    printf("Energy density        : %f\n", out.e_density);
    printf("Magnetization         : %f\n", out.m);
    printf("Magnetization density : %f\n", out.m_density);
    printf("Initialization time (s)        : %f\n", out.initialization_time);
    printf("MH evolution time (s)          : %f\n", out.MH_evolution_time);
    printf("MH time per step (s)           : %e\n", out.MH_evolution_time_over_steps);

    if (DEBUG)
        printf("[CHECKPOINT] main() complete\n");

    annealing();

    return 0;
}
#endif