#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DEBUG 0

void print_lattice(int **lattice, int size_x, int size_y){

    printf("2D Ising Lattice:\n");
        for (int i = 0; i < size_x; i++) {
            for (int j = 0; j < size_y; j++) {
                printf("%d ", lattice[i][j]);
            }
            printf("\n");
        }

}

int **initialize_lattice(int lattice_size_x, int lattice_size_y, int type){
    
    // Allocate memory for the lattice
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++) {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

    switch(type){
        // Switch for different initializations cases
        case 1:
            // Initialize to all spins up (+1)
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = 1;
                }
            }
            break;

        case 2:
            // Initialize to all spins down (-1)
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = -1;
                }
            }
            break;

        case 3:
            // Initialize to random spins (+1 or -1)
            srand(2); // Seed for reproducibility
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
                }

            }
            break;

        default:
            printf("Wrong initialization case submitted!\n");
            break;

    }
        
    //print the lattice (only small size)
    if (lattice_size_x <=4 && lattice_size_y <=4){
        print_lattice(lattice, lattice_size_x, lattice_size_y);
    }  

    return lattice;
}

float energy_2D(int **lattice, int size_x, int size_y, float J, float h) {
    float energy = 0.0;
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

int magnetisation_2D(int **lattice, int size_x, int size_y){

    int magnetisation = 0;

    for (int i = 0; i < size_x; i++){
        for (int j = 0; j < size_y; j++){
            magnetisation += lattice[i][j];
        }
    }

    return magnetisation ;
}

float d_energy_2D(int **lattice, int i, int j, int size_x, int size_y,float J, float h){
    int spin = lattice[i][j];
    float sum_nn = 0.0f;

    if (i > 0)           sum_nn += lattice[i-1][j];
    if (i < size_x - 1)  sum_nn += lattice[i+1][j];
    if (j > 0)           sum_nn += lattice[i][j-1];
    if (j < size_y - 1)  sum_nn += lattice[i][j+1];

    return 2.0f * spin * (J * sum_nn + h);
}

float energy_density_2D(float energy, int size_x, int size_y){

    int N = size_x*size_y;

    float e_density = energy / N;

    return e_density;
}

void MH_step(int **lattice, int size_x, int size_y,float J, float h, float kB, float T){
    int i_s = rand() % size_x;
    int j_s = rand() % size_y;

    if (DEBUG) {
        printf("Proposed flip at (%d, %d)\n", i_s, j_s);
    }

    float d_energy = d_energy_2D(lattice, i_s, j_s,
                                 size_x, size_y, J, h);

    int accept = 0;

    if (d_energy <= 0.0f) {
        accept = 1;
    } else {
        float p = exp(-d_energy / (kB * T));
        float u = (float) rand() / (float) RAND_MAX;
        if (u < p) {
            accept = 1;
        }
    }

    if (accept) {
        lattice[i_s][j_s] *= -1;
        if (DEBUG) {
            printf("Move accepted (ΔE = %.4f)\n", d_energy);
        }
    } else if (DEBUG) {
        printf("Move rejected (ΔE = %.4f)\n", d_energy);
    }
}

void report_state(const char *label, int **lattice, int size_x, int size_y, float J, float h){

    float E = energy_2D(lattice, size_x, size_y, J, h);
    float e_density = energy_density_2D(E, size_x, size_y);
    int m = magnetisation_2D(lattice, size_x, size_y);

    int N = size_x * size_y;

    printf("----------------------------------------\n");
    printf("%s\n", label);
    printf("----------------------------------------\n");
    printf("Total energy        : %f\n", E);
    printf("Energy density      : %f\n", e_density);
    printf("Magnetisation       : %f\n", m);
    printf("Magnetisation/spin  : %f\n",(float) m /(float) N);
    
    if (size_x <= 4 && size_y <= 4) {
        print_lattice(lattice, size_x, size_y);
    }
}

void save_lattice(const char *folder, int **lattice, int size_x, int size_y, float J, float h, float T, int mc_steps){
    
    char filename[512];

    snprintf(filename, sizeof(filename),
             "%s/ising_J%.3f_h%.3f_T%.3f_Lx%d_Ly%d_MC%d.dat",
             folder, J, h, T, size_x, size_y, mc_steps);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file for lattice save");
        return;
    }

    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            fprintf(fp, "%d ", lattice[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Lattice saved to %s\n", filename);
}


int test(){

    // Parse command line arguments
    int lattice_size_x = 10;
    int lattice_size_y = 10;

    int type = 3; // 1 for all spin up, 2 for all spin down, 3 for random initialization of the lattice

    float J = 1.0; // Interaction strength
    float h = 1.0 ; // External magnetic field
    float kB = 1.0*exp(-23);
    float T = 100.0;
    int n_steps = 1000000;

    // simulation header
    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("MC steps            : %d\n", n_steps);
    printf("Initialization type : %s\n",
        type == 1 ? "All up" :
        type == 2 ? "All down" : "Random");
    printf("\n");

    int **lattice = initialize_lattice(lattice_size_x, lattice_size_y, type);

    report_state("Initial state",lattice, lattice_size_x, lattice_size_y,J, h);
    save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, 0);

    for (int i = 0; i < n_steps; i++){
        MH_step(lattice, lattice_size_x, lattice_size_y, J ,h, kB, T);
        if (DEBUG){
            printf("MH step %i executed.\n",i);
        }
    }

    report_state("Final state", lattice, lattice_size_x, lattice_size_y, J, h);
    save_lattice("data", lattice, lattice_size_x, lattice_size_y, J, h, T, n_steps);

    // free the memory
    for (int i = 0; i < lattice_size_x; i++) {
        free(lattice[i]);
    }
    free(lattice);

    return 0;

}

typedef struct {
    float E;
    float e_density;
    float m;
    float m_density;
    float initialization_time;
    float MH_evolution_time;
    float MH_evolution_time_over_steps;
} Observables;

Observables run_ising_simulation(int lattice_size_x, int lattice_size_y,
                                 int type,
                                 float J, float h, float kB, float T,
                                 int n_steps)
{
    clock_t t0, t1;

    /* --- Initialization timing --- */
    t0 = clock();
    int **lattice = initialize_lattice(lattice_size_x, lattice_size_y, type);
    t1 = clock();

    float initialization_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    /* --- Metropolis evolution timing --- */
    t0 = clock();
    for (int i = 0; i < n_steps; i++) {
        MH_step(lattice, lattice_size_x, lattice_size_y, J, h, kB, T);
    }
    t1 = clock();

    float MH_evolution_time =
        (float)(t1 - t0) / CLOCKS_PER_SEC;

    /* --- Measurements --- */
    float E = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
    float e_density = energy_density_2D(E, lattice_size_x, lattice_size_y);
    float m = magnetisation_2D(lattice, lattice_size_x, lattice_size_y);
    float m_density = m;

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
        MH_evolution_time / (float)n_steps;

    return out;
}

int main(int argc, char *argv[]) {

    // Check command line arguments
    if (argc != 3) {
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
    float kB = 1.0*exp(-23);
    float T = 100;
    int n_steps = 50000;

    printf("========================================\n");
    printf("2D Ising Model — Metropolis Simulation\n");
    printf("========================================\n");
    printf("Lattice size        : %d x %d\n", lattice_size_x, lattice_size_y);
    printf("Interaction J       : %.3f\n", J);
    printf("External field h    : %.3f\n", h);
    printf("Temperature T       : %.3f\n", T);
    printf("MC steps            : %d\n", n_steps);
    printf("Initialization type : %s\n",
        type == 1 ? "All up" :
        type == 2 ? "All down" : "Random");
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

    return 0;
}