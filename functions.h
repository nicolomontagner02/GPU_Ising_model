#ifndef FUNCTIONS_H
#define FUNCTIONS_H

//General purposes functions & structures
void print_lattice(int **lattice, int size_x, int size_y);
void report_state(const char *label, int **lattice, int size_x, int size_y, float J, float h);

float d_energy_2D(int **lattice, int i, int j, int size_x, int size_y,float J, float h);
float energy_density_2D(float energy, int size_x, int size_y);

typedef struct {
    float E;
    float e_density;
    float m;
    float m_density;
    float initialization_time;
    float MH_evolution_time;
    float MH_evolution_time_over_steps;
} Observables;

//Single CPU function
int **initialize_lattice(int lattice_size_x, int lattice_size_y, int type);

void MH_step(int **lattice, int size_x, int size_y,float J, float h, float kB, float T);

Observables run_ising_simulation(int lattice_size_x, int lattice_size_y,int type,
                                 float J, float h, float kB, float T, int n_steps);

//OpenMP functions
int **initialize_lattice_openmp(int lattice_size_x, int lattice_size_y, int type);

void MH_sweep_checkerboard_openmp(int **lattice, int size_x, int size_y, float J, float h, float kB, float T);

Observables run_ising_simulation_openmp(int lattice_size_x, int lattice_size_y,int type,
                                 float J, float h, float kB, float T, int n_steps);


#endif