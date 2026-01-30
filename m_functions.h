#ifndef M_FUNCTIONS_H
#define M_FUNCTIONS_H

#include "s_functions.h"

// -------------------------------------------------
// OpenMP functions
// ------------------------------------------------------
int **initialize_lattice_openmp(int lattice_size_x, int lattice_size_y, int type);

void MH_sweep_checkerboard_openmp(int **lattice, int size_x, int size_y, float J, float h, float kB, float T);

Observables run_ising_simulation_openmp(int lattice_size_x, int lattice_size_y, int type, float J, float h, float kB, float T, int n_steps);

#endif