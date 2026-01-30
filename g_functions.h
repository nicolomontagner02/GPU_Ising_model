#ifndef FUNCTIONS_CUDA_H
#define FUNCTIONS_CUDA_H

#include "s_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Observables run_ising_simulation_gpu(
        int lattice_size_x,
        int lattice_size_y,
        int type,
        float J, float h, float kB, float T,
        int n_steps);

#ifdef __cplusplus
}
#endif

#endif
