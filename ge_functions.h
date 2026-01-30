#ifndef FUNCTIONS_CUDA_EFFICIENT_H
#define FUNCTIONS_CUDA_EFFICIENT_H

#include "s_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Observables run_ising_simulation_efficient_gpu(
        int lattice_size_x,
        int lattice_size_y,
        int type,
        float J,
        float h,
        float kB,
        float T,
        int n_steps);

#ifdef __cplusplus
}
#endif

#endif /* FUNCTIONS_CUDA_EFFICIENT_H */
