#ifndef FUNCTIONS_CUDA_1D_BLOCK_H
#define FUNCTIONS_CUDA_1D_BLOCK_H

#include "s_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Observables run_ising_simulation_1Dblock_gpu(
        int lattice_size_x,
        int lattice_size_y,
        int type,
        float J,
        float h,
        float kB,
        float T,
        int n_steps);

    Observables run_ising_simulation_1Dblock_gpu_save(
        int lattice_size_x,
        int lattice_size_y,
        int type,
        float J,
        float h,
        float kB,
        float T,
        int n_steps,
        int save_lattice_flag,
        const char *save_folder);

#ifdef __cplusplus
}
#endif

#endif /* FUNCTIONS_CUDA_1D_BLOCK_H */
