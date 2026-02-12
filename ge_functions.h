#ifndef FUNCTIONS_CUDA_2D_BLOCK_H
#define FUNCTIONS_CUDA_2D_BLOCK_H

#include "s_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Observables run_ising_simulation_2D_block_gpu(
        int lattice_size_x,
        int lattice_size_y,
        int type,
        float J,
        float h,
        float kB,
        float T,
        int n_steps);

    Observables run_ising_simulation_2D_block_gpu_save(
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

    // GPU annealing functions
    int annealing_gpu(int lattice_size_x, int lattice_size_y, int type,
                      float J, float h, float kB,
                      float T_initial, float T_final, float tau,
                      int sweeps_per_temp, int temp_update_interval,
                      int save_trajectory);

#ifdef __cplusplus
}
#endif

#endif /* FUNCTIONS_CUDA_2DBLOCK_H */
