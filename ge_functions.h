#ifndef FUNCTIONS_CUDA_EFFICIENT_H
#define FUNCTIONS_CUDA_EFFICIENT_H

#ifdef __CUDACC__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hot initialization (efficient RNG version) */
void initialize_lattice_gpu_hot_efficient(
    int8_t *d_lattice,
    int size_x,
    int size_y,
    unsigned long long seed
);

/* Energy */
float energy_2D_gpu_efficient(
    int8_t *d_lattice,
    int size_x,
    int size_y,
    float J,
    float h
);

/* Magnetization */
int magnetization_2D_gpu_efficient(
    int8_t *d_lattice,
    int size_x,
    int size_y
);

/* One full Metropolis–Hastings checkerboard sweep */
void MH_checkerboard_sweep_gpu_efficient(
    int8_t *d_lattice,
    int size_x,
    int size_y,
    float J,
    float h,
    float beta,
    int sweep
);

/* Sanity check for hot initialization */
bool check_hot_lattice_randomness(
    const int8_t *d_lattice,
    int size_x,
    int size_y
);

#ifdef __cplusplus
}
#endif

#endif /* __CUDACC__ */
#endif /* FUNCTIONS_CUDA_EFFICIENT_H */
