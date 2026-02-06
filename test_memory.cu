// test_gpu_limits.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU Device Properties ===\n");
    printf("Device: %s\n", prop.name);
    printf("Total memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("Max grid dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // Test maximum allocation
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("\n=== Memory Status ===\n");
    printf("Total: %.2f MB\n", total_mem / 1024.0 / 1024.0);
    printf("Free: %.2f MB\n", free_mem / 1024.0 / 1024.0);
    
    // Try to allocate different sizes
    printf("\n=== Testing Allocations ===\n");
    size_t test_sizes[] = {
        100 * 1024 * 1024,   // 100 MB
        200 * 1024 * 1024,   // 200 MB
        500 * 1024 * 1024,   // 500 MB
        1024 * 1024 * 1024,  // 1 GB
    };
    
    for (int i = 0; i < 4; i++) {
        void *ptr;
        cudaError_t err = cudaMalloc(&ptr, test_sizes[i]);
        if (err == cudaSuccess) {
            printf("✓ Successfully allocated %.2f MB\n", test_sizes[i] / 1024.0 / 1024.0);
            cudaFree(ptr);
        } else {
            printf("✗ Failed to allocate %.2f MB: %s\n", 
                   test_sizes[i] / 1024.0 / 1024.0,
                   cudaGetErrorString(err));
        }
    }
    
    return 0;
}
