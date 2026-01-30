# ---------------------------------------------------------
# MASTER COMPILATION FILE
# ---------------------------------------------------------

# Compile the different files
echo "=== Compiling single CPU code ==="
gcc -O$1 -c Single_CPU/single_cpu_2D_ising.c -o Single_CPU/single_cpu_2D_ising.o
echo "=== Compiling multiple CPU code ==="
gcc -O$1 -fopenmp -c OpenMP/multiple_cpu_openMP_2D_ising.c -o OpenMP/multiple_cpu_openMP_2D_ising.o -lm
echo "=== Compiling GPU code ==="
# nvcc -O$1 -c GPU/gpu_2D_ising.cu -o GPU/gpu_2D_ising.o
echo "=== Compiling efficient GPU code ==="
nvcc -O$1 -c GPU/gpu_2D_ising_efficient.cu -o GPU/gpu_2D_ising_efficient.o

# Compile the merged simulation
echo "=== Compiling merged simulation ==="
nvcc -O$1 -c Ising_simulations.c -o Ising_simulations.o

# Link the compiled files
echo "=== Linking compiled files ==="
# nvcc Ising_simulations.o Single_CPU/single_cpu_2D_ising.o OpenMP/multiple_cpu_openMP_2D_ising.o GPU/gpu_2D_ising.o GPU/gpu_2D_ising_efficient.o -o Ising_simulations
gcc Ising_simulations.o Single_CPU/single_cpu_2D_ising.o OpenMP/multiple_cpu_openMP_2D_ising.o GPU/gpu_2D_ising_efficient.o -L/usr/local/cuda/lib64 -lcudart -lstdc++ -fopenmp -o Ising_simulations -lm