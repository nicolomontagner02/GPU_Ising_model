#!/bin/bash

# ---------------------------------------------------------
# MASTER COMPILATION FILE
# ---------------------------------------------------------

# Usage:
#   ./compile_all.sh <O-level> [build_all]
# Example:
#   ./compile_all.sh 3 1

OLEVEL=$2
BUILD_ALL=$1

if [ -z "$OLEVEL" ]; then
    echo "Usage: $0 <O-level> [build_all]"
    exit 1
fi

# ---------------------------------------------------------
# Compile backend implementations
# ---------------------------------------------------------

if [ "$BUILD_ALL" = "1" ]; then

    echo "=== Compiling single CPU code ==="
    gcc -O${OLEVEL} -c Single_CPU/single_cpu_2D_ising.c \
        -o Single_CPU/single_cpu_2D_ising.o

    echo "=== Compiling OpenMP CPU code ==="
    gcc -O${OLEVEL} -fopenmp -c OpenMP/multiple_cpu_openMP_2D_ising.c \
        -o OpenMP/multiple_cpu_openMP_2D_ising.o

    echo "=== Compiling GPU code ==="
    nvcc -O${OLEVEL} -c GPU/gpu_2D_ising.cu \
        -o GPU/gpu_2D_ising.o

    echo "=== Compiling efficient GPU code ==="
    nvcc -O${OLEVEL} -c GPU/gpu_2D_ising_efficient.cu \
        -o GPU/gpu_2D_ising_efficient.o

    echo "=== Compiling efficient memory GPU code ==="
    nvcc -O${OLEVEL} -c GPU/gpu_2D_ising_eff_memory.cu \
        -o GPU/gpu_2D_ising_eff_memory.o
fi

# ---------------------------------------------------------
# Compile merged simulation (host code)
# ---------------------------------------------------------

echo "=== Compiling merged simulation ==="
gcc -O${OLEVEL} -c Ising_simulations.c -o Ising_simulations.o
gcc -O${OLEVEL} -c Regimes_control.c -o Regimes_control.o

# ---------------------------------------------------------
# Link everything
# ---------------------------------------------------------

echo "=== Linking compiled files ==="

g++ Ising_simulations.o \
    Single_CPU/single_cpu_2D_ising.o \
    OpenMP/multiple_cpu_openMP_2D_ising.o \
    GPU/gpu_2D_ising.o \
    GPU/gpu_2D_ising_efficient.o \
    GPU/gpu_2D_ising_eff_memory.o\
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -fopenmp \
    -lm \
    -o Ising_simulations

g++ Regimes_control.o \
    Single_CPU/single_cpu_2D_ising.o \
    OpenMP/multiple_cpu_openMP_2D_ising.o \
    GPU/gpu_2D_ising.o \
    GPU/gpu_2D_ising_efficient.o \
    GPU/gpu_2D_ising_eff_memory.o\
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -fopenmp \
    -lm \
    -o Regimes_control
