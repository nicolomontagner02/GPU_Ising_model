#!/bin/bash

# ---------------------------------------------------------
# MASTER COMPILATION FILE
# ---------------------------------------------------------
#
# Usage:
#   ./compile_all.sh <build_all> <O-level>
#
# Examples:
#   ./compile_all.sh 1 3   # rebuild everything with -O3
#   ./compile_all.sh 0 3   # reuse objects, only relink
# ---------------------------------------------------------

BUILD_ALL=$1
OLEVEL=$2

if [ -z "$OLEVEL" ]; then
    echo "Usage: $0 <build_all> <O-level>"
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
fi

# ---------------------------------------------------------
# Compile merged simulation (host code)
# ---------------------------------------------------------

echo "=== Compiling Run_simulations.c ==="
gcc -O${OLEVEL} -fopenmp -c Run_simulations.c \
    -o Run_simulations.o

# ---------------------------------------------------------
# Link everything
# ---------------------------------------------------------

echo "=== Linking compiled files ==="

g++ Run_simulations.o \
    Single_CPU/single_cpu_2D_ising.o \
    OpenMP/multiple_cpu_openMP_2D_ising.o \
    GPU/gpu_2D_ising.o \
    GPU/gpu_2D_ising_efficient.o \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -fopenmp \
    -lm \
    -o Run_simulations
