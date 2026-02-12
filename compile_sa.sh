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
    echo "Usage: $0 [build_all] <O-level>"
    exit 1
fi

# ---------------------------------------------------------
# Compile backend implementations
# ---------------------------------------------------------

if [ "$BUILD_ALL" = "1" ]; then

    echo "=== Compiling single CPU code ==="
    gcc -O${OLEVEL} -c Single_CPU/single_cpu_2D_ising.c \
        -o Single_CPU/single_cpu_2D_ising.o

    echo "=== Compiling efficient GPU code ==="
    nvcc -O${OLEVEL} -c GPU/gpu_2D_ising_2D_block.cu \
        -o GPU/gpu_2D_ising_2D_block.o

fi

# ---------------------------------------------------------
# Compile merged simulation (host code)
# ---------------------------------------------------------

echo "=== Compiling merged simulation ==="
gcc -O${OLEVEL} -c Simulated_annealing.c -o Simulated_annealing.o

# ---------------------------------------------------------
# Link everything
# ---------------------------------------------------------

echo "=== Linking compiled files ==="

g++ Simulated_annealing.o \
    Single_CPU/single_cpu_2D_ising.o \
    GPU/gpu_2D_ising_2D_block.o\
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -fopenmp \
    -lm \
    -o Simulated_annealing
