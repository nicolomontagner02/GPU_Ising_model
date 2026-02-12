#!/bin/bash

# ---------------------------------------------------------
# SHARED LIBRARY COMPILATION FILE (Python importable)
# ---------------------------------------------------------
#
# Usage:
#   ./compile_lib.sh <build_all> <O-level>
#
# Example:
#   ./compile_lib.sh 1 3
#
# Output:
#   build/libising.so
# ---------------------------------------------------------

BUILD_ALL=$1
OLEVEL=$2

if [ -z "$OLEVEL" ]; then
    echo "Usage: $0 <build_all> <O-level>"
    exit 1
fi

mkdir -p build

# ---------------------------------------------------------
# Compile backend implementations
# ---------------------------------------------------------

if [ "$BUILD_ALL" = "1" ]; then

    echo "=== Compiling single CPU code ==="
    gcc -O${OLEVEL} -fPIC -c Single_CPU/single_cpu_2D_ising.c \
        -o build/single_cpu_2D_ising.o

    echo "=== Compiling OpenMP CPU code ==="
    gcc -O${OLEVEL} -fPIC -fopenmp -c OpenMP/multiple_cpu_openMP_2D_ising.c \
        -o build/multiple_cpu_openMP_2D_ising.o

    echo "=== Compiling GPU code ==="
    nvcc -O${OLEVEL} -Xcompiler -fPIC -c GPU/gpu_2D_ising.cu \
        -o build/gpu_2D_ising.o

    echo "=== Compiling efficient GPU code ==="
    nvcc -O${OLEVEL} -Xcompiler -fPIC -c GPU/gpu_2D_ising_efficient.cu \
        -o build/gpu_2D_ising_efficient.o

    echo "=== Compiling efficient 1Dthreads GPU code ==="
    nvcc -O${OLEVEL} -Xcompiler -fPIC -c GPU/gpu_2D_ising_eff_memory.cu \
        -o build/gpu_2D_ising_eff_memory.o
fi

# ---------------------------------------------------------
# Compile merged API (host-facing symbols)
# ---------------------------------------------------------

echo "=== Compiling Ising_simulations API ==="
gcc -O${OLEVEL} -fPIC -c Ising_simulations.c \
    -o build/Ising_simulations.o

# ---------------------------------------------------------
# Link shared library
# ---------------------------------------------------------

echo "=== Linking shared library libising.so ==="

g++ -shared \
    build/Ising_simulations.o \
    build/single_cpu_2D_ising.o \
    build/multiple_cpu_openMP_2D_ising.o \
    build/gpu_2D_ising.o \
    build/gpu_2D_ising_efficient.o \
    build/gpu_2D_ising_eff_memory.o\
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -fopenmp \
    -lm \
    -o build/libising.so

echo "=== Build complete: build/libising.so ==="
