# GPU-Accelerated 2D Ising Model Simulation

A high-performance, multi-platform implementation of the 2D Ising Model using CPU (single-threaded and OpenMP) and NVIDIA GPU (CUDA) parallel computing frameworks. This project demonstrates the significant performance gains achievable by leveraging GPU parallelism for large-scale statistical physics simulations.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Scientific Background](#scientific-background)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Results and Data](#results-and-data)
- [Publications and References](#publications-and-references)

---

## Project Overview

The **Ising Model** is a fundamental model in statistical mechanics that describes ferromagnetism in statistical physics. This project implements a comprehensive simulation framework for the 2D Ising Model on square lattices, supporting multiple computational backends to compare performance across different hardware architectures.

**Key Capabilities:**
- Simulate 2D Ising lattices ranging from 16×16 to 4096×4096 sites
- Multiple backend implementations: single CPU, OpenMP (4 threads), and three GPU variants
- Metropolis-Hastings algorithm for thermal equilibrium sampling
- Comprehensive measurement of observables (energy, magnetization, execution time)
- Parameter sweeps across temperature, external field, and coupling constant
- Python bindings for automated benchmarking and data analysis
- Jupyter notebook for post-processing and visualization

---

## Features

### Computational Backends

| Backend | Description | Memory Usage | Parallelization |
|---------|-------------|--------------|-----------------|
| **Single CPU** | Sequential implementation for baseline comparison | O(N) | Single thread |
| **OpenMP** | Multi-threaded CPU implementation | O(N) | 4 threads (configurable) |
| **GPU (Basic)** | Naive CUDA implementation | O(N) | Thousands of threads |
| **GPU (Efficient)** | Optimized GPU with stateless RNG | O(N) | Thousands of threads |
| **GPU (Eff Memory)** | Memory-optimized using int8_t | 1/4 × standard GPU | Thousands of threads |

### Simulation Features

- **Lattice Initialization**: Cold start (all spins up/down), Hot start (random configuration)
- **Boundary Conditions**: Periodic boundary conditions (toroidal geometry)
- **Update Algorithm**: Checkerboard ( checkerboard decomposition ) for parallel updates
- **Random Number Generation**: 
  - CPU: Standard library RNG
  - GPU: cuRAND with stateful and stateless approaches
- **Observables Measured**:
  - Total energy (E)
  - Energy density (E/N)
  - Magnetization (M)
  - Magnetization density (M/N)
  - Execution time profiling

### Data Analysis

- Automated parameter sweeps with CSV output
- Jupyter notebook for heatmaps, lattice visualization, and statistical analysis
- Lattice state saving for later analysis

---

## Scientific Background

### The Ising Model

The 2D Ising Model consists of a discrete grid of spins σᵢⱼ that can take values ±1. The Hamiltonian (energy functional) is:

$$H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i$$

Where:
- $J$ is the coupling constant (ferromagnetic if $J > 0$)
- $h$ is the external magnetic field
- $\langle i,j \rangle$ denotes nearest-neighbor pairs

### Phase Transition

The 2D Ising model exhibits a **second-order phase transition** at the critical temperature:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269 J/k_B$$

Below $T_c$, the system spontaneously magnetizes (ferromagnetic phase). Above $T_c$, thermal fluctuations dominate (paramagnetic phase).

### Metropolis-Hastings Algorithm

The simulation uses the Metropolis-Hastings algorithm to sample configurations at thermal equilibrium:

1. Propose a spin flip: σᵢⱼ → -σᵢⱼ
2. Calculate energy change: ΔE = E_new - E_old
3. Accept with probability: P(accept) = min(1, exp(-ΔE/kBT))
4. Repeat until equilibrium is reached

### Critical Slowing Down

Near the critical temperature $T_c$, the correlation length diverges, leading to **critical slowing down** — the dynamics become increasingly slow as the system takes longer to decorrelate. This motivates the use of GPU acceleration to simulate larger systems and achieve better statistics.

---

## Architecture

```
GPU_Ising_model/
├── Single_CPU/                 # Sequential CPU implementation
│   ├── single_cpu_2D_ising.c
│   ├── Makefile
│   └── single_cpu_2D_ising.o
│
├── OpenMP/                     # Multi-threaded CPU implementation
│   ├── multiple_cpu_openMP_2D_ising.c
│   ├── Makefile
│   └── multiple_cpu_openMP_2D_ising.o
│
├── GPU/                        # CUDA GPU implementations
│   ├── gpu_2D_ising.cu         # Basic GPU implementation
│   ├── gpu_2D_ising_efficient.cu   # Optimized GPU
│   ├── gpu_2D_ising_eff_memory.cu  # Memory-optimized GPU
│   ├── Makefile
│   └── *.o files
│
├── build/                      # Shared library
│   ├── libising.so            # Python-callable shared library
│   └── *.o files
│
├── Ising_simulations.c        # Main comparison program
├── Regimes_control.c           # Parameter sweep program
├── Run_simulations.py          # Python benchmark script
│
├── Post_processing.ipynb       # Jupyter analysis notebook
├── compile_all.sh              # Master compilation script
├── compile_share_library.sh    # Shared library compilation
│
├── s_functions.h               # Shared structure definitions
├── m_functions.h               # OpenMP functions
├── g_functions.h               # Basic GPU functions
├── ge_functions.h              # Efficient GPU functions
├── gm_functions.h              # Memory-efficient GPU functions
│
├── data/                       # Simulation output data
├── results/                    # Benchmark results (CSV)
└── README.md                   # This file
```

### Core Data Structures

```c
typedef struct {
    float E;                           // Total energy
    float e_density;                  // Energy per site
    float m;                           // Total magnetization
    float m_density;                   // Magnetization per site
    float initialization_time;         // Setup time (seconds)
    float MH_evolution_time;          // Evolution time (seconds)
    float MH_evolution_time_over_steps; // Time per MC step
} Observables;
```

### Key Algorithms

1. **Checkerboard Decomposition**: Splits the lattice into two independent sub-lattices (black and white), allowing parallel updates without race conditions.

2. **GPU Memory Optimization**: Uses `int8_t` instead of `int` for spins, reducing memory footprint by 4× and improving cache efficiency.

3. **Stateless RNG**: The efficient GPU implementation uses stateless random number generation (Philox4_32_10), eliminating the need to store RNG state per thread.

---

## Installation

### Prerequisites

- **Operating System**: macOS or Linux
- **Compiler**: GCC/G++ (for CPU code)
- **CUDA Toolkit**: NVIDIA CUDA 11.0 or later
- **Python**: 3.8+ (for Python bindings and analysis)
- **Dependencies**: NumPy, Matplotlib, Pandas, Jupyter

### Installing CUDA

**macOS (with Apple Silicon):**
```bash
# CUDA is integrated via Metal Performance Shaders (MPS)
# No separate installation needed for basic development
```

**Linux:**
```bash
# Download and install from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
sudo apt-get install nvidia-cuda-toolkit
```

### Python Dependencies

```bash
pip install numpy matplotlib pandas jupyter
```

### Compilation

```bash
# Navigate to project directory
cd /path/to/GPU_Ising_model

# Compile with optimization level 2 (or 3 for max optimization)
./compile_all.sh 2 1

# Compile only the main programs (without rebuilding backends)
./compile_all.sh 2 0

# Build shared library for Python
./compile_share_library.sh
```

### Compilation Flags

```bash
# Single CPU
gcc -O2 -c Single_CPU/single_cpu_2D_ising.c -o Single_CPU/single_cpu_2D_ising.o

# OpenMP
gcc -O2 -fopenmp -c OpenMP/multiple_cpu_openMP_2D_ising.c -o OpenMP/multiple_cpu_openMP_2D_ising.o

# CUDA
nvcc -O2 -c GPU/gpu_2D_ising.cu -o GPU/gpu_2D_ising.o
```

---

## Usage

### Command-Line Programs

#### Basic Simulation (All Backends)

```bash
# Run simulation on 512×512 lattice
./Ising_simulations 512 512

# Output:
# ========================================
# 2D Ising Model — 1 CPU
# ========================================
# Lattice size        : 512 x 512
# Interaction J       : 1.000
# External field h   : 1.000
# Temperature T       : 1.000
# MC steps            : 10000000
# Initialization type : Random
# ========================================
# Energy                : -523146.500000
# Energy density        : -2.001
# Magnetization         : -512
# Magnetization density : -0.002
# Initialization time (s)        : 0.002
# MH evolution time (s)          : 45.231
# MH time per step (s)           : 4.523e-08
```

#### Parameter Sweep

```bash
# Generate magnetization phase diagrams
# Parameters: lattice size, temperature, coupling J, field h
./Regimes_control 1000 1000

# Output files saved to data/:
# - magnetization1000_1000_T1.000_type1.csv
# - magnetization1000_1000_T1.000_type3.csv
# - magnetization1000_1000_T100.000_type1.csv
# - magnetization1000_1000_T100.000_type3.csv
```

#### Python Benchmarking

```bash
# Run automated benchmarks comparing all backends
python3 Run_simulations.py

# Results saved to results/ising_results_YYYYMMDD_HHMMSSnew.csv
```

### Python API

```python
import ctypes
import numpy as np

# Load shared library
lib = ctypes.CDLL("./build/libising.so")

# Define observables structure
class Observables(ctypes.Structure):
    _fields_ = [
        ("E", ctypes.c_float),
        ("e_density", ctypes.c_float),
        ("m", ctypes.c_float),
        ("m_density", ctypes.c_float),
        ("initialization_time", ctypes.c_float),
        ("MH_evolution_time", ctypes.c_float),
        ("MH_evolution_time_over_steps", ctypes.c_float),
    ]

# Bind functions
lib.run_ising_simulation_efficient_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int
]
lib.run_ising_simulation_efficient_gpu.restype = Observables

# Run simulation
result = lib.run_ising_simulation_efficient_gpu(
    1024, 1024,      # Lattice size
    3,               # Random initialization
    1.0, 0.5, 1.0, 2.0,  # J, h, kB, T
    100000           # MC steps
)

print(f"Energy density: {result.e_density:.4f}")
print(f"Magnetization density: {result.m_density:.4f}")
print(f"Execution time: {result.MH_evolution_time:.2f}s")
```

---

## Performance

### Benchmark Results (Typical)

| Backend | Lattice | MC Steps | Time (s) | Speedup vs CPU |
|---------|---------|----------|----------|----------------|
| Single CPU | 256×256 | 100K | 12.5 | 1.0× |
| OpenMP | 256×256 | 100K | 3.8 | 3.3× |
| GPU (Basic) | 256×256 | 100K | 0.45 | 27.8× |
| GPU (Efficient) | 256×256 | 100K | 0.38 | 32.9× |
| GPU (Eff Memory) | 256×256 | 100K | 0.35 | 35.7× |

### Lattice Scaling

The GPU implementations show increasing advantages for larger lattices due to better thread utilization and memory bandwidth:

```
Lattice Size    CPU Time (s)   GPU Time (s)   Speedup
---------------------------------------------------
64×64           0.8           0.12           6.7×
256×256         12.5          0.38           32.9×
512×512         52.3          0.85           61.5×
1024×1024       215.7         2.1            102.7×
```

### Memory Usage

| Backend | 4096×4096 Memory |
|---------|-------------------|
| Single CPU | ~64 MB (int arrays) |
| GPU (Basic) | ~256 MB (device memory) |
| GPU (Eff Memory) | ~64 MB (int8_t) |

---

## Dependencies

### Build Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| GCC/G++ | 11.0+ | C/C++ compilation |
| CUDA Toolkit | 11.0+ | GPU kernel compilation |
| cuRAND | Included | GPU random number generation |
| Make | 3.8+ | Build automation |

### Python Dependencies

```
numpy>=1.20.0
matplotlib>=3.5.0
pandas>=1.3.0
jupyter>=1.0.0
ctypes (standard library)
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any x86_64 | Modern multi-core |
| GPU | NVIDIA Kepler+ | NVIDIA Ampere or later |
| VRAM | 256 MB | 4+ GB |
| RAM | 2 GB | 8+ GB |
| Storage | 100 MB | SSD recommended |

---

## Project Structure

### Directory Breakdown

```
GPU_Ising_model/
├── Single_CPU/           # Sequential baseline (1 file)
├── OpenMP/               # Multi-threaded CPU (1 file)
├── GPU/                  # Three CUDA implementations (3 files)
├── build/                # Compiled objects and shared library
├── Ising_simulations.c   # Multi-backend comparison tool
├── Regimes_control.c     # Parameter sweep tool
├── Run_simulations.py    # Python benchmarking script
├── Post_processing.ipynb # Visualization and analysis
├── compile_all.sh        # Full compilation pipeline
├── compile_share_library.sh  # Shared library builder
├── s_functions.h        # Shared structures
├── m_functions.h        # OpenMP functions
├── g_functions.h        # Basic GPU functions
├── ge_functions.h       # Efficient GPU functions
├── gm_functions.h       # Memory-optimized GPU functions
└── README.md           # Documentation
```

### File Dependencies

```
Ising_simulations.c
├── s_functions.h
├── m_functions.h
├── g_functions.h
├── ge_functions.h
└── gm_functions.h

Regimes_control.c
├── s_functions.h
└── ge_functions.h

Run_simulations.py
└── build/libising.so (dynamically loaded)

Post_processing.ipynb
├── numpy
├── matplotlib
├── pandas
└── data/ (simulation outputs)
```

---

## Results and Data

### Output Files

**CSV Benchmark Results** (`results/`):
```csv
backend,L,init_type,J,h,T,n_steps,E,e_density,m,m_density,init_time,mh_time,mh_time_per_step
gpu_efficient,256,random,1.000,0.500,2.000,100000,-131037.500000,-1.999,-22.000,-0.000334,0.001,0.321,3.210e-09
```

**Magnetization Data** (`data/`):
```csv
J,h,magnetization_density
0.000,-1.000,0.002
0.000,-0.900,0.015
...
```

**Lattice States** (`data/`):
```
ising_J1.000_h0.500_T2.000_Lx256_Ly256_MC100000_type3.dat
# Space-separated spin values (±1)
1 -1 1 -1 ...
-1 1 -1 1 ...
```

### Jupyter Analysis

The `Post_processing.ipynb` notebook provides:

1. **Lattice Visualization**: 2D heatmaps of spin configurations
2. **Phase Diagrams**: Heatmaps of magnetization vs (J, h) parameters
3. **Time Series**: Energy and magnetization evolution
4. **Statistical Analysis**: Mean, variance, error estimation

---

## Publications and References

### Scientific Context

1. **Ising, E.** (1925) "Beitrag zur Theorie des Ferromagnetismus" *Zeitschrift für Physik*
2. **Onsager, L.** (1944) "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition" *Physical Review*
3. **Metropolis, N.** et al. (1953) "Equation of State Calculations by Fast Computing Machines" *Journal of Chemical Physics*

### CUDA Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [cuRAND Library](https://docs.nvidia.com/cuda/curand/)

### Related Projects

- **OpenMP CPU Parallelization**: Uses checkerboard decomposition for thread-safe updates
- **GPU Optimization**: Exploits thousands of threads for massive parallelism
- **Memory-Efficient Storage**: `int8_t` reduces memory footprint by 4×

---

## License

This project is released under the MIT License.

---

## Author

**Nicola Montagner**
- Institution: Modern Computing Course
- Project: MCP 2025

---

## Acknowledgments

- CUDA samples and documentation from NVIDIA
- Statistical mechanics concepts from modern condensed matter physics literature
