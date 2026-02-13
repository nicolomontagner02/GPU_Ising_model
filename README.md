# GPU-Accelerated 2D Ising Model Simulation

A high-performance implementation of the 2D Ising Model using CUDA GPU acceleration, with comparisons against single-threaded and OpenMP parallel CPU implementations.

## What is the Ising Model?

The Ising Model is a mathematical model of ferromagnetism in statistical physics. It consists of a lattice of discrete spins that can be either +1 or -1. Spins tend to align with their neighbors (interaction strength J) and may align with an external magnetic field (h). The model exhibits a phase transition at a critical temperature (Tc ≈ 2.269 for zero field).

This project simulates the model using the Metropolis-Hastings (MH) algorithm and includes simulated annealing to find ground states.

## Key Features

### Multiple Backend Implementations

| Backend | Description | Use Case |
|---------|-------------|----------|
| **Single CPU** | Sequential C implementation | Baseline reference |
| **OpenMP CPU** | Multi-threaded parallel (4 threads) | CPU parallel scaling |
| **GPU (Standard)** | Basic CUDA kernel | Initial GPU implementation |
| **GPU 1D Block** | Optimized 1D thread blocks | Large lattices |
| **GPU 2D Block** | 2D thread block mapping | Efficient lattice mapping |

### Simulation Capabilities

- **Metropolis-Hastings Evolution**: Sample equilibrium states at given temperatures
- **Simulated Annealing**: Find ground states by slowly cooling the system
- **Phase Diagram Mapping**: Sweep J-h parameter space to study magnetic regimes
- **Performance Benchmarking**: Compare CPU vs GPU execution times
- **Trajectory Recording**: Track energy/magnetization evolution during annealing

### Observable Measurements

- Energy (E) and energy density (E/N)
- Magnetization (m) and magnetization density (m/N)
- Initialization and evolution timing

## Project Structure

```
GPU_Ising_model/
├── Ising_simulations.c    # Main simulation (all backends)
├── Regimes_control.c      # Phase diagram mapping (J-h sweeps)
├── Simulated_annealing.c  # Annealing comparison (CPU vs GPU)
├── Run_simulations.py     # Python runner for batch jobs
├── compile_all.sh         # Master build script
├── compile_sa.sh          # Annealing-specific build
├── compile_share_library.sh # Shared library compilation
├── s_functions.h          # Core simulation functions
├── g_functions.h          # GPU/CUDA functions
├── ge_functions.h         # GPU + annealing functions
├── gm_functions.h         # GPU memory-optimized functions
├── Single_CPU/            # Single-threaded CPU implementation
├── OpenMP/                # Parallel CPU implementation
├── GPU/                   # CUDA GPU implementations
├── build/                 # Compiled shared library
├── data/                  # Output data storage
└── results/               # CSV results from batch runs
```

## Requirements

- **CUDA Toolkit** (for GPU code)
- **GCC/G++** with OpenMP support
- **Python 3** (for batch runner)
- **NumPy** (for data analysis)

## Compilation

### Quick Start

```bash
# Compile everything with optimization level 3
./compile_all.sh 1 3
```

This creates three executables:
- `Ising_simulations` - Main multi-backend simulation
- `Regimes_control` - Phase diagram scanner
- `Simulated_annealing` - Annealing comparison

### Shared Library (for Python)

```bash
./compile_share_library.sh 3
```

## Basic Usage

### Run Multi-Backend Simulation

```bash
./Ising_simulations <lattice_size_x> <lattice_size_y>

# Example: 100x100 lattice
./Ising_simulations 100 100
```

This runs the simulation on all backends and prints:
- Final energy and magnetization
- Timing for each backend (initialization, evolution, per-step)

### Phase Diagram Mapping

```bash
./Regimes_control <lattice_size_x> <lattice_size_y>

# Example: 1000x1000 lattice
./Regimes_control 1000 1000
```

Scans J values (0.0 to 3.0) and h values (-1.0 to 1.0), saving magnetization density maps to `data/`.

### Simulated Annealing

```bash
./Simulated_annealing
```

Performs annealing on a 1000×1000 lattice, cooling from T=10.0 to T=0.5. Compares CPU vs GPU execution and saves trajectory data.

### Batch Python Runner

```bash
python3 Run_simulations.py
```

Runs parameter sweeps across multiple lattice sizes and temperatures, saving results to CSV.

## Parameters

Key simulation parameters (hardcoded in sources):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `J` | 1.0 | Spin-spin coupling strength |
| `h` | 0.0-2.0 | External magnetic field |
| `T` | 0.5-10.0 | Temperature |
| `n_steps` | 10,000,000 | MC steps (CPU) or sweeps (GPU) |
| `kB` | 1.0 | Boltzmann constant (normalized) |

### Initialization Types

- **1**: All spins up (+1)
- **2**: All spins down (-1)
- **3**: Random configuration

## Performance Notes

- GPU implementations are significantly faster for large lattices (L > 256)
- GPU 1D Block is recommended for lattices up to ~4000×4000
- GPU 2D Block provides efficient thread mapping for square lattices
- CPU OpenMP scales well on multi-core systems

## Data Output

- **Console**: Real-time progress and timing
- **CSV Files**: `results/ising_results_YYYYMMDD_HHMMSS.csv`
- **Trajectory**: `annealing_trajectory.dat` (energy/magnetization vs. step)
- **Phase Diagrams**: `data/magnetization<L>_<L>_T<T>_type<N>.csv`

## References

- Metropolis algorithm for Ising model: Metropolis et al., 1953
- Critical temperature: Tc = 2.269... (Onsager solution, 1944)
- Simulated annealing: Kirkpatrick et al., 1983
