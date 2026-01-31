import ctypes
import csv
from itertools import product
from datetime import datetime

# ------------------------------------------------------------
# Load shared library
# ------------------------------------------------------------

lib = ctypes.CDLL("./build/libising.so")

# ------------------------------------------------------------
# Observables struct
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Declare function signatures
# ------------------------------------------------------------

def bind(fn):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int
    ]
    fn.restype = Observables
    return fn

run_cpu        = bind(lib.run_ising_simulation)
run_openmp    = bind(lib.run_ising_simulation_openmp)
run_gpu       = bind(lib.run_ising_simulation_gpu)
run_gpu_eff   = bind(lib.run_ising_simulation_efficient_gpu)

BACKENDS = {
    "cpu_1": run_cpu,
    "cpu_openmp": run_openmp,
    "gpu": run_gpu,
    "gpu_efficient": run_gpu_eff,
}

# ------------------------------------------------------------
# Parameter space
# ------------------------------------------------------------

lattice_sizes = [64, 128]      # square lattices
J_values      = [1.0]
h_values      = [0.0, 0.5]
T_values      = [0.5, 1.0, 2.0]
init_types    = {
    # 1: "all_up",
    # 2: "all_down",
    3: "random",
}

kB = 1.0
n_steps = 1_000   # MC steps (CPU) or sweeps-derived internally

# ------------------------------------------------------------
# Output CSV
# ------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"results/ising_results_{timestamp}.csv"

csv_fields = [
    "backend",
    "L",
    "init_type",
    "J",
    "h",
    "T",
    "n_steps",
    "E",
    "e_density",
    "m",
    "m_density",
    "init_time",
    "mh_time",
    "mh_time_per_step",
]

# ------------------------------------------------------------
# Run sweep
# ------------------------------------------------------------

with open(csv_name, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()

    for backend_name, backend_fn in BACKENDS.items():
        print(f"\n=== Running backend: {backend_name} ===")

        for L, (type_id, type_name), J, h, T in product(
            lattice_sizes,
            init_types.items(),
            J_values,
            h_values,
            T_values,
        ):
            print(
                f"Backend={backend_name:14s} "
                f"L={L:4d} "
                f"init={type_name:9s} "
                f"J={J:.2f} h={h:.2f} T={T:.2f}"
            )

            obs = backend_fn(
                L, L,
                type_id,
                J, h, kB, T,
                n_steps
            )

            writer.writerow({
                "backend": backend_name,
                "L": L,
                "init_type": type_name,
                "J": J,
                "h": h,
                "T": T,
                "n_steps": n_steps,
                "E": obs.E,
                "e_density": obs.e_density,
                "m": obs.m,
                "m_density": obs.m_density,
                "init_time": obs.initialization_time,
                "mh_time": obs.MH_evolution_time,
                "mh_time_per_step": obs.MH_evolution_time_over_steps,
            })

print(f"\nResults written to {csv_name}")
