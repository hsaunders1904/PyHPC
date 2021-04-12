import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from pyhpc.potential import calculate_grid
from pyhpc._cl_utils import get_device
from pyhpc.utils import gen_particles, gen_charges


def run_benchmarks(functions_dict, particle_coords, grid_resolutions, charges):
    func_times = {x: [] for x in functions_dict.keys()}
    for grid_resolution in grid_resolutions:
        for func_name, func in functions_dict.items():
            times = func_times[func_name]
            # Stop benchmarking when the function
            # takes > n seconds
            if len(times) > 1 and times[-1] > MAX_TIME:
                print(f"{func_name} - {grid_resolution} - skipped")
                continue
            start = time.time()
            func(particle_coords, grid_resolution, charges)
            end = time.time()
            func_times[func_name].append(end - start)
            print(f"{func_name} - {grid_resolution} - {end - start:4f}s")
        print("")
    return func_times


def plot_benchmarks(func_times):
    fig, ax = plt.subplots(figsize=(12, 6))
    for func_name, times in func_times.items():
        x = grid_resolutions[:len(times)]
        ax.plot(x, times, label=func_name, marker="o", linestyle="")

        # Fit quadratic to benchmarks; we expect time to scale
        # as grid_resolution^2
        popt, _ = curve_fit(
            _quadratic, x, times, bounds=([0, 0], [np.inf, np.inf])
        )
        x_space = np.linspace(0, x[-1]*1.1, 200)
        ax.plot(
            x_space,
            _quadratic(x_space, *popt),
            linestyle="-.",
            marker="",
            color=ax.lines[-1].get_color()
        )

    cpu_device = get_device("CPU")
    gpu_device = get_device('GPU')
    ax.set_title(
        f"Function Benchmarks\n"
        f"CPU: {cpu_device.name}, {cpu_device.max_compute_units} cores\n"
        f"GPU: {gpu_device.name}, {gpu_device.max_compute_units} cores"
    )
    ax.set_xlabel("Grid Resolution")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(linewidth=0.2)
    ax.legend(ncol=3, loc="upper right")
    return fig


def _quadratic(x, a, b):
    """Quadratic that goes through origin"""
    return a*x**2 + b*x


if __name__ == "__main__":
    num_particles = 10
    particle_coords = gen_particles(num_particles)
    grid_resolutions = [50, 100, 200, 300, 500, 750, 1000, 1500, 3000, 5000]
    charges = gen_charges(num_particles)
    MAX_TIME = 3  # seconds

    functions_dict = {
        "python": lambda *args: calculate_grid(*args, func="python"),
        "numpy": lambda *args: calculate_grid(*args, func="numpy"),
        "numba": lambda *args: calculate_grid(*args, func="numba"),
        "cython": lambda *args: calculate_grid(*args, func="cython"),
        "cpp": lambda *args: calculate_grid(*args, func="cpp", num_threads=1),
        "cpp_omp": (
            lambda *args:
            calculate_grid(*args, func="cpp", num_threads=os.cpu_count())
        ),
        "cl-cpu": (
            lambda *args:
            calculate_grid(*args, func="opencl", device_type="CPU")
        ),
        "cl-gpu": (
            lambda *args:
            calculate_grid(*args, func="opencl", device_type="GPU")
        ),
        "cuda": lambda *args: calculate_grid(*args, func="cuda"),
    }

    times = run_benchmarks(
        functions_dict, particle_coords, grid_resolutions, charges
    )
    fig = plot_benchmarks(times)
    fig.savefig(f"benchmarks-{time.strftime('%Y-%m-%d-%H-%M-%S')}.png")
    plt.show()
