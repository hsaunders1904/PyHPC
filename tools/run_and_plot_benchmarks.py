import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from pyhpc.potential import calculate_grid
from pyhpc.utils import gen_particles, gen_charges
from pyhpc._cl_utils import get_device


def run_benchmarks(
    functions_dict,
    particle_coords,
    grid_resolutions,
    charges,
    max_time,
    plot=False,
    plot_out_dir="."
):
    if plot:
        current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
        new_dir_name = f"funcs-{current_time}"
        new_dir = os.path.join(plot_out_dir, new_dir_name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)
    func_times = {}
    func_std = {}
    for i, (func_name, func) in enumerate(functions_dict.items()):
        func_times[func_name] = []
        func_std[func_name] = []
        for grid_resolution in grid_resolutions:
            times = func_times[func_name]
            # Stop benchmarking when the function takes > n seconds
            if len(times) > 1 and times[-1] > max_time:
                print(f"{func_name} - {grid_resolution} - skipped")
                break

            elapsed_time, std_dev = _time_function(
                func, 3, particle_coords, grid_resolution, charges
            )
            func_times[func_name].append(elapsed_time)
            func_std[func_name].append(std_dev)
            print(f"{func_name} - {grid_resolution} - {elapsed_time:4f}s")
        print("")
        if plot:
            fig = plot_benchmarks(func_times, func_std, len(particle_coords))
            out_name = _build_plot_name(f"sub_bench-{i}{func_name}")
            fig.savefig(os.path.join(new_dir, out_name))
    return func_times, func_std


def plot_benchmarks(func_times, func_std_devs, num_particles):
    fig, ax = plt.subplots(figsize=(12, 6))
    for func_name, times in func_times.items():
        x = grid_resolutions[:len(times)]
        ax.errorbar(
            x,
            times,
            yerr=func_std_devs[func_name],
            label=func_name,
            marker="o",
            linestyle=""
        )

        # Fit quadratic to benchmarks; we expect time to scale
        # as grid_resolution^2
        popt, _ = curve_fit(
            _quadratic,
            x,
            times,
            bounds=([0, 0], [np.inf, np.inf]),
            # +1 to avoid divsion by zero errors
            sigma=[x + 1 for x in func_std_devs[func_name]]
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
        f"Function Benchmarks ({num_particles} particles)\n"
        f"CPU: {cpu_device.name}, {cpu_device.max_compute_units} cores\n"
        f"GPU: {gpu_device.name}, {gpu_device.max_compute_units} cores"
    )
    ax.set_xlabel("Grid Resolution")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(linewidth=0.2)
    ax.legend(ncol=3, loc="upper right")
    return fig


def _time_function(func, repetitions, *args):
    times = []
    for _ in range(repetitions):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times, ddof=1)


def _build_plot_name(prefix):
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    return f"{prefix}-{current_time}.png"


def _quadratic(x, a, b):
    """Quadratic that goes through origin"""
    return a*x**2 + b*x


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    num_particles = 10
    particle_coords = gen_particles(num_particles)
    grid_resolutions = [50, 100, 200, 300, 500, 750, 1000, 1500, 3000, 5000]
    charges = gen_charges(num_particles)
    max_time = 4  # seconds

    functions_dict = {
        "python": lambda *args: calculate_grid(*args, func="python"),
        "numpy": lambda *args: calculate_grid(*args, func="numpy"),
        "numba": lambda *args: calculate_grid(*args, func="numba"),
        "cython": lambda *args: calculate_grid(*args, func="cython"),
        "cpp": lambda *args: calculate_grid(*args, func="cpp", num_threads=1),
        "cpp-omp": (
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
        functions_dict,
        particle_coords,
        grid_resolutions,
        charges,
        max_time,
        plot=True,
        plot_out_dir=os.path.join(ROOT_DIR, ".benchmarks")
    )
