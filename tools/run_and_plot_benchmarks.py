import json
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from pyhpc.potential import calculate_grid
from pyhpc.utils import gen_particles
from pyhpc._cl_utils import get_device

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _time_function(func_to_run, repetitions, *args, **kwargs):
    times = []
    for _ in range(repetitions):
        start = time.time()
        func_to_run(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    return times


def _build_plot_name(prefix):
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    return f"{prefix}-{current_time}.png"


def _quadratic(x, a, b):
    """Quadratic that goes through origin"""
    return a*x**2 + b*x


def run_benchmarks(
    functions_dict,
    particle_coords,
    grid_resolutions,
    charges,
    max_time,
    repetitions=3
):
    func_times = {}
    for func_name, func_kwargs in functions_dict.items():
        func_times[func_name] = []
        for grid_resolution in grid_resolutions:
            times = func_times[func_name]
            # Stop benchmarking when the function takes > n seconds
            if len(times) > 1 and any(x > max_time for x in times[-1]):
                print(f"{func_name} - {grid_resolution} - skipped")
                break

            func_args = (particle_coords, grid_resolution, charges)
            elapsed_times = _time_function(
                calculate_grid, repetitions, *func_args, **func_kwargs
            )
            func_times[func_name].append(elapsed_times)

            mean_time = np.mean(elapsed_times)
            print(f"{func_name} - {grid_resolution} - {mean_time:4f}s")
        print("")
    return func_times


def plot_benchmarks(results_dict, labels=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, times in results_dict.items():
        if not labels or label in labels:
            grid_res = grid_resolutions[:len(times)]
            line = _plot_errobar(ax, grid_res, times, label)
            _plot_best_fit(ax, grid_res, times, color=line[0].get_color())

    ax.set_title(_generate_benchmark_title(num_particles))
    ax.set_xlabel("Grid Resolution")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(linewidth=0.2)
    ax.legend(ncol=3, loc="upper right")
    return fig


def _plot_errobar(ax, grid_resolutions, times, label):
    mean_times = [np.mean(time_set) for time_set in times]
    std_errs = [np.std(time_set, ddof=1) for time_set in times]
    line = ax.errorbar(
        grid_resolutions,
        mean_times,
        std_errs,
        label=label,
        marker="o",
        linestyle="",
    )
    return line


def _plot_best_fit(ax, grid_resolutions, times, **kwargs):
    # Fit quadratic to benchmarks; we expect time to scale
    # as grid_resolution^2
    popt, _ = curve_fit(
        _quadratic,
        grid_resolutions,
        [np.mean(time_set) for time_set in times],
        bounds=([0, 0], [np.inf, np.inf]),
        # +1 to avoid divsion by zero errors
        sigma=[np.std(time_set, ddof=1) + 1 for time_set in times]
    )
    x_space = np.linspace(0, grid_resolutions[-1]*1.1, 200)
    return ax.plot(
        x_space,
        _quadratic(x_space, *popt),
        linestyle="-.",
        marker="",
        **kwargs
    )


def _generate_benchmark_title(num_particles):
    cpu_device = get_device("CPU")
    gpu_device = get_device('GPU')
    return (
        f"Function Benchmarks ({num_particles} particles)\n"
        f"CPU: {cpu_device.name}, {cpu_device.max_compute_units} cores\n"
        f"GPU: {gpu_device.name}, {gpu_device.max_compute_units} cores"
    )


def _create_plot_dir():
    plot_out_dir = os.path.join(ROOT_DIR, ".benchmarks")
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    new_dir = os.path.join(plot_out_dir, f"funcs-{current_time}")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir, exist_ok=True)
    return new_dir


def _get_git_revision():
    revision_raw = subprocess.check_output(f"git -C {ROOT_DIR} rev-parse HEAD")
    return revision_raw.decode().strip()


def _get_modified_files():
    modified_files = subprocess.check_output(
        f"git -C {ROOT_DIR} diff --name-only"
    ).decode().split("\n")
    return [file_name for file_name in modified_files if file_name]


def _write_json(dict_object, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_object, f, indent=2)


def _plot_benchmarks_recursive(labels, plot_dir):
    to_plot = []
    for i, label in enumerate(labels):
        to_plot.append(label)
        fig = plot_benchmarks(times, labels=to_plot)
        out_name = _build_plot_name(f"sub_bench-{i}")
        fig.savefig(os.path.join(plot_dir, out_name))


if __name__ == "__main__":
    num_particles = 10
    particle_coords = gen_particles(num_particles)
    grid_resolutions = [50, 100, 200, 300, 500, 750, 1000, 1500, 3000, 5000]
    charges = np.ones(num_particles, dtype=np.int32)
    max_time = 3  # seconds
    repetitions = 5

    functions_dict = {
        "python": {
            "func": "python"
        },
        "numpy": {
            "func": "numpy"
        },
        "cython": {
            "func": "cython"
        },
        "numba": {
            "func": "numba"
        },
        "cpp": {
            "func": "cpp"
        },
        "cpp-omp": {
            "func": "cpp",
            "num_threads": os.cpu_count()
        },
        "cl-cpu": {
            "func": "opencl",
            "device_type": "CPU"
        },
        "cl-gpu": {
            "func": "opencl",
            "device_type": "GPU"
        },
        "cuda": {
            "func": "cuda"
        },
    }

    times = run_benchmarks(
        functions_dict,
        particle_coords,
        grid_resolutions,
        charges,
        max_time,
        repetitions=repetitions
    )
    metadata = {
        "args": {
            "num_particles": num_particles,
            "charges": charges.tolist(),
            "particle_coords": particle_coords.tolist()
        },
        "grid_resolutions": grid_resolutions,
        "git": {
            "revision": _get_git_revision(),
            "modified": _get_modified_files()
        },
    }
    times.update(metadata)

    plot_dir = _create_plot_dir()
    _write_json(times, os.path.join(plot_dir, "results.json"))
    _plot_benchmarks_recursive(functions_dict.keys(), plot_dir)
