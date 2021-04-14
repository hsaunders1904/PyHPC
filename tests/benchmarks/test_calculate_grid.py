import os

import numpy as np
import pytest

from pyhpc.potential import calculate_grid
from pyhpc.utils import gen_particles

arg_combos = {
    "python": {
        "func": "python"
    },
    "numpy": {
        "func": "numpy",
    },
    "numba": {
        "func": "numba",
    },
    "cython": {
        "func": "cython",
    },
    "cpp_1_thread": {
        "num_threads": 1
    },
    "cpp_max_threads": {
        "func": "cpp",
        "num_threads": os.cpu_count()
    },
    "opencl_cpu": {
        "func": "opencl",
        "device_type": "CPU"
    },
    "opencl_gpu": {
        "func": "opencl",
        "device_type": "GPU"
    },
    "cuda": {
        "func": "cuda"
    }
}


class TestPotentialBenchmark:

    @pytest.mark.parametrize(
        "kwargs", arg_combos.values(), ids=arg_combos.keys()
    )
    def test_benchmark_calculate_grid_small(self, kwargs, benchmark):
        num_particles, grid_resolution = 10, 100
        particle_coords, charges = self.get_particles(num_particles)

        args = (particle_coords, grid_resolution, charges)
        grid = benchmark(calculate_grid, *args, **kwargs)
        assert grid.shape == (grid_resolution, grid_resolution)

    @pytest.mark.parametrize(
        "kwargs", arg_combos.values(), ids=arg_combos.keys()
    )
    def test_benchmark_calculate_grid_large(self, kwargs, benchmark):
        if kwargs.get("func") == "python":
            pytest.skip("Python implementation only run on small grid")

        num_particles, grid_resolution = 20, 1024
        particle_coords, charges = self.get_particles(num_particles)

        args = (particle_coords, grid_resolution, charges)
        grid = benchmark(calculate_grid, *args, **kwargs)
        assert grid.shape == (grid_resolution, grid_resolution)

    @staticmethod
    def get_particles(num_particles):
        particle_coords = gen_particles(num_particles)
        charges = np.array(
            [-1 if i % 2 == 0 else 1 for i in range(num_particles)],
            dtype="int32",
        )
        return particle_coords, charges
