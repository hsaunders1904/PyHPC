import os

import numpy as np
import pytest

from potential.grid import calculate_grid

arg_combos = {
    "python": {
        "func": "python",
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


class TestGrid:

    ref_10x10_grid = None

    @classmethod
    def setup_class(cls):
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_10x10_file = os.path.join(test_data_dir, "ref_10x10_grid.ndarray")
        with open(ref_10x10_file, "rb") as f:
            cls.ref_10x10_grid = np.fromfile(f).reshape((10, 10))

    @pytest.mark.parametrize(
        "kwargs", arg_combos.values(), ids=arg_combos.keys()
    )
    def test_potential_grid_equals_10x10_ref_file(self, kwargs):
        ref_params = self.get_ref_data_params()
        out = calculate_grid(*ref_params, **kwargs)
        assert np.allclose(out, self.ref_10x10_grid)

    @staticmethod
    def get_ref_data_params():
        particle_coords = np.array([[0.25, 0.75], [0.5, 0.25], [0.75, 0.75]])
        grid_resolution = 10
        charges = np.array([-1, 1, -1])
        return particle_coords, grid_resolution, charges

    @staticmethod
    def generate_reference_file():
        ref_params = TestGrid.get_ref_data_params()
        # Use the Python implementation to generate regression test data
        pot_grid_np = calculate_grid(*ref_params, func="python")

        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_10x10_file = os.path.join(test_data_dir, "ref_10x10_grid.ndarray")
        with open(ref_10x10_file, "wb") as f:
            f.write(pot_grid_np.tobytes())


if __name__ == "__main__":
    TestGrid.generate_reference_file()
