import os

import numpy as np
import pytest

from potential import _potential as potential

POTENTIAL_FUNCTIONS = {
    potential.potential_cpp, potential.potential_np, potential.potential_py,
    potential.potential_numba, potential.potential_cl_cpu,
    potential.potential_cl_gpu, potential.potential_cuda,
    potential.potential_cython
}


class TestPotential:

    ref_10x10_grid = None

    @classmethod
    def setup_class(cls):
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_10x10_file = os.path.join(test_data_dir, "ref_10x10_grid.ndarray")
        with open(ref_10x10_file, "rb") as f:
            cls.ref_10x10_grid = np.fromfile(f).reshape((10, 10))

    @pytest.mark.parametrize("func", POTENTIAL_FUNCTIONS)
    def test_potential_grid_equals_10x10_ref_file(self, func):
        ref_params = TestPotential.get_ref_data_params()
        out = func(*ref_params)
        assert np.allclose(out, self.ref_10x10_grid)

    @staticmethod
    def get_ref_data_params():
        particle_coords = np.array([[0.25, 0.75], [0.5, 0.25], [0.75, 0.75]])
        grid_resolution = 10
        charges = np.array([-1, 1, -1])
        return particle_coords, grid_resolution, charges

    @staticmethod
    def generate_reference_file():
        ref_params = TestPotential.get_ref_data_params()
        # Use the Python implementation to generate regression test data
        pot_grid_np = potential.potential_py(*ref_params)

        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_10x10_file = os.path.join(test_data_dir, "ref_10x10_grid.ndarray")
        with open(ref_10x10_file, "wb") as f:
            f.write(pot_grid_np.tobytes())


if __name__ == "__main__":
    TestPotential.generate_reference_file()
