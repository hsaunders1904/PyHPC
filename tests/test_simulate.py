import os

import numpy as np

from pyhpc.potential import simulate_particles


class TestSimulateParticles:

    DATA_FILE_NAME = "ref_100x100x10_frames.ndarray"

    @classmethod
    def setup_class(cls):
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_100x100x10_file = os.path.join(test_data_dir, cls.DATA_FILE_NAME)
        with open(ref_100x100x10_file, "rb") as f:
            cls.ref_100x100x10_grid = np.fromfile(f).reshape((100, 100, 10))

    def test_simulate_particles_equal_to_100x100x10_ref_file(self):
        ref_params = self.get_ref_data_params()
        out = simulate_particles(*ref_params)
        assert np.allclose(out, self.ref_100x100x10_grid)

    @staticmethod
    def get_ref_data_params():
        # Set up params such that we have particles travelling toward
        # the image border, so we check that particles' velocities are
        # reversed when hitting the boundary
        particle_coords = np.array([[0.54, 0.9], [0.9, 0.46], [0.46, 0.1],
                                    [0.1, 0.54]])
        charges = np.array([1, -1, 1, -1], dtype=np.int32)
        n_frames = 10
        grid_resolution = 100
        delta_t = 0.2
        initial_velocities = np.array([[0, 1], [1, -0.5], [0.2, -0.1], [0, 0]])
        return (
            particle_coords,
            grid_resolution,
            charges,
            initial_velocities,
            n_frames,
            delta_t,
        )

    @staticmethod
    def generate_reference_file():
        ref_params = TestSimulateParticles.get_ref_data_params()
        pot_grid_np = simulate_particles(*ref_params)
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        ref_file_path = os.path.join(
            test_data_dir, TestSimulateParticles.DATA_FILE_NAME
        )
        with open(ref_file_path, "wb") as f:
            f.write(pot_grid_np.tobytes())
        print(f"Reference file {ref_file_path} created.")


if __name__ == "__main__":
    TestSimulateParticles.generate_reference_file()
