import numpy as np
import pycuda.autoinit  # must be imported so Cuda can compile things
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from potential._cuda_kernel import FUNC_NAME, KERNEL


class _PotentialCuda:

    def __init__(self, kernel, func_name, dtype="float64"):
        self._dtype = dtype
        self.module = SourceModule(kernel)
        self.func = self.module.get_function(func_name)

    def __call__(self, particle_coords, grid_resolution, charges):
        return self.run(particle_coords, grid_resolution, charges)

    def run(self, particle_coords, grid_resolution, charges):
        # All numpy arrays must be C contiguous to pass to Cuda
        x_pos = particle_coords[:, 0].copy(order="C").astype(self._dtype)
        y_pos = particle_coords[:, 1].copy(order="C").astype(self._dtype)
        particle_coords = particle_coords.copy(order="C").astype(self._dtype)
        grid_space = np.linspace(0, 1, grid_resolution, dtype=self._dtype)
        num_particles = np.int32(len(particle_coords))

        potential_grid = np.zeros((1, grid_resolution**2), dtype=self._dtype)

        # Generate the memory buffers/arguments to copy to the GPU
        args = (
            drv.In(x_pos), drv.In(y_pos),
            drv.In(grid_space), drv.In(grid_space), drv.In(charges),
            np.int32(grid_resolution), num_particles, drv.Out(potential_grid)
        )

        # Number of threads along each axis should divide by 4
        num_axis_threads = np.ceil(grid_resolution/4)*4
        threads_per_block = 256

        # Get the number of threads in each direction in each block.
        # Since we're calculating on a square, each direction is the
        # square root of the total number of threads in the block
        block_axis_size = np.sqrt(threads_per_block)

        # The number of threads to run in each direction in each block
        # to cover the number of "pixels" on each axis
        grid_threads = (
            int(np.ceil(grid_resolution/block_axis_size)),
            int(np.ceil(grid_resolution/block_axis_size)), 1
        )
        # The number of blocks on which to run threads on each axis
        block_size = (
            int(np.ceil(num_axis_threads/grid_threads[0])),
            int(np.ceil(num_axis_threads/grid_threads[1])), 1
        )
        self.func(*args, block=block_size, grid=grid_threads)
        grid = np.reshape(potential_grid, (grid_resolution, grid_resolution))
        return grid


_FUNC = _PotentialCuda(KERNEL, FUNC_NAME)


def potential_cuda(particle_coords, grid_resolution, charges):
    return _FUNC(particle_coords, grid_resolution, charges)
