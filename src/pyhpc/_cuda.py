import numpy as np
import pycuda.autoinit  # must be imported so Cuda can compile things
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pyhpc._cuda_kernel import FUNC_NAME, KERNEL


class _PotentialCuda:

    def __init__(self, kernel, func_name, dtype="float64"):
        self._dtype = dtype
        self.module = SourceModule(kernel)
        self.func = self.module.get_function(func_name)

    def run(self, particle_coords, grid_resolution, charges):
        # All numpy arrays must be C contiguous to pass to Cuda
        x_pos = np.ascontiguousarray(particle_coords[:, 0], dtype=self._dtype)
        y_pos = np.ascontiguousarray(particle_coords[:, 1], dtype=self._dtype)
        grid_resolution = np.int32(grid_resolution)
        charges = np.ascontiguousarray(charges, dtype="int32")
        num_particles = np.int32(particle_coords.shape[0])
        potential_grid = np.zeros((1, grid_resolution**2), dtype=self._dtype)

        threads_per_block = 256
        grid_threads = self._get_grid_threads(
            grid_resolution, threads_per_block
        )
        block_size = self._get_block_size(grid_threads, grid_resolution)

        # Generate the memory buffers/arguments to copy to the GPU
        args = (
            drv.In(x_pos),
            drv.In(y_pos),
            drv.In(charges),
            grid_resolution,
            num_particles,
            drv.Out(potential_grid),
        )
        self.func(*args, block=block_size, grid=grid_threads)
        return np.reshape(potential_grid, (grid_resolution, grid_resolution))

    @staticmethod
    def _get_grid_threads(grid_resolution, threads_per_block):
        # Get the number of threads in each direction in each block.
        # Since we're calculating on a square, each direction is the
        # square root of the total number of threads in the block
        block_axis_size = np.sqrt(threads_per_block)

        # The number of threads to run in each direction in each block
        # to cover the number of "pixels" on each axis
        return (
            int(np.ceil(grid_resolution/block_axis_size)),
            int(np.ceil(grid_resolution/block_axis_size)), 1
        )

    @staticmethod
    def _get_block_size(grid_threads, grid_resolution):
        # Number of threads along each axis should divide by 4
        num_axis_threads = np.ceil(grid_resolution/4)*4
        # The number of blocks on which to run threads on each axis
        return (
            int(np.ceil(num_axis_threads/grid_threads[0])),
            int(np.ceil(num_axis_threads/grid_threads[1])), 1
        )


_CUDA_PROG = _PotentialCuda(KERNEL, FUNC_NAME)


def potential_cuda(particle_coords, grid_resolution, charges):
    return _CUDA_PROG.run(particle_coords, grid_resolution, charges)
