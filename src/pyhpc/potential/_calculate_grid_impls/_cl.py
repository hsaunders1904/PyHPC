import math

import numpy as np
import pyopencl as cl

from pyhpc._cl_utils import get_device
from ._cl_kernel import KERNEL


class _PotentialCL:

    mf = cl.mem_flags

    def __init__(self, kernel, device_type=None, device_name=None):
        # Create a context and compile the OpenCL kernel for the device
        if all([x is None for x in [device_type, device_name]]):
            self.context = cl.create_some_context()
        else:
            device = get_device(device_type, device_name)
            self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.mf = cl.mem_flags
        self.program = cl.Program(self.context, kernel).build()

    def run(self, particle_coords, grid_resolution, charges):
        x_pos = np.ascontiguousarray(particle_coords[:, 0], dtype="float64")
        y_pos = np.ascontiguousarray(particle_coords[:, 1], dtype="float64")
        grid_resolution = np.int32(grid_resolution)
        charges = np.ascontiguousarray(charges, dtype="int32")
        num_particles = np.int32(particle_coords.shape[0])
        potential_grid = np.zeros((1, grid_resolution**2), dtype="float64")

        # Create the OpenCL memory buffers to pass to the kernel
        x_pos_buf = self._get_buffer(x_pos, self.mf.READ_ONLY)
        y_pos_buf = self._get_buffer(y_pos, self.mf.READ_ONLY)
        charges_buf = self._get_buffer(charges, self.mf.READ_ONLY)
        potential_grid_buf = self._get_buffer(
            potential_grid, self.mf.WRITE_ONLY
        )

        global_id_sizes = (
            math.ceil(grid_resolution/4)*4, math.ceil(grid_resolution/4)*4
        )

        kernel_args = (
            x_pos_buf, y_pos_buf, charges_buf, grid_resolution, num_particles,
            potential_grid_buf
        )
        self.program.potential_cl(
            self.queue, global_id_sizes, None, *kernel_args
        )

        cl.enqueue_copy(self.queue, potential_grid, potential_grid_buf)
        return potential_grid.reshape((grid_resolution, grid_resolution))

    def _get_buffer(self, hostbuf, permissions):
        return cl.Buffer(
            self.context, permissions | self.mf.COPY_HOST_PTR, hostbuf=hostbuf
        )


_CPU_PROG = _PotentialCL(KERNEL, device_type="CPU")
_GPU_PROG = _PotentialCL(KERNEL, device_type="GPU")


def _potential_cl(particle_coords, grid_resolution, charges, device_type):
    if device_type == "CPU":
        func = _CPU_PROG.run
    elif device_type == "GPU":
        func = _GPU_PROG.run
    return func(particle_coords, grid_resolution, charges)


def potential_cl_cpu(particle_coords, grid_resolution, charges):
    return _potential_cl(particle_coords, grid_resolution, charges, "CPU")


def potential_cl_gpu(particle_coords, grid_resolution, charges):
    return _potential_cl(particle_coords, grid_resolution, charges, "GPU")
