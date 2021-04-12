import math

import numpy as np
import pyopencl as cl

from potential._cl_utils import get_device
from potential._cl_kernel import KERNEL


class _PotentialCL:

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

    def __call__(self, particle_coords, grid_resolution, charges):
        return self.run(particle_coords, grid_resolution, charges)

    def run(self, particle_coords, grid_resolution, charges):
        x_pos = particle_coords[:, 0].astype("float64")
        y_pos = particle_coords[:, 1].astype("float64")
        x_grid = np.linspace(0, 1, grid_resolution, dtype="float64")
        y_grid = np.linspace(0, 1, grid_resolution, dtype="float64")
        num_particles = np.int32(charges.shape[0])
        potential_grid = np.zeros(grid_resolution**2, dtype="float64")

        # Create the OpenCL memory buffers to pass to the kernerl
        x_pos_buf = self._get_buffer(x_pos, "READ_ONLY")
        y_pos_buf = self._get_buffer(y_pos, "READ_ONLY")
        x_grid_buf = self._get_buffer(x_grid, "READ_ONLY")
        y_grid_buf = self._get_buffer(y_grid, "READ_ONLY")
        charges_buf = self._get_buffer(charges, "READ_ONLY")
        potential_grid_buf = self._get_buffer(potential_grid, "WRITE_ONLY")

        global_id_sizes = (
            math.ceil(grid_resolution/4)*4,
            math.ceil(grid_resolution/4)*4,
        )
        kernel_args = (
            x_pos_buf, y_pos_buf, x_grid_buf, y_grid_buf, charges_buf,
            grid_resolution, num_particles, potential_grid_buf
        )

        self.program.potential_cl(
            self.queue, global_id_sizes, None, *kernel_args
        )

        cl.enqueue_copy(self.queue, potential_grid, potential_grid_buf)
        potential_grid = potential_grid.reshape(
            (grid_resolution, grid_resolution)
        )

        return potential_grid

    def _get_buffer(self, hostbuf, permissions):
        mf = cl.mem_flags
        if permissions == "READ_ONLY":
            p_flag = mf.READ_ONLY
        elif permissions == "WRITE_ONLY":
            p_flag = mf.WRITE_ONLY
        else:
            raise ValueError(
                "Arg 'permission', must be 'READ_ONLY' or 'WRITE_ONLY'."
            )
        return cl.Buffer(
            self.context, p_flag | mf.COPY_HOST_PTR, hostbuf=hostbuf
        )


_CPU_FUNC = _PotentialCL(KERNEL, device_type="CPU")
_GPU_FUNC = _PotentialCL(KERNEL, device_type="GPU")


def _potential_cl(particle_coords, grid_resolution, charges, device_type):
    if device_type == "CPU":
        func = _CPU_FUNC
    elif device_type == "GPU":
        func = _GPU_FUNC

    particle_coords = particle_coords.astype("float64")
    grid_resolution = np.int32(grid_resolution)
    charges = charges.astype("int32")
    return func(particle_coords, grid_resolution, charges)


def potential_cl_cpu(particle_coords, grid_resolution, charges):
    return _potential_cl(particle_coords, grid_resolution, charges, "CPU")


def potential_cl_gpu(particle_coords, grid_resolution, charges):
    return _potential_cl(particle_coords, grid_resolution, charges, "GPU")
