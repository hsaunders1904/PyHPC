from pyhpc._cpp import potential_cpp
from pyhpc._numpy import potential_np
from pyhpc._py import potential_py
from pyhpc._numba import potential_numba
from pyhpc._cython import potential_cython
from pyhpc._cl import potential_cl_cpu, potential_cl_gpu
from pyhpc._cuda import potential_cuda


def calculate_grid(
    particle_coords, grid_resolution, charges, func="numpy", **kwargs
):
    """
    Calculate the electrical potential generated by the particles at the
    given coordinates.

    The grid is calculated between 0 and 1 over the x and y axes, so, in
    order for your particles to be visible, they should have their
    positions between 0 and 1.

    :param particle_coords: 2xN array of the x, y coordinates
        of the particles (where N is the number of particles).
    :type particle_coords: np.ndarray
    :param grid_resolution: The resolution (in each dimension) of the
        grid over which to calculate the potential.
        Each dimension has the same resolution, hence the calculation is
        performed over a square.
    :type grid_resolution: int
    :param charges: 1xN array specifying the charge of each particle.
    :type charges: np.array
    :param func: The method to use to calculate the grid, one of:
        - "python" or "py"
        - "numpy" or "np"
        - "numba"
        - "cython"
        - "cpp"
        - "opencl" or "cl"
        - "cuda"
        Default is "numpy".
    :type func: str

    :key num_threads: The number of threads to use to perform the
        calculation, only applies if 'func' is 'cpp'. Default is 1.
    :type num_threads: int
    :key device_type: The device type to run the calculation on,
        either 'CPU' or 'GPU'. This option only applies if 'func' is
        'opencl'. Default is 'CPU'.
    :type device_type: str
    """
    args = (particle_coords, grid_resolution, charges)
    if func.lower() in ["py", "python"]:
        return potential_py(*args)
    elif func.lower() in ["np", "numpy"]:
        return potential_np(*args)
    elif func.lower() == "numba":
        return potential_numba(*args)
    elif func.lower() == "cython":
        return potential_cython(*args)
    elif func.lower() == "cpp":
        return potential_cpp(*args, **kwargs)
    elif func.lower() in ["cl", "opencl"]:
        device_type = kwargs.pop("device_type", "CPU")
        if device_type == "CPU":
            return potential_cl_cpu(*args)
        elif device_type == "GPU":
            return potential_cl_gpu(*args)
        else:
            raise ValueError(
                f"Invalid device type. Must be on of 'CPU' or 'GPU', "
                f"found {device_type}."
            )
    elif func.lower() == "cuda":
        return potential_cuda(*args)
    else:
        raise ValueError("Invalid value for 'func'.")