import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def potential_numba(particle_coords, grid_resolution, charges):
    potential_grid = np.zeros((grid_resolution, grid_resolution))

    # Below does the equivalent of np.meshgrid(x, x), which is not
    # allowed in numba
    space = np.repeat(np.linspace(0, 1, grid_resolution), grid_resolution)
    yy = space.reshape((grid_resolution, grid_resolution))
    xx = yy.T

    # Increment the matrix for each particle
    for i in numba.prange(len(charges)):  # note TBB is required for prange
        delta_x = np.square(xx - particle_coords[i, 0])
        delta_y = np.square(yy - particle_coords[i, 1])
        distance = np.sqrt(delta_x + delta_y)
        potential_grid -= charges[i]*np.log(distance)

    return potential_grid
