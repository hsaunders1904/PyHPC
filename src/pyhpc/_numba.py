import numba
import numpy as np


@numba.jit(
    numba.double[:, :](numba.double[:, :], numba.int32, numba.int32[:]),
    nopython=True,
    parallel=True,
    nogil=True,
    cache=True
)
def potential_numba(particle_coords, grid_resolution, charges):
    potential_grid = np.zeros((grid_resolution, grid_resolution),
                              dtype="float64")

    grid_step_denom = grid_resolution - 1
    for n in range(charges.shape[0]):
        for i in range(grid_resolution):
            for j in numba.prange(grid_resolution):
                delta_x = i/grid_step_denom - particle_coords[n, 0]
                delta_y = j/grid_step_denom - particle_coords[n, 1]
                euclid_distance = np.sqrt(delta_x**2 + delta_y**2)
                potential_grid[j, i] -= charges[n]*np.log(euclid_distance)
    return potential_grid
