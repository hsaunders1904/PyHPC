import numba
import numpy as np


@numba.jit(nopython=True)
def potential_numba(particle_coords, grid_resolution, charges):
    potential_grid = np.zeros((grid_resolution, grid_resolution))

    # Below does the equivalent of np.meshgrid(x, x), which is not
    # allowed in numba
    space = np.repeat(np.linspace(0, 1, grid_resolution), grid_resolution)
    yy = space.reshape((grid_resolution, grid_resolution))
    xx = yy.T

    # Increment the matrix for each particle
    num_particles = particle_coords.shape[0]
    for coords, charge in zip(particle_coords, charges):
        delta_x = np.square(xx - coords[0])
        delta_y = np.square(yy - coords[1])
        distance = np.sqrt(delta_x + delta_y)
        potential_grid -= charge*np.log(distance)

    return potential_grid
