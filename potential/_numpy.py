import numpy as np


def potential_np(particle_coords, grid_resolution, charges):
    # Create mesh xx, yy
    x = np.linspace(0, 1, grid_resolution)
    xx, yy = np.meshgrid(x, x)

    potential_grid = np.zeros((grid_resolution, grid_resolution))
    # Increment the grid for each particle
    for coords, charge in zip(particle_coords, charges):
        delta_x = np.square(xx - coords[0])
        delta_y = np.square(yy - coords[1])
        distance = np.sqrt(delta_x + delta_y)
        potential_grid -= charge*np.log(distance)

    return potential_grid
