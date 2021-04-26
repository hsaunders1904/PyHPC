import numpy as np


def potential_np(particle_coords, grid_resolution, charges):
    # Create mesh xx, yy
    x = np.linspace(0, 1, grid_resolution)
    xx, yy = np.meshgrid(x, x)

    # Allocating an empty array and using the "out" kwarg prevents
    # creation of new objects, which boosts performance
    field_increment = np.empty((grid_resolution, grid_resolution))

    potential_grid = np.zeros((grid_resolution, grid_resolution))
    # Increment the grid for each particle
    for coords, charge in zip(particle_coords, charges):
        np.hypot(xx - coords[0], yy - coords[1], out=field_increment)
        np.log(field_increment, out=field_increment)
        field_increment *= charge
        potential_grid -= field_increment
    return potential_grid
