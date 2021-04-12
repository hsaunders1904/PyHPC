import numpy as np


def V_potential(grid_pos, particle_coords, k):
    """
    Determine potential at a given grid point by
    summing the contribution from each particle
    """
    V = 0
    for j in range(len(particle_coords)):
        dist = np.linalg.norm(grid_pos - particle_coords[j])
        if dist > 0:
            V -= k[j]*np.log(dist)
        else:
            V += k[j]*np.inf
    return V


def potential_py(particle_coords, grid_resolution, charges):
    potential_grid = np.zeros((grid_resolution, grid_resolution))

    grid_step_denom = (grid_resolution - 1)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            grid_pos = [i/grid_step_denom, j/grid_step_denom]
            potential_grid[
                j, i] = V_potential(grid_pos, particle_coords, charges)
    return potential_grid
