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

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            grid_pos = [i/(grid_resolution - 1), j/(grid_resolution - 1)]
            potential_grid[
                j, i] = V_potential(grid_pos, particle_coords, charges)
    return potential_grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    particle_coords = np.array([[0.25, 0.75], [0.5, 0.25], [0.75, 0.75]])
    grid_resolution = 10
    charges = np.array([-1, 1, -1])

    pot_grid = potential_py(particle_coords, grid_resolution, charges)

    plt.imshow(pot_grid, origin="lower")
    plt.show()
