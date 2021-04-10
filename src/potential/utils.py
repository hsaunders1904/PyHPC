import matplotlib.pyplot as plt
import numpy as np


def gen_particles(N, dist="circle", dtype="float64"):
    if dist.lower() == "circle":
        coords = np.zeros((N, 2))
        r = np.arange(0, N)
        coords[:, 0] = 0.5 + 0.4*np.sin(2*np.pi*r/N + 0.1)
        coords[:, 1] = 0.5 + 0.4*np.cos(2*np.pi*r/N + 0.1)
        return coords.astype(dtype)
    elif dist.lower() == "random":
        return np.random.rand(N, 2).astype(dtype)
    else:
        raise ValueError(f"Unrecognised distribution {dist}.")


def potential_np(particle_coords, grid_resolution, charges):
    # Create mesh xx, yy
    x = np.linspace(0, 1, grid_resolution)
    xx, yy = np.meshgrid(x, x)

    # Create matrix of zeros
    potential_grid = np.zeros((grid_resolution, grid_resolution))

    # Increment the matrix for each particle
    for i, coords in enumerate(particle_coords):
        delta_x = np.square(xx - coords[0])
        delta_y = np.square(yy - coords[1])
        distance = np.sqrt(delta_x + delta_y)
        potential_grid -= charges[i]*np.log(distance)
    return potential_grid
