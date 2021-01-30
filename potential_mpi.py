import math

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from utils import potential_np, gen_particles, plot_potential_grid


class MSG:
    COORDS = 1
    CHARGES = 2
    GRID_RES = 3
    POT_GRID = 4


def _distribute_particles(particle_coord_list, charges_list, grid_resolution,
                          max_num_workers, comm):
    for i in range(1, max_num_workers):
        comm.send(particle_coord_list[i], dest=i, tag=MSG.COORDS)
        comm.send(charges_list[i], dest=i, tag=MSG.CHARGES)
        comm.send(grid_resolution, dest=i, tag=MSG.GRID_RES)

    sub_particle_coords = particle_coord_list[0]
    sub_charges = charges_list[0]
    return sub_particle_coords, sub_charges


def _divide_particles(coords, charges, num_workers):
    num_particles = coords.shape[0]
    coords_per_worker = math.floor(num_particles / num_workers)
    leftover = num_particles % num_workers
    split_coords = []
    split_charges = []
    allocated = 0
    for i in range(num_workers):
        num_coords = coords_per_worker
        if leftover > 0:
            num_coords = coords_per_worker + 1
        sub_coords = coords[allocated:(allocated + num_coords)]
        sub_charges = charges[allocated:(allocated + num_coords)]
        if sub_coords.shape[0] > 0:
            split_coords.append(sub_coords)
            split_charges.append(sub_charges)
            leftover -= 1
            allocated += len(sub_coords)
        else:
            break

    return split_coords, split_charges


def _worker_needed(comm, rank, max_num_workers):
    max_num_workers = comm.bcast(max_num_workers, root=0)
    return rank < max_num_workers


def potential_mpi(particle_coords, grid_resolution, charges):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # Rank 0 splits the particle coordinates and works out the max number of
    # workers we need from those available
    if rank == 0:
        split_coords, split_charges = _divide_particles(
            particle_coords, charges, world_size)
        max_num_workers = len(split_coords)
    else:
        max_num_workers = None

    if not _worker_needed(comm, rank, max_num_workers):
        import sys
        sys.exit()

    # The root worker sends particle information to other workers
    if rank == 0:
        sub_particle_coords, sub_charges = _distribute_particles(
            split_coords, split_charges, grid_resolution, max_num_workers,
            comm)
    else:
        # Workers receive particle information from workers
        sub_particle_coords = comm.recv(source=0, tag=MSG.COORDS)
        sub_charges = np.array(comm.recv(source=0, tag=MSG.CHARGES))
        grid_resolution = comm.recv(source=0, tag=MSG.GRID_RES)

    # Each worker calculates the potential grid for its given particles
    # We then accumulate the result from each worker
    pot_grid = potential_np(sub_particle_coords, grid_resolution, sub_charges)
    pot_grid = comm.allreduce(pot_grid)

    return pot_grid


if __name__ == "__main__":
    NUM_PARTICLES = 80
    GRID_RESOLUTION = 1000
    particle_coords = gen_particles(NUM_PARTICLES, "circle")
    charges = np.random.choice(np.int32([-1, 1]), (NUM_PARTICLES, ))

    pot_grid = potential_mpi(particle_coords, GRID_RESOLUTION, charges)

    if MPI.COMM_WORLD.rank == 0:
        plot_potential_grid(pot_grid)
        plt.show()
