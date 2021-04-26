import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from pyhpc.plotting import plot_potential_grid
from pyhpc.potential import calculate_grid
from pyhpc.utils import gen_particles


class MSG(Enum):
    COORDS = 1
    CHARGES = 2
    GRID_RES = 3
    POT_GRID = 4


def run_and_plot(num_particles, grid_resolution, dist="random", func="numpy"):
    particle_coords = gen_particles(num_particles, dist)
    charges = np.random.choice(np.int32([-1, 1]), (num_particles, ))

    pot_grid = potential_mpi(particle_coords, grid_resolution, charges, func)

    if MPI.COMM_WORLD.rank == 0:
        plot_potential_grid(pot_grid)
        plt.show()


def potential_mpi(particle_coords, grid_resolution, charges, func="numpy"):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # Rank 0 splits the particle coordinates and works out the max number of
    # workers we need from those available
    if rank == 0:
        split_coords, split_charges = _divide_particles(
            particle_coords, charges, world_size
        )
        max_num_workers = len(split_coords)
    else:
        max_num_workers = None

    worker_is_needed = _worker_needed(comm, rank, max_num_workers)
    # Create a new communicator with only the required nodes, we
    # can then exit nodes that aren't required and perform collective
    # operations using this new communicator
    split_comm = MPI.Comm.Split(comm, worker_is_needed, rank)
    if not worker_is_needed:
        return

    # The root worker sends particle information to other workers
    if rank == 0:
        sub_particle_coords, sub_charges = _distribute_particles(
            split_coords, split_charges, grid_resolution, max_num_workers,
            split_comm
        )
    else:
        # Workers receive particle information from workers
        sub_particle_coords = split_comm.recv(source=0, tag=MSG.COORDS.value)
        sub_charges = np.array(
            split_comm.recv(source=0, tag=MSG.CHARGES.value)
        )
        grid_resolution = split_comm.recv(source=0, tag=MSG.GRID_RES.value)

    # Each worker calculates the potential grid for its given particles
    # We then accumulate the result from each worker
    local_pot_grid = calculate_grid(
        sub_particle_coords, grid_resolution, sub_charges, func=func
    ).astype(np.float64)

    # pot_grid = split_comm.allreduce(pot_grid)
    pot_grid = np.empty_like(local_pot_grid)
    split_comm.Reduce([local_pot_grid, MPI.DOUBLE], [pot_grid, MPI.DOUBLE],
                      root=0)
    return pot_grid


def _divide_particles(coords, charges, num_workers):
    num_particles = coords.shape[0]
    coords_per_worker = math.floor(num_particles/num_workers)
    leftover = num_particles % num_workers
    split_coords = []
    split_charges = []
    allocated = 0
    for _ in range(num_workers):
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


def _distribute_particles(
    particle_coord_list, charges_list, grid_resolution, max_num_workers, comm
):
    for i in range(1, max_num_workers):
        comm.send(particle_coord_list[i], dest=i, tag=MSG.COORDS.value)
        comm.send(charges_list[i], dest=i, tag=MSG.CHARGES.value)
        comm.send(grid_resolution, dest=i, tag=MSG.GRID_RES.value)

    sub_particle_coords = particle_coord_list[0]
    sub_charges = charges_list[0]
    return sub_particle_coords, sub_charges
