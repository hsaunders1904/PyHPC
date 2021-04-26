from argparse import ArgumentParser

from pyhpc.potential.mpi._mpi import run_and_plot

parser = ArgumentParser("Run calculate_grid using MPI.")
parser.add_argument("num_particles", type=int)
parser.add_argument("grid_resolution", type=int)
parser.add_argument(
    "-d", "--dist", choices={"circle", "random"}, default="random"
)
parser.add_argument("-f", "--func", default="numpy")
args = parser.parse_args()

run_and_plot(**vars(args))
