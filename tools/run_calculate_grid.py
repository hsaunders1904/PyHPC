from time import time

import argparse
import numpy as np

from pyhpc.plotting import plot_potential_grid
from pyhpc.potential import calculate_grid
from pyhpc.utils import gen_particles, gen_charges


def time_func(func_to_time, *args, **kwargs):
    start = time()
    out = func_to_time(*args, **kwargs)
    end = time()
    return out, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run calculate_grid and time it.")
    parser.add_argument("num_particles", type=int)
    parser.add_argument("grid_resolution", type=int)
    parser.add_argument(
        "-f",
        "--func",
        default="numpy",
        help="the method to run calculate grid with",
        nargs="+"
    )
    parser.add_argument(
        "-d",
        "--dist",
        choices={"circle", "random"},
        default="random",
        help="the distribution of particle positions"
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        help=(
            "keyword args to pass to calculate_grid, should be in the form "
            "'keyword1=value1 [keyword2=value2 ...]'"
        ),
        default={}
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot the output grid."
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        help="seed for the random number generator",
        default=None
    )
    cli_args = parser.parse_args()

    particle_coords = gen_particles(cli_args.num_particles, cli_args.dist)
    charges = gen_charges(cli_args.num_particles)
    args = (particle_coords, cli_args.grid_resolution, charges)

    kwargs = {k: v for k, v in [x.split("=") for x in cli_args.kwargs]}

    grids = []
    for func_name in cli_args.func:
        kwargs["func"] = func_name
        grid, elapsed_time = time_func(calculate_grid, *args, **kwargs)
        grids.append(grid)
        print(f"Function call took {elapsed_time} seconds using {func_name}")

    if cli_args.plot:
        plot_potential_grid(*grids)
