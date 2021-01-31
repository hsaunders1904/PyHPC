import math
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from utils import gen_particles, plot_potential_grid

c_dtype = "float"
dtype = "float32"

cuda_func_name = "potential_cuda"
cuda_module = SourceModule(f"""
    __global__ void {cuda_func_name}(
        const {c_dtype} *x_pos,
        const {c_dtype} *y_pos,
        const {c_dtype} *x_grid,
        const {c_dtype} *y_grid,
        const int *charges,
        const int grid_resolution,
        const int num_particles,
        {c_dtype} *potential_grid
    ) {{
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;

        if (i >= grid_resolution || j >= grid_resolution) {{
            return;
        }}

        int index;
        {c_dtype} x_step, y_step, dist;
        for (int k = 0; k < num_particles; ++k) {{
            x_step = x_grid[i] - x_pos[k];
            y_step = y_grid[j] - y_pos[k];
            dist = sqrt(x_step*x_step + y_step*y_step);

            index = i + grid_resolution*j;
            potential_grid[index] -= charges[k]*log(dist);
        }}
    }}
""")

cuda_func = cuda_module.get_function(cuda_func_name)

grid_resolution = np.int32(3000)
num_particles = np.int32(1000)
particle_coords = gen_particles(num_particles, "random", dtype=dtype)
charges = np.random.choice(np.int32([-1, 1]), (num_particles, ))

# All numpy arrays must be C contiguous to pass to Cuda
x_pos = particle_coords[:, 0].copy(order="C")
y_pos = particle_coords[:, 1].copy(order="C")
x_grid = np.linspace(0, 1, grid_resolution, dtype=dtype)
y_grid = np.linspace(0, 1, grid_resolution, dtype=dtype)
potential_grid = np.zeros((1, grid_resolution**2), dtype=dtype)

args = (drv.In(x_pos), drv.In(y_pos), drv.In(x_grid), drv.In(y_grid),
        drv.In(charges), grid_resolution, num_particles,
        drv.Out(potential_grid))

# Threads in a grid execute the same kernel function.
# A grid comprises blocks and each block comprises threads.
# Each thread in a block shares the same blockIdx and has a unique
# threadIdx
max_threads_per_block = cuda_func.max_threads_per_block
print(f"max_threads_per_block: {max_threads_per_block}")

# The grid we are calculating has grid_resolution squared "pixels", so
# we need (at least) that many threads. Keeping the dimensions as
# multiples of 4 benefits more than the cost of the extra threads
required_threads = grid_resolution**2

# Number of threads along each axis should divide by 4
num_x_threads = np.ceil(grid_resolution / 4) * 4
num_y_threads = np.ceil(grid_resolution / 4) * 4
print(f"num_x_threads: {num_x_threads}")

threads_per_block = 256  # this is the number to tweak for performance

# Get the number of threads in each direction in each block, since we're
# calculating on a square, each direction is the square root of the
# total number of threads in the block
block_axis_size = np.sqrt(threads_per_block)
print(f"block_axis_size: {block_axis_size}")

# The number of threads to run in each direction in each block to cover
# the number of "pixels" on each axis
grid_threads = (math.ceil(grid_resolution / block_axis_size),
                math.ceil(grid_resolution / block_axis_size), 1)
print(f"grid_threads: {grid_threads}")

block_size = (math.ceil(num_x_threads / grid_threads[0]),
              math.ceil(num_y_threads / grid_threads[1]), 1)
print(f"block_size: {block_size}")

start = time.time()
cuda_func(*args, block=block_size, grid=grid_threads)
print(f"Cuda: {time.time() - start} seconds")

potential_grid = np.reshape(potential_grid, (grid_resolution, grid_resolution))

plot_potential_grid(potential_grid)