FUNC_NAME = "potential_cuda"

KERNEL = """
__global__ void potential_cuda(const double *x_pos, const double *y_pos,
                               const int *charges, const int grid_resolution,
                               const int num_particles,
                               double *potential_grid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= grid_resolution || j >= grid_resolution) {
    return;
  }

  int thread_id = i + grid_resolution * j;
  double x_step, y_step, dist;
  double grid_x_pos = i / (grid_resolution - 1.0);
  double grid_y_pos = j / (grid_resolution - 1.0);
  for (int n = 0; n < num_particles; ++n) {
    x_step = grid_x_pos - x_pos[n];
    y_step = grid_y_pos - y_pos[n];
    dist = sqrt(x_step * x_step + y_step * y_step);
    potential_grid[thread_id] -= charges[n] * log(dist);
  }
}
"""
