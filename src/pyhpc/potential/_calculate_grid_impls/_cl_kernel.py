KERNEL = """
__kernel void potential_cl(__global const double *x_coords,
                           __global const double *y_coords,
                           __global const int *charges,
                           const int grid_resolution, const int num_particles,
                           __global double *potential_grid) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= grid_resolution || j >= grid_resolution) {
    // Executor not required
    return;
  }

  int thread_id = i + grid_resolution * j;
  double x_step, y_step, euclid_dist;
  double grid_x_pos = i / (grid_resolution - 1.0);
  double grid_y_pos = j / (grid_resolution - 1.0);
  for (int n = 0; n < num_particles; ++n) {
    x_step = grid_x_pos - x_coords[n];
    y_step = grid_y_pos - y_coords[n];
    euclid_dist = sqrt(x_step * x_step + y_step * y_step);
    potential_grid[thread_id] -= charges[n] * log(euclid_dist);
  }
}

"""
