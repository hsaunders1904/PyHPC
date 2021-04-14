KERNEL = """
__kernel void potential_cl(
    __global const double *x_pos_buf,
    __global const double *y_pos_buf,
    __global const int *charges_buf,
    const int grid_resolution,
    const int num_particles,
    __global double *potential_grid_buf
) {{

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= grid_resolution || j >= grid_resolution) {
        // Executor not required
        return;
    }

    int index;
    double grid_step_denom = grid_resolution - 1.0;
    double x_step, y_step, dist;
    for (int n = 0; n < num_particles; ++n) {{
        x_step = i/grid_step_denom - x_pos_buf[n];
        y_step = j/grid_step_denom - y_pos_buf[n];
        dist = sqrt(x_step*x_step + y_step*y_step);

        potential_grid_buf[i + grid_resolution*j] -= charges_buf[n]*log(dist);
    }}
}}
"""
