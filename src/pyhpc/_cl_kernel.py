KERNEL = """
__kernel void potential_cl(
    __global const double *x_pos_buf,
    __global const double *y_pos_buf,
    __global const double *x_grid_buf,
    __global const double *y_grid_buf,
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
    double x_step, y_step, dist;
    for (int k = 0; k < num_particles; ++k) {{
        x_step = x_grid_buf[i] - x_pos_buf[k];
        y_step = y_grid_buf[j] - y_pos_buf[k];
        dist = sqrt(x_step*x_step + y_step*y_step);

        potential_grid_buf[i + grid_resolution*j] -= charges_buf[k]*log(dist);
    }}
}}
"""
