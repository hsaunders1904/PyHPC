c_dtype = "double"
dtype = "float64"
FUNC_NAME = "potential_cuda"

KERNEL = f"""
    __global__ void {FUNC_NAME}(
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
"""
