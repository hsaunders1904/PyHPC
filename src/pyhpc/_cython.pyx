import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport log, sqrt


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing
cpdef np.ndarray[double, ndim=2] potential_cython(
    double[:, :] particle_coords,
    int grid_resolution,
    int[:] charges
):
    # Allocate the output array
    cdef np.ndarray[double, ndim=2] \
        potential_grid = np.zeros((grid_resolution, grid_resolution))

    cdef double delta_x, delta_y, delta_denom
    cdef Py_ssize_t i, j, n

    delta_denom = grid_resolution - 1
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for n in range(charges.shape[0]):
                delta_x = i/delta_denom - particle_coords[n, 0]
                delta_y = j/delta_denom - particle_coords[n, 1]
                potential_grid[j, i] -= \
                    charges[n]*log(sqrt(delta_x*delta_x + delta_y*delta_y))

    return potential_grid
