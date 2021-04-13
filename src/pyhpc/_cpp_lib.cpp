#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <memory>
#include <omp.h>
#include <vector>

namespace py = pybind11;

std::unique_ptr<double[]> calc_potential_grid(const double *x_coords,
                                              const double *y_coords,
                                              const std::size_t grid_resolution,
                                              const int *charges,
                                              const std::size_t num_particles,
                                              const int num_threads = 1) {
  std::unique_ptr<double[]> potential_grid(
      new double[grid_resolution * grid_resolution]);

  const double grid_step_denom = static_cast<double>(grid_resolution) - 1.0;

#pragma omp parallel firstprivate(grid_step_denom), private(grid_pos),         \
    num_threads(num_threads)
  {
    for (std::size_t n = 0; n < num_particles; ++n) {
      for (std::size_t i = 0; i < grid_resolution; ++i) {
#pragma omp for schedule(static)
        for (long j = 0; j < grid_resolution; ++j) {
          double delta_x = i / grid_step_denom - x_coords[n];
          double delta_y = j / grid_step_denom - y_coords[n];
          double euclid_distance = sqrt(delta_x * delta_x + delta_y * delta_y);
          std::size_t idx = i + grid_resolution * j;
          potential_grid[idx] -= charges[n] * log(euclid_distance);
        }
      }
    }
  }
  return potential_grid;
}

// ----------------------------------------------------------------------------
// Python Interface
// ----------------------------------------------------------------------------
py::array_t<double>
py_calc_potential_grid(const py::array_t<double, py::array::c_style> &x_coords,
                       const py::array_t<double, py::array::c_style> &y_coords,
                       const std::size_t grid_resolution,
                       const py::array_t<int, py::array::c_style> &charges,
                       const int num_threads = 1) {

  auto x_coords_buf = x_coords.request();
  auto y_coords_buf = y_coords.request();
  auto charges_buf = charges.request();
  if (x_coords_buf.ndim != 1 || y_coords_buf.ndim != 1 ||
      charges_buf.ndim != 1) {
    throw std::runtime_error(
        "x_coords, y_coords and charges must be 1D numpy arrays.");
  }
  if (x_coords_buf.size != y_coords_buf.size ||
      x_coords_buf.size != charges_buf.size) {
    throw std::runtime_error(
        "x_coords, y_coords and charges must have equal size.");
  }

  // Get pointers to numpy arrays
  const auto *x_coords_ptr = static_cast<double *>(x_coords_buf.ptr);
  const auto *y_coords_ptr = static_cast<double *>(y_coords_buf.ptr);
  const auto *charges_ptr = static_cast<int *>(charges_buf.ptr);

  const py::ssize_t num_particles{x_coords_buf.size};

  auto potential_grid =
      calc_potential_grid(x_coords_ptr, y_coords_ptr, grid_resolution,
                          charges_ptr, num_particles, num_threads);

  // Create capsule to delete memory when capsule is deleted - the capsule is
  // passed to Python so the memory is cleared when the output numpy array is
  // deleted
  py::capsule free_when_done(potential_grid.get(), [](void *f) {
    double *potential_grid = reinterpret_cast<double *>(f);
    delete[] potential_grid;
  });

  return py::array_t<double>(
      {static_cast<long>(grid_resolution * grid_resolution)},
      potential_grid.release(), free_when_done);
}

PYBIND11_MODULE(_cpp_lib, m) {
  m.def("calc_potential_grid", &py_calc_potential_grid,
        "Calculate potential of grid points caused by given particles.");
}
