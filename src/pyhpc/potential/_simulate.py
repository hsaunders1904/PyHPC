import numpy as np

from pyhpc.potential import calculate_grid


def _update_velocity(velocity, particle_coords, delta_t, charges):
    velocity += delta_t*_get_force_on_particles(
        particle_coords[:, 0], particle_coords[:, 1], charges
    )
    out_of_bounds = np.logical_or(particle_coords >= 1, particle_coords <= 0)
    return _reflect_velocity(velocity, out_of_bounds)


def _reflect_velocity(velocity, reflect):
    reflector = np.ones(velocity.shape)
    reflector[reflect] = -1
    return velocity*reflector


def _get_force_on_particles(pos_x, pos_y, k):
    n_particles = len(k)
    k_mult_outer = np.multiply.outer(k, k)
    sub_outer_x = np.multiply(k_mult_outer, np.subtract.outer(pos_x, pos_x))
    sub_outer_y = np.multiply(k_mult_outer, np.subtract.outer(pos_y, pos_y))
    sub_outer = np.vstack((
        np.reshape(sub_outer_x, (1, n_particles**2)),
        np.reshape(sub_outer_y, (1, n_particles**2))
    ))
    norms = np.linalg.norm(sub_outer, axis=0)
    norms = np.reshape(norms, (n_particles, n_particles)) + np.eye(
        n_particles
    )  # Add I to avoid division by zero
    force_x = np.sum(np.divide(sub_outer_x, norms), axis=0)
    force_y = np.sum(np.divide(sub_outer_y, norms), axis=0)
    return -np.vstack((force_x, force_y)).T


def simulate_particles(
    particle_coords,
    grid_resolution,
    charges,
    velocities,
    num_frames,
    time_step,
    func="numba",
    logging=True,
    **kwargs
):
    frames = np.zeros((grid_resolution, grid_resolution, num_frames))

    if logging:
        print(f"Generating frame 01")
    # Calculate initial potential
    frames[:, :, 0] = calculate_grid(
        particle_coords, grid_resolution, charges, func=func, **kwargs
    )

    if logging:
        print(f"Generating frame 02")
    # Initial Forward Euler half-step
    velocities = 0.5*_update_velocity(
        velocities, particle_coords, time_step, charges
    )
    particle_coords += time_step*velocities
    frames[:, :, 1] = calculate_grid(
        particle_coords, grid_resolution, charges, func=func, **kwargs
    )

    for frame_num in range(2, num_frames):
        if logging:
            print(f"Generating frame {frame_num + 1:02}")
        velocities = _update_velocity(
            velocities, particle_coords, time_step, charges
        )
        particle_coords += time_step*velocities
        frames[:, :, frame_num] = calculate_grid(
            particle_coords, grid_resolution, charges, func=func, **kwargs
        )
    return frames
