import os
import platform
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from pyhpc.plotting import animate_frames
from pyhpc.potential import simulate_particles
from pyhpc.utils import gen_particles, gen_charges

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _find_ffmpeg():
    if platform.system() == "Windows":
        path_strs = subprocess.check_output("where.exe ffmpeg.exe")
        return path_strs.decode().split("\n")[0].strip()
    else:
        path_str = subprocess.check_output("which ffmpeg")
        return path_str.decode().strip()


def _get_unique_path(path):
    iters = 0
    p, ext = os.path.splitext(path)
    while os.path.exists(path):
        path = f"{p}{iters:02}{ext}"
        iters += 1
    return path


if __name__ == "__main__":
    # pyplot often complains about not finding ffmpeg, even if it's on
    # the system path. Pointing to it explicitly seems to work
    plt.rcParams['animation.ffmpeg_path'] = _find_ffmpeg()

    n_frames = 200
    grid_resolution = 1080
    n_particles = 2
    particle_coords = gen_particles(n_particles, "circle")
    charges = gen_charges(n_particles)
    delta_t = 0.03

    initial_velocities = np.zeros((n_particles, 2))
    frames = simulate_particles(
        particle_coords,
        grid_resolution,
        charges,
        initial_velocities,
        n_frames,
        delta_t,
        func="numba"
    )

    out_dir = os.path.join(ROOT_DIR, ".animations")
    base_anim_name = "particle_anim.mp4"

    path_out = _get_unique_path(os.path.join(out_dir, base_anim_name))
    print("Animating frames...")
    anim = animate_frames(frames)
    anim.save(path_out)
    print(f"Animation saved to {path_out}.")
