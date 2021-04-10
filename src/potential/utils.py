import matplotlib.pyplot as plt
import numpy as np


def gen_particles(N, dist="circle", dtype="float64"):
    if dist.lower() == "circle":
        coords = np.zeros((N, 2))
        r = np.arange(0, N)
        coords[:, 0] = 0.5 + 0.4*np.sin(2*np.pi*r/N + 0.1)
        coords[:, 1] = 0.5 + 0.4*np.cos(2*np.pi*r/N + 0.1)
        return coords.astype(dtype)
    elif dist.lower() == "random":
        return np.random.rand(N, 2).astype(dtype)
    else:
        raise ValueError(f"Unrecognised distribution {dist}.")


def gen_charges(N):
    return np.random.choice([-1, 1], (N, )).astype("int32")
