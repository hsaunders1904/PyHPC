import os
import platform
from setuptools import find_packages, setup, extension

import numpy as np
import pybind11
from Cython.Build import cythonize

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
POTENTIAL_SRC_DIR = os.path.join(ROOT_DIR, "src", "pyhpc", "potential")


def read_requirements_file():
    with open(os.path.join(ROOT_DIR, "requirements.txt"), "r") as f:
        return [req.strip() for req in f.readlines()]


if platform.system() == "Windows":
    extra_compile_args = ["/openmp", "/Ox"]
elif platform.system() == "Linux":
    extra_compile_args = ["-openmp", "-O3"]

cython_ext = cythonize(
    extension.Extension(
        "pyhpc.potential._calculate_grid_impls._cython", [
            os.path.join(
                POTENTIAL_SRC_DIR, "_calculate_grid_impls", "_cython.pyx"
            )
        ],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args
    )
)[0]

cpp_ext = extension.Extension(
    "pyhpc.potential._calculate_grid_impls._cpp_lib",
    sources=[
        os.path.join(
            POTENTIAL_SRC_DIR, "_calculate_grid_impls", "_cpp_lib.cpp"
        )
    ],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args
)

setup(
    name="pyhpc",
    version="0.0.1",
    url="https://github.com/hsaunders1904/PyHPC",
    packages=find_packages(where=os.path.join(ROOT_DIR, "src")),
    package_dir={"": "src"},
    ext_modules=[cython_ext, cpp_ext],
    setup_requires=["cython", "numpy", "pybind11"],
    extras_require={"test": ["pytest", "pytest-benchmark"]},
    install_requires=read_requirements_file()
)
