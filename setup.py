import os
import platform

from Cython.Build import cythonize
from setuptools import find_packages, setup, extension

import numpy as np
import pybind11

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def read_requirements_file():
    if platform.system() == "Windows":
        req_file_name = "requirements_win10.txt"
    else:
        req_file_name = "requirements_linux.txt"
    with open(os.path.join(ROOT_DIR, req_file_name), "r") as f:
        return [req.strip() for req in f.readlines()]


if platform.system() == "Windows":
    extra_compile_args = ["/openmp", "/Ox"]
elif platform.system() == "Linux":
    extra_compile_args = ["-openmp", "-O3"]

cython_ext = cythonize(
    extension.Extension(
        "pyhpc._cython",
        [os.path.join(ROOT_DIR, "src", "pyhpc", "_cython.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args
    )
)[0]

cpp_ext = extension.Extension(
    "pyhpc._cpp_lib",
    sources=[os.path.join(ROOT_DIR, "potential_cpp.cpp")],
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
