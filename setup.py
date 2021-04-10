import os
from Cython.Build import cythonize
from setuptools import find_packages, setup, extension

import numpy as np
import pybind11

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

cython_ext = extension.Extension(
    "potential._cython",
    [os.path.join(ROOT_DIR, "src", "potential", "_cython.pyx")],
    include_dirs=[np.get_include()],
)

cpp_ext = extension.Extension(
    "potential._cpp_lib",
    sources=[os.path.join(ROOT_DIR, "potential_cpp.cpp")],
    include_dirs=[pybind11.get_include()],
    language="c++"
)

setup(
    name="potential",
    version="0.0.1",
    url="https://github.com/hsaunders1904/PyHPC",
    packages=find_packages(exclude=["*tests*"]),
    package_dir={"": "src"},
    ext_modules=cythonize(cython_ext) + [cpp_ext],
    setup_requires=["cython", "numpy", "pybind11"]
)
