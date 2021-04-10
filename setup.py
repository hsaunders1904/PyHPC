from Cython.Build import cythonize
from setuptools import find_packages, setup, extension

import numpy as np

extensions = extension.Extension(
    "potential._cython",
    ["src/potential/_cython.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    name="potential",
    version="0.0.1",
    url="https://github.com/hsaunders1904/PyHPC",
    packages=find_packages(exclude=["*tests*"]),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions)
)
