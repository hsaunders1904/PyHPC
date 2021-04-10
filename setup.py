import os
import subprocess
import sys
from Cython.Build import cythonize
from setuptools import find_packages, setup, extension
from setuptools.command.build_ext import build_ext as build_ext_orig

import numpy as np
import pybind11

cython_ext = extension.Extension(
    "potential._cython",
    [os.path.join(os.path.dirname(__file__), "src/potential/_cython.pyx")],
    include_dirs=[np.get_include()],
)


class ExtensionBuilder(build_ext_orig):
    """
    Extend the :class:`build_ext_orig` to build our C++ library with
    CMake.
    """

    def run(self):
        super().run()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import cmake  # import here so install_requires runs first

        cmake_path = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
        ext_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        configure_args = [
            "-S",
            os.path.dirname(__file__), "-B", self.build_temp,
            f"-DPython_EXECUTABLE=\"{sys.executable}\"",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_ROOT={pybind11.get_cmake_dir()}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}"
        ]
        build_args = ["--build", self.build_temp, "--config", "Release"]

        print(" ".join([cmake_path, *configure_args]))
        subprocess.check_call([cmake_path, *configure_args])
        print(" ".join([cmake_path, *build_args]))
        subprocess.check_call([cmake_path, *build_args])


setup(
    name="potential",
    version="0.0.1",
    url="https://github.com/hsaunders1904/PyHPC",
    packages=find_packages(exclude=["*tests*"]),
    package_dir={"": "src"},
    ext_modules=cythonize(cython_ext),
    cmdclass={"build_ext": ExtensionBuilder},
    setup_requires=["cmake", "cython", "numpy", "pybind11"]
)
