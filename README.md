# HPC in Python

This repository contains examples of high performance computing techniques in
Python.

In the Jupyter notebook file [HPC.ipynb](HPC.ipynb) an example calculating the
potential field created by charged particles is used.
The example is implemented, and benchmarked, using several tools:

- Pure Python
- Numpy
- Numba
- Cython
- C++ with pybind11
  - with multithreading using OpenMP
- PyOpenCL
  - with use of GPU

## Windows Setup Instructions

I set this up using a Conda environment,
but install the requirements using pip may also work.

1) Create a Conda environment

   ```shell
   conda create -n hpc --file requirements.txt -c conda-forge
   ```

2) To build the C++ library you'll need a C++ compiler.
   On Windows you can use Visual Studio (I've only tested with 2019,
   but 2015 or above should work) or MinGW.

   After installing a compiler, compile the Python library using CMake.

   ```shell
   mkdir build
   cmake -S . -B build
   cmake --build build --config Release
   ```

   If everything is successful, the file `pyhpc_cpp.cp37-win_amd64.pyd` should be
   created in the root of the repository.

3) To install pyopencl download the relevant `.whl` for your system from a
   [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl).
   I downloaded `pyopencl‑2020.3.1+cl21‑cp37‑cp37m‑win_amd64.whl` as I'm
   working with Python 3.7 on a 64-bit system.

   Pip install this `.whl` in your Conda environment.

   ```shell
   pip install ./pyopencl‑2020.3.1+cl21‑cp37‑cp37m‑win_amd64.whl
   ```
