cmake_minimum_required(VERSION 3.10)

project(PY_HPC LANGUAGES CXX)

if(MSVC)
  add_compile_options("/W3" "/O2")
elseif(GCC)
  add_compile_options("-Wall" "-O3")
endif()

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR})

set(PY_LIB_NAME "pyhpc_cpp")
pybind11_add_module(${PY_LIB_NAME} potential_cpp.cpp)
target_link_libraries(${PY_LIB_NAME} PRIVATE OpenMP::OpenMP_CXX)
