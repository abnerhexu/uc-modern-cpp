cmake_minimum_required(VERSION 3.19)
project(uctc)
set(CMAKE_CXX_STANDARD 17)
set(CXX g++)
set(CMAKE_CXX_FLAGS -O3)
set(PYBIND11_DIR /home/hexu/miniconda3/lib/python3.11/site-packages/pybind11)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_DIR})
pybind11_add_module(uctc uctc.cc math/arith.cc operators/nn.cc operators/ops.cc tensor/tensor.cc)

add_custom_command(
    TARGET uctc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Changing directory and running Python script for generate interpreter annotations"
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} pybind11-stubgen uctc --output-dir .
)