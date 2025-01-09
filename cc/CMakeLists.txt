cmake_minimum_required(VERSION 3.19)
project(hwbuilder)
set(CMAKE_CXX_STANDARD 17)
set(CXX g++)
set(CMAKE_CXX_FLAGS -Og)
set(PYBIND11_DIR /home/hexu/miniconda3/envs/hwisa/lib/python3.11/site-packages/pybind11)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_DIR})
pybind11_add_module(hwisac inst/instBuilder.cc inst/cellBuilder.cc inst/mixedPrecis.cc pb11/bindings.cc)

add_custom_command(
    TARGET hwisac POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Changing directory and running Python script for generate interpreter annotations"
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} pybind11-stubgen hwisac --output-dir .
)