#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyarr {

std::vector<float> ndarray_to_vector(py::array_t<float> array);

}