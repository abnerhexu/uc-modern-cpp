#include <pybind11/pybind11.h>
#include "math/arith.h"
namespace py = pybind11;

PYBIND11_MODULE(uctc, m) {
    py::module C = m.def_submodule("C", "C module");
    py::module arith = C.def_submodule("arith", "Arithmetic module");
    arith.def("sqrt", &arith::sqrt, "Square root function", py::arg("x") = 0.0);
}

