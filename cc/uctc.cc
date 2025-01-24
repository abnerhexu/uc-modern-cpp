#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "math/arith.h"
#include "operators/nn.h"
#include "tensor/tensor.h"
namespace py = pybind11;

PYBIND11_MODULE(uctc, m) {

    py::module C = m.def_submodule("C", "C module");
    py::module arith = C.def_submodule("arith", "Arithmetic module");
    arith.def("sqrt", &arith::sqrt, "Square root function", py::arg("x") = 0.0);

    py::module nn = m.def_submodule("nn", "Neural network module");
    py::class_<nn::Node, std::shared_ptr<nn::Node>>(nn, "Node");

    py::class_<tensor::Tensor, std::shared_ptr<tensor::Tensor>>(m, "Tensor")
    .def_readonly("shape", &tensor::Tensor::shape)
    .def_readonly("size", &tensor::Tensor::size);

    py::class_<nn::DataNode, nn::Node, std::shared_ptr<nn::DataNode>>(nn, "DataNode");

    py::class_<nn::Parameter, nn::DataNode, std::shared_ptr<nn::Parameter>>(nn, "Parameter")
    .def(pybind11::init<const std::vector<std::size_t>&>(), "Create a parameter node with a given shape");

    py::class_<nn::FunctionNode, nn::Node, std::shared_ptr<nn::FunctionNode>>(nn, "FunctionNode");

    py::class_<nn::Linear, nn::FunctionNode, std::shared_ptr<nn::Linear>>(nn, "Linear")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create a linear function node")
    .def("forward", &nn::Linear::forward, "Forward function");

    py::class_<nn::ReLU, nn::FunctionNode, std::shared_ptr<nn::ReLU>>(nn, "ReLU")
    .def(py::init<std::shared_ptr<nn::Node>>(), "Create a ReLU function node");

    py::class_<nn::Loss, nn::FunctionNode, std::shared_ptr<nn::Loss>>(nn, "Loss");

    py::class_<nn::SquareLoss, nn::Loss, std::shared_ptr<nn::SquareLoss>>(nn, "SquareLoss")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create a square loss function node");
}

