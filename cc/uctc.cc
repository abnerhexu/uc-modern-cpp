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

    py::class_<tensor::Tensor, std::shared_ptr<tensor::Tensor>>(m, "Tensor")
    .def_readonly("shape", &tensor::Tensor::shape)
    .def_readonly("size", &tensor::Tensor::size)
    .def("data", &tensor::Tensor::get_data, "Get the data of the tensor", pybind11::return_value_policy::copy);
    
    py::module nn = m.def_submodule("nn", "Neural network module");
    py::class_<nn::Node, std::shared_ptr<nn::Node>>(nn, "Node")
    .def("data", &nn::Node::get_data, "Get the data of the node", pybind11::return_value_policy::copy)
    .def("tensor", &nn::Node::get_tensor, "Get the tensor of the node", pybind11::return_value_policy::automatic_reference);

    py::class_<nn::DataNode, nn::Node, std::shared_ptr<nn::DataNode>>(nn, "DataNode");

    py::class_<nn::Parameter, nn::DataNode, std::shared_ptr<nn::Parameter>>(nn, "Parameter")
    .def(pybind11::init<py::array_t<float>>(), "Create a parameter from an array.")
    .def("update", &nn::Parameter::update, "Update the parameter node", py::arg("grad") = nullptr, py::arg("learning_rate") = 0.001);

    py::class_<nn::Constant, nn::DataNode, std::shared_ptr<nn::Constant>>(nn, "Constant")
    .def(pybind11::init<py::array_t<float>>(), "Create a constant node from a numpy array");

    py::class_<nn::FunctionNode, nn::Node, std::shared_ptr<nn::FunctionNode>>(nn, "FunctionNode");

    py::class_<nn::Add, nn::FunctionNode, std::shared_ptr<nn::Add>>(nn, "Add")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create an add function node")
    .def("forward", &nn::Add::forward, "Forward function");

    py::class_<nn::AddBias, nn::FunctionNode, std::shared_ptr<nn::AddBias>>(nn, "AddBias")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create an add bias function node")
    .def("forward", &nn::AddBias::forward, "Forward function");

    py::class_<nn::Linear, nn::FunctionNode, std::shared_ptr<nn::Linear>>(nn, "Linear")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create a linear function node")
    .def("forward", &nn::Linear::forward, "Forward function");

    py::class_<nn::ReLU, nn::FunctionNode, std::shared_ptr<nn::ReLU>>(nn, "ReLU")
    .def(py::init<std::shared_ptr<nn::Node>>(), "Create a ReLU function node");

    py::class_<nn::Loss, nn::FunctionNode, std::shared_ptr<nn::Loss>>(nn, "Loss");

    py::class_<nn::SquareLoss, nn::Loss, std::shared_ptr<nn::SquareLoss>>(nn, "SquareLoss")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create a square loss function node");
    py::class_<nn::SoftmaxLoss, nn::Loss, std::shared_ptr<nn::SoftmaxLoss>>(nn, "SoftmaxLoss")
    .def(py::init<std::shared_ptr<nn::Node>, std::shared_ptr<nn::Node>>(), "Create a softmax loss function node");
    
    nn.def("log_softmax", &nn::log_softmax, "Log softmax function", py::arg("logits"));

    nn.def("gradients", &nn::gradients, "Calculate the gradients", py::arg("loss") = nullptr, py::arg("nodes") = std::vector<std::shared_ptr<nn::Node>>{});
    nn.def("pyarray_to_tensor", &tensor::pyarray_to_tensor, "Convert a numpy array to a tensor", py::arg("arr"));
    nn.def("argmax", &tensor::argmax, "Get a tensor's argmax", py::arg("tensor"), py::arg("axis"));
    nn.def("mean", &tensor::mean, "Get a tensor element's mean value", py::arg("tensor"));
    nn.def("exp", &tensor::exp, "Get exp of a tensor", py::arg("tensor"));
}

