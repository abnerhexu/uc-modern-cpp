#include <pybind11/pybind11.h>
#include <vector>
#include <pybind11/stl.h>

namespace nn {

class Node {
public:
    Node() {}
}; // class Node

class DataNode: public Node {
public:
    std::vector<float> data;
    std::vector<Node*> parents;
public:
    DataNode(std::vector<float> data): data(data) {}
    std::vector<float> forward(std::vector<float> x);
    std::vector<float> backward(std::vector<float> gradient, std::vector<float> x);
}; // class DataNode

class Parameter: public DataNode {
public:
    const std::tuple<int> shape;
public:
    Parameter(const std::tuple<int> shape): shape(shape) {
        if (std::tuple_size<decltype(this->shape)>::value != 2) {
            throw std::runtime_error("Parameter shape must be a tuple of 2 integers");
        }

    }
}; // class Parameter

}