#include <pybind11/pybind11.h>
#include <vector>
#include <pybind11/stl.h>
#include <numeric>
#include "function/data.h"

namespace nn {

class Node {
public:
    Node() {}
}; // class Node

template <typename T>
class DataNode: public Node {
public:
    std::vector<T> data;
    std::vector<Node*> parents;
public:
    DataNode(std::vector<T> data): data(data) {}
    std::vector<T> forward(std::vector<float> x);
    std::vector<T> backward(std::vector<float> gradient, std::vector<float> x);
}; // class DataNode

template<typename T>
class Parameter: public DataNode {
public:
    std::vector<size_t> shape;
    size_t size = 0;
public:
    Parameter(const std::vector<size_t> shape): shape(shape), DataNode({}){
        if (this->shape.size() != 2) {
            throw std::runtime_error("Parameter shape must be a tuple of 2 integers");
        }
        bool allPositive = std::all_of(shape.begin(), shape.end(), [](size_t value) {
            return value > 0;
        });
        if (!allPositive) {
            throw std::runtime_error("Parameter shape must be a tuple of positive integers");
        }
        this->size = std::accumulate(this->shape.begin(), this->shape.end(), 0);
        this->data = generate_random_array<float>(this->size, -1.0, 1.0);
    }
    void update(Parameter& direction, float learning_rate);
}; // class Parameter

template <typename T>
class Constant: public DataNode {
    /*
    A Constant node is used to represent:
    * Input features
    * Output labels
    * Gradients computed by back-propagation
    You should not need to construct any Constant nodes directly; they will
    instead be provided by either the dataset or when you call `nn.gradients`.
    */
public:
    Constant(std::vector<T> data): DataNode(data) {}
}; // class Constant

template<typename T>
class FunctionNode: public Node {
public:
    std::vector<Node*> parents;
public:
    FunctionNode(){}
    template <typename... Args>
    FunctionNode(Args... args) {
        // 将参数包展开并存入 vector
        (this->parents.push_back(args), ...);
        this->forward();
    }
    void forward();
};

}