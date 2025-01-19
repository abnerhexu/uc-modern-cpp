#include <vector>
#include <memory>
#include "../tensor/tensor.h"

namespace nn {

class Node {
public:
    std::shared_ptr<tensor::Tensor> data;
public:
    Node() {}
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;
    virtual void zero_grad() = 0;
    virtual ~Node() {}
};

class DataNode: public Node {
public:
    DataNode() {}
}; // class DataNode

class Parameter: public DataNode {
public:
    Parameter(const std::vector<std::size_t>& shape) {
        this->data = std::make_shared<tensor::Tensor>(shape, true);
    }
    void update(tensor::Tensor *grad, float lr);
}; // class Parameter

class Constant: public DataNode {
public:
    Constant(std::shared_ptr<tensor::Tensor> data) {
        this->data = data;
    }
}; // class Constant

class FunctionNode: public Node {
public:
    std::vector<std::shared_ptr<Node>> objects;
public:
    template <typename... Args>
    FunctionNode(Args ... args) {
        this->objects = {std::dynamic_pointer_cast<A>(
            std::shared_ptr<std::remove_pointer_t<std::decay_t<Args>>>(args...)
        )
        ...};
        this->data = this->forward(this->objects);
    }

    virtual std::shared_ptr<tensor::Tensor> forward(std::vector<std::shared_ptr<Node>> objects) = 0;
}; //class FunctionNode


}