#include <vector>
#include <memory>
#include "../tensor/tensor.h"
#include "../math/arith.h"

namespace nn {

class Node {
public:
    std::shared_ptr<tensor::Tensor> data;
public:
    Node() {}
    virtual std::shared_ptr<tensor::Tensor> forward() = 0;
    virtual std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> backward(std::shared_ptr<Node> gradient) = 0;
    virtual void update(std::shared_ptr<tensor::Tensor> grad, float lr) = 0;
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
    void update(std::shared_ptr<tensor::Tensor> grad, float lr) override {
        for (auto i = 0; i < this->data->size; i++) {
            this->data->data[i] -= lr * grad->data[i];
        }
    }
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
    std::vector<std::shared_ptr<Node>> gradient;
public:
    template <typename... Args>
    FunctionNode(Args ... args) {
        this->objects = {std::dynamic_pointer_cast<A>(
            std::shared_ptr<std::remove_pointer_t<std::decay_t<Args>>>(args...)
        )
        ...};
        this->data = this->forward();
    }

}; //class FunctionNode

class Add: public FunctionNode {
public:
    Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) : FunctionNode(a, b) {}
    std::shared_ptr<tensor::Tensor> forward() override {
        auto a = this->objects[0];
        auto b = this->objects[1];
        auto data = std::make_shared<tensor::Tensor>(a->data->shape);
        for (auto i = 0; i < a->data->size; i++) {
            data->data[i] = a->data->data[i] + b->data->data[i];
        }
        return data;
    }
    std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> backward(std::shared_ptr<Node> gradient) override {
        // assertion needed
        return {gradient, gradient};
    }
};

class Linear: public FunctionNode {
public:
    Linear(std::shared_ptr<Node> a, std::shared_ptr<Node> b) : FunctionNode(a, b) {}
    std::shared_ptr<tensor::Tensor> forward() override {
        // features: (batch_size x input_features)
        auto features = this->objects[0];
        // weights: (input_features x output_features)
        auto weights = this->objects[1];
        auto m = features->data->shape[0];
        auto k = features->data->shape[1];
        auto n = weights->data->shape[1];
        // output: (batch_size x output_features)
        auto shape = {m, n};
        auto data = std::make_shared<tensor::Tensor>(shape);
        arith::mm(features->data->data, weights->data->data, data->data, m, k, n);
        return data;
    }

    
}; //class Linear

}