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
    virtual std::vector<std::shared_ptr<tensor::Tensor>> backward(std::shared_ptr<tensor::Tensor> gradient) = 0;
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
    std::vector<std::shared_ptr<tensor::Tensor>> gradient;
public:
    template <typename... Args>
    FunctionNode(Args ... args) {
        this->objects = {std::dynamic_pointer_cast<Node>(
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
        auto outNode = std::make_shared<tensor::Tensor>(a->data->shape);
        for (auto i = 0; i < a->data->size; i++) {
            outNode->data[i] = a->data->data[i] + b->data->data[i];
        }
        return outNode;
    }
    std::vector<std::shared_ptr<tensor::Tensor>> backward(std::shared_ptr<tensor::Tensor> gradient) override {
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
        auto outNode = std::make_shared<tensor::Tensor>(shape);
        arith::mm(features->data->data, weights->data->data, outNode->data, m, k, n);
        return outNode;
    }

    std::vector<std::shared_ptr<tensor::Tensor>> backward(std::shared_ptr<tensor::Tensor> gradient) override {
        auto features = this->objects[0];
        auto weights = this->objects[1];
        // gradient.shape[0] == features.shape[0]
        // gradient.shape[1] == weights.shape[1]
        auto grad_features_shape = {gradient->shape[0], weights->data->shape[0]};
        auto grad_features = std::make_shared<tensor::Tensor>(grad_features_shape);
        auto grad_weights_shape = {features->data->shape[1], gradient->shape[1]};
        auto grad_weights = std::make_shared<tensor::Tensor>(grad_weights_shape);
        arith::mm(gradient->data, weights->data->transpose()->data, grad_features->data, gradient->shape[0], gradient->shape[1], weights->data->shape[0]);
        arith::mm(features->data->transpose()->data, gradient->data, grad_weights->data, features->data->shape[1], features->data->shape[0], gradient->shape[1]);
        return {grad_features, grad_weights};
    }
}; //class Linear

class ReLU: public FunctionNode {
public:
    ReLU(std::shared_ptr<Node> a) : FunctionNode(a) {}
    std::shared_ptr<tensor::Tensor> forward() override {
        auto outNode = std::make_shared<tensor::Tensor>(this->objects[0]->data->shape);
        arith::vector_scalar_max(this->objects[0]->data->data, outNode->data, 0.0f);
    }
    std::vector<std::shared_ptr<tensor::Tensor>> backward(std::shared_ptr<tensor::Tensor> gradient) override {
        auto grads = std::make_shared<tensor::Tensor>(this->objects[0]->data->shape);
        for (auto i = 0; i < grads->size; i++) {
            if (this->objects[0]->data->data[i] > 0) {
                grads->data[i] = gradient->data[i] * 1.0f;
            }
            else {
                grads->data[i] = 0.0f;
            }
        }
        return {grads};
    }
}; // class ReLU

class SquareLoss: public FunctionNode {
public:
    SquareLoss(std::shared_ptr<Node> a, std::shared_ptr<Node> b): FunctionNode(a, b) {}
    std::shared_ptr<tensor::Tensor> forward() {
        // a: a Node with shape (batch_size x dim)
        // b: a Node with shape (batch_size x dim)
        auto a = this->objects[0];
        auto b = this->objects[1];
        double meanv = 0.0;
        for (auto i = 0; i < a->data->size; i++) {
            meanv += (a->data->data[i] - b->data->data[i]) * (a->data->data[i] - b->data->data[i]) / 2.0;
        }
        meanv /= a->data->size;
        auto res_shape = {1};
        auto res = std::make_shared<tensor::Tensor>(res_shape);
        res->data[0] = (float)meanv;
        return res;
    }
}; // class SquareLoss


}