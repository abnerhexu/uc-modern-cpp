#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>
#include <vector>
#include "function/function.h"

namespace nn {

class Node {
   public:
    Node() {}
    virtual ~Node() {}
};  // class Node

class DataNode : public Node {
   public:
    std::vector<float> data;
    std::vector<Node*> parents;

   public:
    DataNode(std::vector<float> data)
        : data(data) {}
    std::vector<float> forward(std::vector<float>& x);
    std::vector<float> backward(std::vector<float>& gradient, std::vector<float>& x);
};  // class DataNode

class Constant : public DataNode {
    /*
    A Constant node is used to represent:
    * Input features
    * Output labels
    * Gradients computed by back-propagation
    You should not need to construct any Constant nodes directly; they will
    instead be provided by either the dataset or when you call `nn.gradients`.
    */
   public:
    Constant(std::vector<float> data)
        : DataNode(data) {}
};  // class Constant

class Parameter : public DataNode {
   public:
    std::vector<size_t> shape;
    size_t size = 0;

   public:
    Parameter(const std::vector<size_t> shape, bool zero_filling = false)
        : shape(shape), DataNode({}) {
        if (this->shape.size() != 2) {
            throw std::runtime_error("Parameter shape must be a tuple of 2 integers");
        }
        bool allPositive = std::all_of(shape.begin(), shape.end(), [](size_t value) {
            return value > 0;
        });
        if (!allPositive) {
            throw std::runtime_error("Parameter shape must be a tuple of positive integers");
        }
        this->size = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());
        if (!zero_filling) {
            this->data = generate_random_array<float>(this->size, -1.0, 1.0);
        } else {
            this->data = std::vector<float>(this->size, 0.0);
        }
    }

    void update(Node* direction, float learning_rate);
};  // class Parameter

class FunctionNode : public Node {
   public:
    std::vector<Node*> parents;
    Parameter* out;
    Parameter* back;

   public:
    FunctionNode() {}
    template <typename... Args>
    FunctionNode(Args... args) {
        // 将参数包展开并存入 vector
        (this->parents.push_back(args), ...);
        this->alloc_out();
        this->forward();
    }
    virtual void forward() = 0;
    virtual void alloc_out() = 0;
};  // class FunctionNode

class Add : public FunctionNode {
    /*
    Add matrices element-wise
    */
   public:
    Add(Node* a, Node* b)
        : FunctionNode(a, b) {}

    void forward() {
        auto data_a = dynamic_cast<Parameter*>(this->parents[0]);
        auto data_b = dynamic_cast<Parameter*>(this->parents[1]);
        if (data_a->data.size() != data_b->data.size()) {
            throw std::runtime_error("Add: data size mismatch");
        }
        for (int row = 0; row < data_a->shape[0]; row++) {
            for (int col = 0; col < data_a->shape[1]; col++) {
                this->out->data[row * data_a->shape[1] + col] = data_a->data[row * data_a->shape[1] + col] + data_b->data[row * data_b->shape[1] + col];
            }
        }
    };

    void alloc_out() {
        auto data_a = dynamic_cast<Parameter*>(this->parents[0]);
        this->out = new Parameter(data_a->shape, true);
    }

    std::tuple<Parameter*, Parameter*> backward(std::vector<Parameter*> payload) {
        // the first is gradient
        // the second is input
        auto gradient = payload[0];
        auto input = payload[1];
        if (gradient->shape != input->shape) {
            throw std::runtime_error("Add: gradient size mismatch");
        }
        auto ga = gradient;
        auto gb = new Parameter({gradient->shape[1], 1}, true);
        for (auto col = 0; col < gradient->shape[1]; col++) {
            float psum = 0.0f;
            for (auto row = 0; row < gradient->shape[0]; row++) {
                psum += gradient->data[row * gradient->shape[1] + col];
            }
            gb->data[col] = psum;
        }
    }
};  // class Add

class DotProduct: public FunctionNode {
public:
    DotProduct(Node* features, Node* weights): FunctionNode(features, weights) {}
    void alloc_out() {
        auto data_a = dynamic_cast<Parameter*>(this->parents[0]);
        auto data_b = dynamic_cast<Parameter*>(this->parents[1]);
        this->out = new Parameter({data_a->shape[1], data_b->shape[1]}, true);
    }
    void forward() {
        auto data_a = dynamic_cast<Parameter*>(this->parents[0]);
        auto data_b = dynamic_cast<Parameter*>(this->parents[1]);
        if (data_a->shape[1] != data_b->shape[0]) {
            throw std::runtime_error("DotProduct: data size mismatch");
        } else if (data_a->shape[0] != data_b->shape[1]) {
            throw std::runtime_error("DotProduct: data size mismatch");
        }
        mma(data_a->data, transpose(data_b->data, data_b->shape[0], data_b->shape[1]), this->out->data, data_a->shape[0], data_b->shape[0], data_a->shape[1]);
    }
    std::tuple<Parameter*, Parameter*> backward(std::vector<Parameter*> payload) {
        auto gradient = payload[0];
        auto input1 = payload[1];
        auto input2 = payload[2];
        if (gradient->shape[0] != input1->shape[0]) {
            throw std::runtime_error("DotProduct: gradient size mismatch");
        }
        if (gradient->shape[1] != 1) {
            throw std::runtime_error("DotProduct: gradient shape[1] should be 1");
        }
        auto ga = new Parameter({gradient->shape[0], input2->shape[1]}, true);
        auto gb = new Parameter({gradient->shape[1], input1->shape[1]}, true);
        mma(gradient->data, input2->data, ga->data, gradient->shape[0], input2->shape[1], gradient->shape[1]);
        mma(transpose(gradient->data, gradient->shape[0], gradient->shape[1]), input1->data, gb->data, gradient->shape[1], input1->shape[1], gradient->shape[0]);
        return std::make_tuple(ga, gb);
    }
}; // class DotProduct

class Linear: public FunctionNode {
public:
    Linear(Node* features, Node* weights): FunctionNode(features, weights) {}
    void alloc_out() {
        auto features = dynamic_cast<Parameter*>(this->parents[0]);
        auto weights = dynamic_cast<Parameter*>(this->parents[1]);
        this->out = new Parameter({features->shape[0], weights->shape[1]}, true);
    }
    void forward() {
        auto features = dynamic_cast<Parameter*>(this->parents[0]);
        auto weights = dynamic_cast<Parameter*>(this->parents[1]);
        if (features->shape[1] != weights->shape[0]) {
            throw std::runtime_error("Linear: data size mismatch");
        }
        mma(features->data, weights->data, this->out->data, features->shape[0], weights->shape[1], weights->shape[0]);
    }
    std::tuple<Parameter*, Parameter*> backward(std::vector<Parameter*> payload) {
        auto gradient = payload[0];
        auto input1 = payload[1];
        auto input2 = payload[2];
        if (gradient->shape[0] != input1->shape[0]) {
            throw std::runtime_error("Linear: gradient size mismatch");
        }
        if (gradient->shape[1] != input2->shape[1]) {
            throw std::runtime_error("Linear: gradient shape[1] should be input2 shape[1]");
        }
        auto ga = new Parameter({gradient->shape[0], input2->shape[0]}, true);
        auto gb = new Parameter({input1->shape[1], gradient->shape[1]}, true);
        mma(gradient->data, transpose(input2->data, input2->shape[0], input2->shape[1]), ga->data, gradient->shape[0], input2->shape[0], gradient->shape[1]);
        mma(transpose(input1->data, input1->shape[0], input1->shape[1]), gradient->data, gb->data, input1->shape[1], gradient->shape[1], input1->shape[0]);
        return std::make_tuple(ga, gb);
    }
}; // class Linear

class ReLU: public FunctionNode {
public:
    ReLU(Node* input): FunctionNode(input) {}
    void alloc_out() {
        auto input = dynamic_cast<Parameter*>(this->parents[0]);
        this->out = new Parameter(input->shape, true);
    }
    void forward() {
        auto input = dynamic_cast<Parameter*>(this->parents[0]);
        for (auto i = 0; i < input->size; i++) {
            this->out->data[i] = std::max(input->data[i], 0.0f);
        }
    }
    std::tuple<Parameter*> backward(std::vector<Parameter*> payload) {
        auto gradient = payload[0];
        auto input = payload[1];
        auto g = new Parameter(gradient->shape, true);
        for (auto i = 0; i < gradient->size; i++) {
            g->data[i] = gradient->data[i] * (input->data[i] > 0.0f ? 1.0f : 0.0f);
        }
        return std::make_tuple(g);
    }
}; // class relu

class SquareLoss: public FunctionNode {
public:
    SquareLoss(Node* inputs): FunctionNode(inputs) {}
    void alloc_out() {
        this->out = new Parameter({1}, true);
    }
    void forward() {
        auto input1 = dynamic_cast<Parameter*>(this->parents[0]);
        auto input2 = dynamic_cast<Parameter*>(this->parents[1]);
        if (input1->shape != input2->shape) {
            throw std::runtime_error("SquareLoss: input size mismatch");
        }
        for (auto i = 0; i < input1->size; i++) {
            this->out->data[0] += (input1->data[i] - input2->data[i]) * (input1->data[i] - input2->data[i]) / 2.0;
        }
        this->out->data[0] = this->out->data[0] / input1->size;
    }
    std::tuple<Parameter*, Parameter*> backward(std::vector<Parameter*> payload) {
        auto gradient = payload[0];
        auto input1 = payload[1];
        auto input2 = payload[2];
        auto ga = new Parameter(input1->shape, true);
        auto gb = new Parameter(input2->shape, true);
        for (auto i = 0; i < input1->size; i++) {
            ga->data[i] = gradient->data[0] * (input1->data[i] - input2->data[i]) / input1->size;
            gb->data[i] = gradient->data[0] * (input2->data[i] - input1->data[i]) / input1->size;
        }
        return std::make_tuple(ga, gb);
    }
}; // class SquareLoss

class SoftmaxLoss: public FunctionNode {
public:
    SoftmaxLoss(Node* inputs): FunctionNode(inputs) {}
    void alloc_out() {
        this->out = new Parameter({1}, true);
    }
    void forward() {
        auto input = dynamic_cast<Parameter*>(this->parents[0]);
        auto target = dynamic_cast<Parameter*>(this->parents[1]);
        if (input->shape != target->shape) {
            throw std::runtime_error("SoftmaxLoss: input size mismatch");
        }
        
    }
}; // class SoftmaxLoss

} // namespace nn