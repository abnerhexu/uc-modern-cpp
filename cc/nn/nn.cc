#include "nn.h"

namespace nn {

std::vector<float> DataNode::forward(std::vector<float>& x) {
    return x;
}

std::vector<float> DataNode::backward(std::vector<float>& gradient, std::vector<float>& x) {
    return {};
}

void Parameter::update(Node* direction, float learning_rate) {
    auto constant = dynamic_cast<Constant*>(direction);
    if (!constant) {
        throw std::runtime_error("Parameter::update: direction must be a Constant");
    }
    if (constant->data.size() != this->data.size()) {
        throw std::runtime_error("Parameter::update: direction size must match parameter size");
    }
    for (int i = 0; i < this->data.size(); i++) {
        this->data[i] -= learning_rate * constant->data[i];
    }
}

}