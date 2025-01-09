#include "nn.h"

namespace nn {

void Linear::forward(std::vector<float>& x) {
    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t ou = 0; ou < this->out_features; ou++) {
            this->out[batch * this->out_features + ou] = 0;
            for (size_t in = 0; in < this->in_features; in++) {
                this->out[batch * this->out_features + ou] += x[batch * this->in_features + in] * this->weight[in * this->out_features + ou];
            }
        }
    }
}

void Linear::backward(std::vector<float>& x, std::vector<float>& o) {
    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t in = 0; in < this->in_features; in++) {
            for (size_t ou = 0; ou < this->out_features; ou++) {
                this->grad[in * this->out_features + ou] += x[batch * this->in_features + in] * o[batch * this->out_features + ou];
            }
        }
    }
}

void Linear::update(float lr) {
    for (size_t in = 0; in < this->in_features; in++) {
        for (size_t ou = 0; ou < this->out_features; ou++) {
            this->weight[in * this->out_features + ou] -= lr * this->grad[in * this->out_features + ou];
        }
    }
}

void Linear::zero_grad() {
    for (auto &it: this->grad) {
        it = 0.0f;
    }
}


}