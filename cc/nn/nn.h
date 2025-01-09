#pragma once
#include <string>
#include <map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <numeric>

#include "../rand/data.h"

namespace nn {

class Module {
public:
    Module();

public:
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;
    virtual void zero_grad() = 0;
};

class Linear: public Module {
private:
    std::vector<Module*> prev; // previous layers
public:
    size_t in_features;
    size_t out_features;
    size_t batch_size;
    std::vector<float> weight; // size: (in_features, out_features)
    std::vector<float> grad;
    std::vector<float> out; // out = xW
public:
    Linear(size_t in_features, size_t out_features, size_t batch_size): in_features(in_features), out_features(out_features), batch_size(batch_size) {
        this->weight = generate_random_array<float>(in_features * out_features, -1.0, 1.0);
        this->grad = std::vector<float>(in_features * out_features, 0.0);
    };
    void forward(std::vector<float>& x);
    void backward(std::vector<float>& x, std::vector<float>& o);
    void update(float lr);
    void zero_grad();
};

}