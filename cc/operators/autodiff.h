#pragma once
#include <vector>
#include <memory>

namespace autodiff {

template<typename T, typename F>
auto central_difference(std::vector<T>& vec, F func, std::size_t arg, float epsilon = 1e-6) {
    T original_value = vec[arg];

    vec[arg] = original_value + epsilon;
    auto rplus = func(vec);

    vec[arg] = original_value - epsilon;
    auto rminus = func(vec);

    vec[arg] = original_value;

    return (rplus - rminus) / (2 * epsilon);
}

class ScalarFunction {
public:
    std::vector<float> ctx;
    float data;
public:
    ScalarFunction(float data) : data(data) {}
}; // class ScalarFunction

class Add {
public:
    std::shared_ptr<ScalarFunction> a;
    std::shared_ptr<ScalarFunction> b;
public:
    Add(std::shared_ptr<ScalarFunction> a, std::shared_ptr<ScalarFunction> b) : a(a), b(b) {}
    std::shared_ptr<ScalarFunction> forward() {
        return std::make_shared<ScalarFunction>(a->data + b->data);
    }
    std::vector<std::shared_ptr<ScalarFunction>> backward(std::shared_ptr<ScalarFunction> d_input) {
        return {d_input, d_input};
    }
}; // class Add

}