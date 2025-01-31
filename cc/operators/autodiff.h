#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <unordered_map>

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
    float data;
    float grad;
    int degree = 0;
public:
    ScalarFunction() {}
}; // class ScalarFunction

class Add: public ScalarFunction {
public:
    std::shared_ptr<ScalarFunction> a;
    std::shared_ptr<ScalarFunction> b;
public:
    Add(std::shared_ptr<ScalarFunction> a, std::shared_ptr<ScalarFunction> b): a(a), b(b) {
        this->data = a->data + b->data;
        this->degree = 2;
    }
    float forward() {
        return a->data + b->data;
    }
    std::vector<float> backward(float d_input) {
        return {d_input, d_input};
    }
}; // class Add

class Log: public ScalarFunction {
public:
    std::shared_ptr<ScalarFunction> a;
public:
    Log(std::shared_ptr<ScalarFunction> a): a(a) {
        this->data = this->forward();
        this->degree = 1;
    }
    float forward() {
        return logf(a->data);
    }
    std::vector<float> backward(float d_input) {
        return {(1.0f * d_input / a->data)};
    }
}; // class Log

class Mul: public ScalarFunction {
public:
    std::shared_ptr<ScalarFunction> a;
    std::shared_ptr<ScalarFunction> b;
public:
    Mul(std::shared_ptr<ScalarFunction> a, std::shared_ptr<ScalarFunction> b) : a(a), b(b) {
        this->data = this->forward();
        this->degree = 2;
    }
    float forward() {
        return a->data * b->data;
    }
    std::vector<float> backward(float d_input) {
        return {d_input * b->data, d_input * a->data};
    }
}; // class Mul

class Inv: public ScalarFunction {
public:
    std::shared_ptr<ScalarFunction> a;
public:
    Inv(std::shared_ptr<ScalarFunction> a): a(a) {
        this->data = this->forward();
        this->degree = 1;
    }
    float forward() {
        return 1.0f / a->data;
    }
    std::vector<float> backward(float d_input) {
        return {-1.0f * d_input / (a->data * a->data)};
    }
}; // class Inv

class Sigmoid: public ScalarFunction {
public:
    std::shared_ptr<ScalarFunction> a;
public:
    Sigmoid(std::shared_ptr<ScalarFunction> a): a(a) {
        this->data = this->forward();
        this->degree = 1;
    }
    float forward() {
        if (this->a->data >= 0.0) {
            return 1.0 / (1.0 + expf(-this->a->data));
        }
        else {
            return expf(this->a->data) / (1.0 + expf(this->a->data));
        }
    }
    std::vector<float> backward(float d_input) {
        return {d_input * this->data * (1.0f - this->data)};
    }
}; // class Sigmoid
}