#pragma once
#include <numeric>
#include <random>
#include <vector>
#include <memory>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tensor {

class Tensor {
public:
    std::vector<float> data;
    std::vector<std::size_t> shape;
    std::size_t size;

public:
    Tensor(const std::vector<std::size_t>& shape, bool rand_init = false) {
        this->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        this->data.resize(this->size);
        this->shape = shape;
        if (rand_init) {
            double limit = std::sqrt(3.0 / ((shape[0] + shape[1]) / 2.0));
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dis(-limit, limit);
            for (std::size_t i = 0; i < this->size; ++i) {
                this->data[i] = dis(gen);
            }
        }
    }
    std::shared_ptr<Tensor> transpose();

    Tensor operator+(const Tensor& other) const {
        if (this->shape != other.shape) {
            throw std::runtime_error("Shapes do not match");
        }
        Tensor result(this->shape);
        for (std::size_t i = 0; i < this->size; ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    std::vector<std::size_t> get_shape() const {
        return this->shape;
    }

    std::vector<float> get_data() const {
        return this->data;
    }
    ~Tensor() = default;
};  // class Tensor

std::shared_ptr<Tensor> pyarray_to_tensor(py::array_t<float> array);

}  // namespace tensor