#include <numeric>
#include <random>
#include <vector>
#include <memory>
#include <stdexcept>

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
        if (rand_init) {
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
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
    ~Tensor() = default;
};  // class Tensor

}  // namespace tensor