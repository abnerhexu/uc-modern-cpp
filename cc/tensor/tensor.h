#include <numeric>
#include <random>
#include <vector>
#include <memory>

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
    ~Tensor() = default;
};  // class Tensor

}  // namespace tensor