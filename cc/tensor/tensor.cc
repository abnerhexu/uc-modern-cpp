#include "tensor.h"

namespace tensor {

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    std::size_t size;
public:
  Tensor(const std::vector<int>& shape): shape(shape) {
    this->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    this->data.resize(this->size);
  }
  ~Tensor() = default;
}; // class Tensor

}