#include "tensor.h"

namespace tensor {

std::shared_ptr<Tensor> Tensor::transpose() {
    // if (shape.size() != 2) {
    //     throw std::runtime_error("Transpose is only supported for 2D tensors.");
    // }

    std::size_t rows = shape[0];
    std::size_t cols = shape[1];

    std::vector<float> transposed_data(size);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            transposed_data[j * rows + i] = this->data[i * cols + j];
        }
    }
    // not in-place transform
    auto transposed_shape = {cols, rows};
    auto t = std::make_shared<Tensor>(transposed_shape);
    t->data = transposed_data;
    return t;
}

}