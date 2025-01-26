#include "tensor.h"

namespace tensor {

std::shared_ptr<Tensor> Tensor::transpose() {
    // if (shape.size() != 2) {
    //     throw std::runtime_error("Transpose is only supported for 2D tensors.");
    // }

    std::size_t rows = shape[0];
    std::size_t cols = shape[1];

    std::vector<float> transposed_data(size);

    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            transposed_data[j * rows + i] = this->data[i * cols + j];
        }
    }
    // not in-place transform
    auto transposed_shape = {cols, rows};
    auto t = std::make_shared<Tensor>(transposed_shape);
    t->data = transposed_data;
    return t;
}

std::shared_ptr<Tensor> pyarray_to_tensor(py::array_t<float> array) {
    py::buffer_info info = array.request();
    float* dataPtr = static_cast<float*>(info.ptr);
    std::vector<std::size_t> shape = {};
    for (auto &it: info.shape) {
        shape.push_back(it);
    }
    auto tensor = std::make_shared<Tensor>(shape);
    std::vector<float> result(dataPtr, dataPtr + info.size);
    tensor->data = result;
    return tensor;
}

std::shared_ptr<Tensor> argmax(const std::shared_ptr<Tensor>& tensor, int axis) {
    // you only need to handle the two dimensional tensor, and the axis can be either 0 or 1
    // the tensor's shape is (batch_size, features)
    // if the axis is 0, it outputs a tensor (1, features)
    // if the axis is 1, it outputs a tensor (batch_size, 1)

    // compute the output's shape
    std::vector<std::size_t> output_shape = tensor->shape;
    output_shape.erase(output_shape.begin() + axis);

    auto result = std::make_shared<Tensor>(output_shape);

    std::vector<std::size_t> indices(tensor->shape.size(), 0);
    std::vector<std::size_t> output_indices(output_shape.size(), 0);

        for (std::size_t i = 0; i < result->size; ++i) {
        float max_value = -std::numeric_limits<float>::infinity();
        std::size_t max_index = 0;

        // 遍历axis维度的所有值
        for (std::size_t j = 0; j < tensor->shape[axis]; j++) {
            indices[axis] = j;
            float value = tensor->get(indices);
            if (value > max_value) {
                max_value = value;
                max_index = j;
            }
        }

        // 设置输出Tensor的值
        result->set(output_indices, static_cast<float>(max_index));

        // 更新输出索引
        for (int k = output_indices.size() - 1; k >= 0; k--) {
            output_indices[k]++;
            if (output_indices[k] < output_shape[k]) {
                break;
            }
            output_indices[k] = 0;
        }

        // 更新输入索引
        for (int k = indices.size() - 1; k >= 0; k--) {
            if (k == axis) continue; // 跳过axis维度
            indices[k]++;
            if (indices[k] < tensor->shape[k]) {
                break;
            }
            indices[k] = 0;
        }
    }
    return result;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& tensor) {
    std::vector<std::size_t> shape = {1};
    auto result = std::make_shared<Tensor>(shape);
    auto sum = 0.0f;
    for (auto &it: tensor->data) {
        sum += it;
    }
    sum /= tensor->size;
    result->data[0] = sum;
    return result;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& tensor) {
    auto result = std::make_shared<Tensor>(tensor->shape);
    for (auto i = 0; i < tensor->size; i++) {
        result->data[i] = expf(tensor->data[i]);
    }
    return result;
}

}