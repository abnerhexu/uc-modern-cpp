#include <vector>
#include <stdexcept>
#include <cmath>

void relu(std::vector<float>& x) {
    for (auto& i : x) {
        i = std::max(i, 0.0f);
    }
}

size_t argmax(std::vector<float>& x) {
    size_t max_index = 0;
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] > x[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

float crossEntropyLoss(std::vector<float>& y_predict, std::vector<float>& y, size_t num_class) {
    if (y_predict.size() / num_class != y.size()) {
        throw std::invalid_argument("y_predict and y must have the same size.");
    }
    float loss = 0.0f;
    constexpr float epsilon = 1e-5f;
    for (size_t i = 0; i < y.size(); i++) {
        if (y_predict[i * num_class + y[i]] <= epsilon) {
            loss += std::log(epsilon);
        }
        else {
            loss += std::log(y_predict[i * num_class + y[i]]);
        }
    }
    return -loss / y.size();
}