#include "ops.h"

namespace operators {
static float epsilon = 1e-6;

float is_close(float x, float y) {
    auto diff = x - y > 0.0 ? x - y: y - x;
    return diff < epsilon? 1.0 : 0.0;
}

float sigmoid(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + expf(-x));
    }
    else {
        return expf(x) / (1.0 + expf(x));
    }
}

float relu(float x) {
    return x > 0.0 ? x : 0.0;
}

float inv(float x) {
    return 1.0 / x;
}

float inv_back(float x, float d) {
    return -d * inv(x * x);
}

float relu_back(float x, float d) {
    return x > 0.0 ? d : 0.0;
}

auto sumList(const std::vector<float>& vec) -> float {
    return reduce(vec, 0.0f, add<float>);
}

auto prodList(const std::vector<float>& vec) -> float {
    return reduce(vec, 1.0f, mul<float>);
}

auto addLists(const std::vector<float>& vec1, const std::vector<float>& vec2) -> std::vector<float> {
    return zipWith(vec1, vec2, add<float>);
}

auto negList(const std::vector<float>& vec) -> std::vector<float> {
    return map(vec, neg<float>);
}
}
