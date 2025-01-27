#pragma once
#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace operators {

template<typename T>
T mul(T a, T b) {
    return a * b;
}

template<typename T>
T id(T a) {
    return a;
}

template<typename T>
T add(T a, T b) {
    return a + b;
}

template<typename T>
T neg(T a) {
    return -a;
}

template<typename T>
float lt(T a, T b) {
    return a < b? 1.0 : 0.0;
}

template<typename T>
float eq(T a, T b) {
    return a == b? 1.0: 0.0;
}

template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

template<typename T, typename F>
auto map(const std::vector<T>& vec, F func) -> std::vector<decltype(func(std::declval<T>()))> {

    std::vector<decltype(func(std::declval<T>()))> result;
    result.reserve(vec.size());

    std::transform(vec.begin(), vec.end(), std::back_inserter(result), func);

    return result;
}

template <typename T1, typename T2, typename F>
auto zipWith(const std::vector<T1>& vec1, const std::vector<T2>& vec2, F func)
    -> std::vector<decltype(func(std::declval<T1>(), std::declval<T2>()))> {

    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<decltype(func(std::declval<T1>(), std::declval<T2>()))> result;
    result.reserve(vec1.size());

    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(func(vec1[i], vec2[i]));
    }

    return result;
}

template<typename T, typename F>
auto reduce(const std::vector<T>& vec, T init, F func) -> T {
    return std::accumulate(vec.begin(), vec.end(), init, func);
}

float is_close(float x, float y);
float sigmoid(float x);
float relu(float x);
float inv(float x);
float inv_back(float x, float d);
float relu_back(float x, float d);

auto sumList(const std::vector<float>& vec) -> float;
auto prodList(const std::vector<float>& vec) -> float;
auto addLists(const std::vector<float>& vec1, const std::vector<float>& vec2) -> std::vector<float>;
auto negList(const std::vector<float>& vec) -> std::vector<float>;
}
