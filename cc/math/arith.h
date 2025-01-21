#include <cmath>
#include <vector>
#include <numeric>

namespace arith {

float sqrt(float x);
float mean(const std::vector<int>& x);

template<typename T>
void mm(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i * n + j] = 0;
            for (size_t l = 0; l < k; ++l) {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

template<typename T>
void vector_scalar_max(const std::vector<T>& a, std::vector<T> &b, T scalar) {
    for (size_t i = 0; i < a.size(); ++i) {
        b[i] = std::max(a[i], scalar);
}
}