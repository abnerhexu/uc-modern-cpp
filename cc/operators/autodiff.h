#include <vector>

namespace autodiff {

template<typename T, typename F>
auto central_difference(std::vector<T>& vec, F func, std::size_t arg, float epsilon = 1e-6) {
    T original_value = vec[arg];

    vec[arg] = original_value + epsilon;
    auto rplus = func(vec);

    vec[arg] = original_value - epsilon;
    auto rminus = func(vec);

    vec[arg] = original_value;

    return (rplus - rminus) / (2 * epsilon);
}

}