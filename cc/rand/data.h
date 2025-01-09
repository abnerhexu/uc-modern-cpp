#include <random>
#include <vector>

template<typename T>
std::vector<T> generate_random_array(size_t size, T min, T max) {
    static_assert(std::is_arithmetic<T>::value, "Type must be arithmetic (integral or floating-point)");

    // static std::random_device rd;  // random device to generate seed
    static std::mt19937 gen(42); // use Mersenne Twister engine

    std::vector<T> result(size);  // vector to store random numbers

    if constexpr (std::is_integral<T>::value) {
        // generate random integer numbers
        std::uniform_int_distribution<T> dis(min, max);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        // generate random floating-point numbers
        std::uniform_real_distribution<T> dis(min, max);
        for (size_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
    }

    return result;
}