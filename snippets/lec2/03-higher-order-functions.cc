#include <iostream>
#include <functional>
#include <vector>

// define a higher order function, composing all functions together
template<typename T>
std::function<T(T)> compose(std::vector<std::function<T(T)>> functions) {
    return [functions](T x) {
        T result = x;
        // apply all these function from the right to the left
        for (auto it = functions.rbegin(); it != functions.rend(); ++it) {
            result = (*it)(result);
        }
        return result;
    };
}

int main() {
    // define a couple of simple functions
    auto add_one = [](int x) { return x + 1; };
    auto square = [](int x) { return x * x; };
    auto double_value = [](int x) { return x * 2; };

    // compose the functions
    std::vector<std::function<int(int)>> functions = {double_value, square, add_one};
    auto pipeline = compose(functions);

    // test the composed functions
    int result = pipeline(3);
    std::cout << result << std::endl;  // output: 32

    return 0;
}