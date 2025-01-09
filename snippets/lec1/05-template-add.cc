#include <iostream>

template<typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    double a = 1.0;
    double b = 2.0;
    std::cout << add<double>(a, b) << std::endl;
    return 0;
}