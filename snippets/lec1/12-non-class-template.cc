#include <iostream>

// define a template for calculating factorial
template<int N>
struct Factorial {
    static const int value = N * Factorial<N - 1>::value;
};

// specialized template to end the factorial recursive
template<>
struct Factorial<0> {
    static const int value = 1;
};

int main() {
    std::cout << "Factorial of 5 is: " << Factorial<5>::value << std::endl;
    std::cout << "Factorial of 10 is: " << Factorial<10>::value << std::endl;
    return 0;
}
