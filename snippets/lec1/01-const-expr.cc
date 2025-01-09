#include <iostream>

constexpr int fibonacci(const int n) {
    return n == 1 || n == 2 ? 1 : fibonacci(n-1)+fibonacci(n-2);
}

int main(){
    int fib_5 = fibonacci(5);
    std::cout << "fib 5: " << fib_5 << std::endl;
    return 0;
}