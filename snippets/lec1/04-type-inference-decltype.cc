#include <iostream>

int main() {
    auto x = 1;
    auto y = 2.0;
    decltype(x+y) z;
    if (std::is_same<decltype(x+y), double>::value) {
        std::cout << "x+y is double" << std::endl;
    }
    return 0;
}