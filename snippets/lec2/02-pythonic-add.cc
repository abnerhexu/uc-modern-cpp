#include <iostream>

int main() {
    auto add = [](auto x, auto y) {
        return x+y;
    };

    auto r1 = add(1, 2);
    auto r2 = add(1.1, 2.2);
    std::cout << r1 << " " << r2 << std::endl;
    return 0;
}