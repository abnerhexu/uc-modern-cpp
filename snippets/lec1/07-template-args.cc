#include <iostream>

template<typename... Ts>
void magic(Ts... args) {
    std::cout << sizeof...(args) << std::endl;
}

int main() {
    magic();
    magic(1);
    magic(1, 2);
    magic(1, "hello", "world");
    return 0;
}