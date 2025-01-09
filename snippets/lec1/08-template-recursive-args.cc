#include <iostream>
template<typename T>
void my_printf(T value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Ts>
void my_printf(T value, Ts... args) {
    std::cout << value << std::endl;
    my_printf(args...);
}

int main() {
    my_printf(1, 2, "San Francisco", 1.1);
    return 0;
}