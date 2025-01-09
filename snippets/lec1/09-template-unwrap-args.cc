#include <iostream>

template<typename T, typename... Ts>
void my_printf(T t0, Ts... t) {
    std::cout << t0 << std::endl;
    if constexpr (sizeof...(t) > 0) my_printf(t...);
}

int main() {
    my_printf(1, 2, "San Francisco", 1.1);
    return 0;
}