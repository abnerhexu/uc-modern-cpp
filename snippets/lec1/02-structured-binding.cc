#include <iostream>
#include <tuple>

std::tuple<int, double, std::string> get_data() {
    return std::make_tuple(1, 2.71828, "hello");
}

int main() {
    auto [a, b, c] = get_data();
    std::cout << a << " " << b << " " << c << std::endl;
    return 0;
}