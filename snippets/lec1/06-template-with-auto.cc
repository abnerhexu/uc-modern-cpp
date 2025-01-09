#include <iostream>

template<typename T, typename U>
auto add(T x, U y){
    return x + y;
}

int main() {
    int a = 1;
    double b = 2.71828;
    std::cout << add(a, b) << std::endl;
    if (std::is_same<decltype(add(a, b)), double>::value) {
        std::cout << "result type of add(a, b) is double!" << std::endl;
    }
    return 0;
}