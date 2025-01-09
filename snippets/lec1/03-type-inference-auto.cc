#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec;
    std::vector<int>::const_iterator a = vec.cbegin();
    // using type inference
    auto b = vec.cbegin();
    return 0;
}