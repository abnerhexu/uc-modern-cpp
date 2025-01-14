#include <algorithm>
#include <iostream>
#include <vector>

struct CppRecord {
    size_t id;
    double value;
};

int main() {
    std::vector<CppRecord> records = {
        {1, 3.5},
        {2, 2.0},
        {3, 3.5},
        {4, 1.0},
        {5, 2.0}};

    std::sort(records.begin(), records.end(), [](const CppRecord& a, const CppRecord& b) {
        if (a.value != b.value) {
            return a.value < b.value;
        } else {
            return a.id < b.id;
        }
    });

    for (const auto& record : records) {
        std::cout << record.id << " " << record.value << std::endl;
    }
    return 0;
}