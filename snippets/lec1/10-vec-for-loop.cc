#include <iostream>
#include <vector>

int main() {
  std::vector<int> v = {1, 2, 3, 4, 5};
  for (auto it: v) {
    std::cout << it << std::endl;
  }
  for (auto &it: v) {
    it = it * 2;
  }
  for (auto it: v) {
    std::cout << it << std::endl;
  }
  return 0;
}