#include <memory>
#include <iostream>
void foo(std::shared_ptr<int> p) {
  (*p)++;
}

void create() {
  std::shared_ptr<int> p = std::make_shared<int>(42);
  foo(p);
  // p is still valid
  std::cout << *p << std::endl;
} // when leaving this scope, p will be destroyed (deleted)