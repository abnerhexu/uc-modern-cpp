#include "../checker.h"
#include <vector>

// READ: 复制构造函数 <https://zh.cppreference.com/w/cpp/language/copy_constructor>
// READ: 函数定义（显式弃置）<https://zh.cppreference.com/w/cpp/language/function>


class Fibonacci {
    std::vector<size_t> cache;
    int cached;

public:
    // TODO: 实现动态设置容量的构造器
    Fibonacci(int capacity) {}

    // TODO: 实现复制构造器
    Fibonacci(Fibonacci const &other) {}

    // TODO: 实现正确的缓存优化斐波那契计算
    size_t get(int i) {
        
    }

    // NOTICE: 不要修改这个方法
    // NOTICE: 名字相同参数也相同，但 const 修饰不同的方法是一对重载方法，可以同时存在
    //         本质上，方法是隐藏了 this 参数的函数
    //         const 修饰作用在 this 上，因此它们实际上参数不同
    size_t get(int i) const {
        if (i <= cached) {
            return cache[i];
        }
        ASSERT(false, "i out of range");
    }
};

int main(int argc, char **argv) {
    Fibonacci fib(12);
    ASSERT(fib.get(10) == 55, "fibonacci(10) should be 55");
    Fibonacci const fib_ = fib;
    ASSERT(fib_.get(10) == fib.get(10), "Object cloned");
    return 0;
}
