#include "../checker.h"
#include <vector>
// C++ 中，`class` 和 `struct` 之间的**唯一区别**是
// `class` 默认访问控制符是 `private`，
// `struct` 默认访问控制符是 `public`。
// READ: 访问说明符 <https://zh.cppreference.com/w/cpp/language/access>

// 这个 class 中的字段被 private 修饰，只能在 class 内部访问。
// 因此必须提供构造器来初始化字段。
// READ: 构造器 <https://zh.cppreference.com/w/cpp/language/constructor>

// 缓存的Fibonacci数列计算：
// 可以考虑将之前计算好的结果保存到vector中，计算时先检查是否已经计算过，如果已经计算过，
// 直接返回结果，反之按照递推公式进行计算

class Fibonacci {
    std::vector<size_t> cache;
    int cached;

public:
    // 实现构造器 初始化Fibonacci类的某些字段
    Fibonacci() {}

    // TODO: 实现正确的缓存优化斐波那契计算
    size_t get(int i) {
        
    }
};

int main() {
    // 现在类型拥有无参构造器，声明时会直接调用。
    // 这个写法不再是未定义行为了。
    Fibonacci fib;
    ASSERT(fib.get(10) == 55, "fibonacci(10) should be 55");
    std::cout << "fibonacci(10) = " << fib.get(10) << std::endl;
    return 0;
}
