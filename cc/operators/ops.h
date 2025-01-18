#include <cmath>
namespace operators {

template<typename T>
T mul(T a, T b) {
    return a * b;
}

template<typename T>
T id(T a) {
    return a;
}

template<typename T>
T add(T a, T b) {
    return a + b;
}

template<typename T>
T neg(T a) {
    return -a;
}

template<typename T>
float lt(T a, T b) {
    return a < b? 1.0 : 0.0;
}

template<typename T>
float eq(T a, T b) {
    return a == b? 1.0: 0.0;
}

template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

}
