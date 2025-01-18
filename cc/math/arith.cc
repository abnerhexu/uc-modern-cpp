#include "arith.h"

namespace arith {

float sqrt(float x) {
    return sqrtf(x);
}

float mean(const std::vector<int>& x) {
    return std::accumulate(x.begin(), x.end(), 0) / x.size();
}

}