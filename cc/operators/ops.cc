#include "ops.h"

namespace operators {
static float epsilon = 1e-6;

float is_close(float x, float y) {
    return (x - y) < epsilon && (y - x) > -epsilon? 1.0 : 0.0;
}

float sigmoid(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + expf(-x));
    }
    else {
        return expf(x) / (1.0 + expf(x));
    }
}

float relu(float x) {
    return x > 0.0 ? x : 0.0;
}

float inv(float x) {
    return 1.0 / x;
}

float inv_back(float x, float d) {
    return -d * inv(x * x);
}

float relu_back(float x, float d) {
    return x > 0.0 ? d : 0.0;
}

}
