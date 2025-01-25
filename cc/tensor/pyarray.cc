#include "pyarray.h"

namespace pyarr {

std::vector<float> ndarray_to_vector(py::array_t<float> array) {
    py::buffer_info info = array.request();
    float* dataPtr = static_cast<float*>(info.ptr);
    std::vector<float> result(dataPtr, dataPtr + info.size);
    return result;
}

}