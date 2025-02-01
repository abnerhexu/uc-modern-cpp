#include "autodiff.h"

namespace autodiff {

std::vector<std::shared_ptr<ScalarFunction>> topoSort(const std::vector<std::shared_ptr<ScalarFunction>>& scalars) {
    std::vector<std::shared_ptr<ScalarFunction>> sorted;
    std::vector<std::shared_ptr<ScalarFunction>> frontier;
    std::unordered_map<std::shared_ptr<ScalarFunction>, int> degree;
    for (auto it: scalars) {
        if (it->degree == 0) {
            frontier.push_back(it);
        }
        else {
            degree.insert({it, it->degree});
        }
    }
    while (!frontier.empty()) {
        auto back = frontier.back();
        sorted.push_back(back);
        for (auto &it: degree) {
            if (it.second > 0 && it.first == back) {
                it.second--;
                if (it.second == 0) {
                    frontier.push_back(it.first);
                }
            }
        }
    }
    return sorted;
}

}