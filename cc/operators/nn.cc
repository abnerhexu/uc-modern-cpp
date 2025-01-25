#include "nn.h"

namespace nn {

std::shared_ptr<tensor::Tensor> SoftmaxLoss::log_softmax(std::shared_ptr<tensor::Tensor> logits) {
    auto batch_size = logits->shape[0];
    auto num_classes = logits->shape[1];
    auto log_probs_shape = {batch_size, num_classes};
    auto log_probs = std::make_shared<tensor::Tensor>(log_probs_shape);

    for (auto i = 0; i < batch_size; i++) {
        auto max_logit = logits->data[i * num_classes];
        for (auto j = 1; j < num_classes; j++) {
            max_logit = max_logit > logits->data[i * num_classes + j] ? max_logit : logits->data[i * num_classes + j];
        }

        auto sum_exp = 0.0;
        for (auto j = 0; j < num_classes; j++) {
            log_probs->data[i * num_classes + j] = logits->data[i * num_classes + j] - max_logit;
            sum_exp += exp(log_probs->data[i * num_classes + j]);
        }

        // calculate log(softmax)
        auto log_sum_exp = log(sum_exp);
        for (auto j = 0; j < num_classes; j++) {
            log_probs->data[i * num_classes + j] -= log_sum_exp;
        }
    }

    return log_probs;
}

std::vector<std::shared_ptr<tensor::Tensor>> gradients(std::shared_ptr<Loss> loss, std::vector<std::shared_ptr<Node>> parameters) {
    loss->used = true;

    std::unordered_set<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Node>> tape;

    // 递归遍历图并构建计算图
    std::function<void(std::shared_ptr<Node>)> visit = [&](std::shared_ptr<Node> node) {
        if (nodes.find(node) == nodes.end()) {
            for (const auto& parent : node->get_parents()) {
                visit(parent);
            }
            nodes.insert(node);
            tape.push_back(node);
        }
    };

    visit(loss);
    for (const auto& param : parameters) {
        nodes.insert(param);
    }

    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<tensor::Tensor>> grads;
    for (const auto& node : nodes) {
        grads[node] = std::make_shared<tensor::Tensor>(node->data->shape);
    }
    grads[loss] = std::make_shared<tensor::Tensor>(loss->data->shape);
    grads[loss]->data[0] = 1.0;

    for (auto it = tape.rbegin(); it != tape.rend(); it++) {
        // std::cout << "tape it: " << std::endl;
        auto node = *it;
        // if (node->data->shape[0] == 1) {
        //     std::cout << "coming to squareloss" << std::endl;
        // }
        auto parent_grads = node->backward(grads[node]);
        auto parents = node->get_parents();
        for (size_t i = 0; i < parents.size(); i++) {
            // std::cout << "this grad shape: " << grads[parents[i]]->data.size() << std::endl;
            for (auto ind = 0; ind < parents[i]->data->size; ind++) {
                grads[parents[i]]->data[ind] += parent_grads[i]->data[ind];
            }
        }
    }

    std::vector<std::shared_ptr<tensor::Tensor>> result;
    for (const auto& param : parameters) {
        result.emplace_back(grads[param]);
    }

    // std::cout << "len(result): " << result.size() << std::endl;
    return result;
}

}