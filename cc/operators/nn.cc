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

}