#include "optimizer.h"
#include <cstddef>

void SGD::update(std::vector<std::vector<float>>& weights, std::vector<float>& biases, 
                 const std::vector<std::vector<float>>& weight_gradients, 
                 const std::vector<float>& bias_gradients, float learning_rate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * weight_gradients[i][j];
        }
        biases[i] -= learning_rate * bias_gradients[i];
    }
}
