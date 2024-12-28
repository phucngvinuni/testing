#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

class SGD {
public:
    static void update(std::vector<std::vector<float>>& weights, std::vector<float>& biases, 
                       const std::vector<std::vector<float>>& weight_gradients, 
                       const std::vector<float>& bias_gradients, float learning_rate);
};

#endif // OPTIMIZER_H
