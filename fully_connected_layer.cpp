#include "fully_connected_layer.h"
#include <cmath>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    weights.resize(output_size, std::vector<float>(input_size));
    biases.resize(output_size);

    for (int o = 0; o < output_size; ++o) {
        biases[o] = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            weights[o][i] = distribution(generator);
        }
    }
}

std::vector<float> FullyConnectedLayer::forward(const std::vector<float>& input) {
    std::vector<float> output(output_size, 0.0f);
    for (int o = 0; o < output_size; ++o) {
        for (int i = 0; i < input_size; ++i) {
            output[o] += input[i] * weights[o][i];
        }
        output[o] += biases[o];
        output[o] = std::max(0.0f, output[o]); // ReLU activation
    }
    return output;
}

std::vector<float> FullyConnectedLayer::backward(const std::vector<float>& grad_output) {
    std::vector<float> grad_input(input_size, 0.0f);
    // Compute gradients for weights, biases, and input...
    return grad_input;
}


