#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <vector>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);

private:
    int input_size, output_size;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};

#endif // FULLY_CONNECTED_LAYER_H
