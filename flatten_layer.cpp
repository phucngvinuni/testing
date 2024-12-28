#include "flatten_layer.h"

std::vector<float> FlattenLayer::forward(const Tensor& input) {
    std::vector<float> output;
    for (int d = 0; d < input.getDepth(); ++d) {
        for (int h = 0; h < input.getHeight(); ++h) {
            for (int w = 0; w < input.getWidth(); ++w) {
                output.push_back(input.at(h, w, d));
            }
        }
    }
    return output;

}
