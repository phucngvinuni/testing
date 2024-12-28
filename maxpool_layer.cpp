#include "maxpool_layer.h"
#include <algorithm>

MaxPoolingLayer::MaxPoolingLayer(int pool_size) : pool_size(pool_size) {}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    int out_height = input.getHeight() / pool_size;
    int out_width = input.getWidth() / pool_size;
    Tensor output(out_height, out_width, input.getDepth());

    for (int d = 0; d < input.getDepth(); ++d) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float max_val = -99999999;
                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        max_val = std::max(max_val, input.at(h * pool_size + ph, w * pool_size + pw, d));
                    }
                }
                output.at(h, w, d) = max_val;
            }
        }
    }
    return output;
}
