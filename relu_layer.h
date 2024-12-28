#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../utils/tensor.h"

class ReLULayer {
public:
    Tensor forward(const Tensor& input);
};

#endif // RELU_LAYER_H
