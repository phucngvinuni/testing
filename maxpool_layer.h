#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "../utils/tensor.h"

class MaxPoolingLayer {
public:
    MaxPoolingLayer(int pool_size);
    Tensor forward(const Tensor& input);

private:
    int pool_size;
};

#endif // MAXPOOL_LAYER_H
