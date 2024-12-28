#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "../utils/tensor.h"
#include <vector>

class FlattenLayer {
public:
    std::vector<float> forward(const Tensor& input);
};

#endif // FLATTEN_LAYER_H
