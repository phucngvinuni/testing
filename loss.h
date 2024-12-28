#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <cmath>

class Loss {
public:
    static float cross_entropy(const std::vector<float>& predicted, int true_label);
};

#endif // LOSS_H
