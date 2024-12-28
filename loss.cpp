#include "loss.h"

float Loss::cross_entropy(const std::vector<float>& predicted, int true_label) {
    return -std::log(predicted[true_label] + 1e-7); // Add small value to avoid log(0)
}
