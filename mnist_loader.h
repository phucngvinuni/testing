#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include "tensor.h"

class MNISTLoader {
public:
    static std::vector<Tensor> load_images(const std::string& path, int count, int width, int height, int depth);
    static std::vector<int> load_labels(const std::string& path, int count);
};

#endif // MNIST_LOADER_H
