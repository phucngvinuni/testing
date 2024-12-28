#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>

std::vector<Tensor> MNISTLoader::load_images(const std::string& path, int count, int width, int height, int depth) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Unable to open file " + path);

    file.ignore(16); // Skip header
    std::vector<Tensor> images;
    for (int i = 0; i < count; ++i) {
        Tensor img(height, width, depth);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                uint8_t pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                img.at(h, w, 0) = pixel / 255.0f; // Normalize to [0, 1]
            }
        }
        images.push_back(img);
    }
    return images;
}

std::vector<int> MNISTLoader::load_labels(const std::string& path, int count) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Unable to open file " + path);

    file.ignore(8); // Skip header
    std::vector<int> labels(count);
    for (int i = 0; i < count; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }
    return labels;
}
