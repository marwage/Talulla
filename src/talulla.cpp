// 2021 Copyright Marcel Wagenländer

#include "tensors.hpp"
#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "convolution.hpp"
#include <iostream>
#include <random>


double abs_difference(Matrix<float> *a, Matrix<float> *b) {
    double abs_diff = 0.0;
    for (long i = 0; i < a->size_; i = i + 1) {
        float diff = a->values_[i] - b->values_[i];
        if (diff < 0.0) {
            abs_diff = abs_diff - (double) diff;
        } else {
            abs_diff = abs_diff + (double) diff;
        }
    }

    return abs_diff;
}


double abs_difference(float *a, float *b, long size) {
    double abs_diff = 0.0;
    for (long i = 0; i < size; i = i + 1) {
        float diff = a[i] - b[i];
        if (diff < 0.0) {
            abs_diff = abs_diff - (double) diff;
        } else {
            abs_diff = abs_diff + (double) diff;
        }
    }

    return abs_diff;
}


void dropout() {
    long num_rows = 384;
    long num_columns = 786;
    CudaHelper cuda_helper;
    float probability = 0.2;
    unsigned long long seed = 123456;

    Matrix<float> features(num_rows, num_columns, true);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (long i = 0; i < features.size_; i = i + 1) {
        features.values_[i] = distribution(generator);
    }

    Matrix<float> activations_1(num_rows, num_columns, true);
    Dropout dropout_layer_1(&cuda_helper, probability, seed);
    dropout_layer_1.forward(&features, &activations_1);

    Matrix<float> activations_2(num_rows, num_columns, true);
    Dropout dropout_layer_2(&cuda_helper, probability, seed);
    dropout_layer_2.forward(&features, &activations_2);

    double abs_diff = abs_difference(&activations_1, &activations_2);
    std::cout << "Dropout" << std::endl;
    std::cout << "Absolute difference " << abs_diff << std::endl;
}


void convolution() {
    long batch_size = 1;
    long num_channels = 3;
    long height = 256;
    long width = 512;
    CudaHelper cuda_helper;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    long x_size = batch_size * num_channels * height * width;
    float *x;
    check_cuda(cudaMallocHost(&x, x_size * sizeof(float)));
    for (long i = 0; i < x_size; i = i + 1) {
        x[i] = distribution(generator); 
    }

    long y_size = x_size;
    float *y_1;
    check_cuda(cudaMallocHost(&y_1, y_size * sizeof(float)));
    float *y_2;
    check_cuda(cudaMallocHost(&y_2, y_size * sizeof(float)));

    long filter_size = num_channels * num_channels * height * width;
    float *w;
    check_cuda(cudaMallocHost(&w, filter_size * sizeof(float)));
    for (long i = 0; i < filter_size; i = i + 1) {
        w[i] = distribution(generator);
    }

    Convolution convolution_layer_1(&cuda_helper,
                                    batch_size,
                                    num_channels,
                                    num_channels,
                                    height,
                                    width);
    convolution_layer_1.forward(x, y_1, w);

    Convolution convolution_layer_2(&cuda_helper,
                                    batch_size,
                                    num_channels,
                                    num_channels,
                                    height,
                                    width);
    convolution_layer_2.forward(x, y_2, w);

    double abs_diff = abs_difference(y_1, y_2, x_size);
    std::cout << "Convolution" << std::endl;
    std::cout << "Absolute difference " << abs_diff << std::endl;
}


int main() {
    dropout();

    convolution();
}
