// Copyright 2020 Marcel Wagenländer

#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>


class Convolution {
protected:
    CudaHelper *cuda_helper_ = NULL;
    long batch_size_;
    long num_output_channels_;
    long num_input_channels_;
    long height_;
    long width_;

public:
    Convolution(CudaHelper *helper,
                long batch_size,
                long num_output_channels,
                long num_input_channels,
                long height,
                long width);
    void set(CudaHelper *helper,
             long batch_size,
             long num_output_channels,
             long num_input_channels,
             long height,
             long width);
    void forward(float *x, float *y, float *w);
    void forward(float *x, float *y, float *w, cudnnConvolutionFwdAlgo_t algo);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients);
};

#endif
