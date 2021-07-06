// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>


class Dropout {
protected:
    CudaHelper *cuda_helper_ = NULL;
    float probability_;
    unsigned long long seed_;
    size_t state_size_;
    char *reserve_space_ = NULL;
    size_t reserve_space_size_;

public:
    Dropout(CudaHelper *helper, float probability, unsigned long long seed);
    ~Dropout();
    void set(CudaHelper *helper, float probability, unsigned long long seed);
    void forward(Matrix<float> *x, Matrix<float> *y);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients);
};

#endif
