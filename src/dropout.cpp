// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>


Dropout::Dropout(CudaHelper *helper, float probability, unsigned long long seed) {
    set(helper, probability, seed);
}

Dropout::~Dropout() {
    check_cuda(cudaFreeHost(reserve_space_));
}

void Dropout::set(CudaHelper *helper, float probability, unsigned long long seed) {
    cuda_helper_ = helper;
    probability_ = probability;
    seed_ = seed;

    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
}

void Dropout::forward(Matrix<float> *x, Matrix<float> *y) {
    to_row_major_inplace(x);

    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));

    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y->num_rows_, 1, 1, y->num_columns_));

    void *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    void *d_y;
    check_cuda(cudaMalloc(&d_y, y->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));

    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc, x_descr, d_x,
                                    y_descr, d_y,
                                    d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(y->values_, d_y, y->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y->is_row_major_ = true;

    if (reserve_space_ == NULL) {
        check_cuda(cudaMallocHost(&reserve_space_, reserve_space_size_));
    }
    check_cuda(cudaMemcpy(reserve_space_, d_reserve_space,
                          reserve_space_size_,
                          cudaMemcpyDeviceToHost));

    // free
    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_reserve_space));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));
}

void Dropout::backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients) {
    if (y->num_rows_ != incoming_gradients->num_rows_ || y->num_columns_ != incoming_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    to_row_major_inplace(incoming_gradients);

    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));

    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_,
                          incoming_gradients->size_,
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, incoming_gradients->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cuda(cudaMemcpy(d_reserve_space, reserve_space_,
                          reserve_space_size_,
                          cudaMemcpyHostToDevice));

    // It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle,
                                     dropout_desc,
                                     dy_desc, d_dy,
                                     dx_desc, d_dx,
                                     d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(gradients->values_, d_dx,
                          gradients->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free
    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_reserve_space));
}
