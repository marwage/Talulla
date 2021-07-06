// Copyright 2020 Marcel Wagenländer

#include "convolution.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>


Convolution::Convolution(CudaHelper *helper,
                         long batch_size,
                         long num_output_channels,
                         long num_input_channels,
                         long height,
                         long width) {
    set(helper,
        batch_size,
        num_output_channels,
        num_input_channels,
        height,
        width);
}


void Convolution::set(CudaHelper *helper,
                      long batch_size,
                      long num_output_channels,
                      long num_input_channels,
                      long height,
                      long width) {
    cuda_helper_ = helper;
    batch_size_ = batch_size;
    num_output_channels_ = num_output_channels;
    num_input_channels_ = num_input_channels;
    height_ = height;
    width_ = width;
}

void Convolution::forward(float *x, float *y, float *w) {
    forward(x, y, w, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
}

void Convolution::forward(float *x, float *y, float *w, cudnnConvolutionFwdAlgo_t algo) {
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size_,
                                           num_input_channels_,
                                           height_,
                                           width_));
    long x_size = batch_size_ * num_input_channels_ * height_ * width_;
    void *d_x;
    check_cuda(cudaMalloc(&d_x,
                          x_size * sizeof(float)));
    check_cuda(cudaMemcpy(d_x,
                          x,
                          x_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    long filter_height = 3;
    long filter_width = 3;
    cudnnFilterDescriptor_t w_desc;
    check_cudnn(cudnnCreateFilterDescriptor(&w_desc));
    check_cudnn(cudnnSetFilter4dDescriptor(w_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           num_output_channels_,
                                           num_input_channels_,
                                           filter_height,
                                           filter_width));
    long filter_size = num_output_channels_ * num_input_channels_ * filter_height * filter_width;
    void *d_w;
    check_cuda(cudaMalloc(&d_w, filter_size * sizeof(float)));
    check_cuda(cudaMemcpy(d_w,
                          w,
                          filter_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudnnConvolutionDescriptor_t conv_desc;
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc));
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                1,
                                                1,
                                                1,
                                                1,
                                                1,
                                                1,
                                                CUDNN_CONVOLUTION,
                                                CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size_,
                                           num_output_channels_,
                                           height_,
                                           width_));
    long y_size = batch_size_ * num_output_channels_ * height_ * width_;
    void *d_y;
    check_cuda(cudaMalloc(&d_y,
                          y_size * sizeof(float)));
    check_cuda(cudaMemcpy(d_y,
                          y,
                          y_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    size_t workspace_size;
    check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(cuda_helper_->cudnn_handle,
                                                        x_desc,
                                                        w_desc,
                                                        conv_desc,
                                                        y_desc,
                                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                        &workspace_size));
    void *d_workspace;
    check_cuda(cudaMalloc(&d_workspace, workspace_size * sizeof(float)));

    float alpha = 1.0;
    float beta = 1.0;
    check_cudnn(cudnnConvolutionForward(cuda_helper_->cudnn_handle,
                                        &alpha,
                                        x_desc,
                                        d_x,
                                        w_desc,
                                        d_w,
                                        conv_desc,
                                        algo,
                                        d_workspace,
                                        workspace_size,
                                        &beta,
                                        y_desc,
                                        d_y));

    check_cuda(cudaMemcpy(y, d_y, y_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_w));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_workspace));
}

void Convolution::backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients) {
    throw "Not implemented";
}
