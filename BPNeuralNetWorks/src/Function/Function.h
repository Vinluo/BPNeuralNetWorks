#pragma once
#include<iostream>
#include<opencv2/core/core.hpp>

static float LRL_a = 0.2f;


//sigmoid function
cv::Mat Sigmoid(cv::Mat &x);

//Tanh function
cv::Mat Tanh(cv::Mat &x);

//ReLU function
cv::Mat ReLU(cv::Mat &x);

cv::Mat LeakyRelu(cv::Mat &x, float a = 0.2f);

//Derivative function
cv::Mat DerivativeFunction(cv::Mat& fx, std::string func_type);

//Objective function
void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss);