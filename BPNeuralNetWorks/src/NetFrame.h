#ifndef NETFrame_H
#define NETFrame_H

#endif // !NETFrame_H


#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>



namespace BP 
{

	class NetFrame
	{
	public:

		//每层神经元数
		std::vector<int> Layer_neuron_num;

		//层
		std::vector<cv::Mat> Layer;

		//权重
		std::vector<cv::Mat> Weights;

		//偏置项
		std::vector<cv::Mat> Bias;

		//激活公式
		std::string activation_function = "Sigmoid";


	public:
		NetFrame();
		~NetFrame();


		//Initialize net:genetate weights matrices、layer matrices and bias matrices
		//initNet()：用来初始化神经网络
		void initNet(std::vector<int> Layer_neuron_num_);


		//初始化权值矩阵，调用initWeight()函数
		void initWeights(int type = 0,double a=0., double b =0.1);


		//初始化偏置项
		void initBias(cv::Scalar& Bias_);


		//前向运算，包括线性运算和非线性激活，同时计算误差
		void forward();


		//反向传播，调用updateWeights()函数更新权值
		void backward();


	protected:
			
		//initialise the weight matrix.if type =0,Gaussian.else uniform.
		void initWeight(cv::Mat &dst, int type, double a, double b);

		//Activation function
		cv::Mat activationFunction(cv::Mat &x, std::string func_type);

		//Compute delta error
		void deltaError();

		//Update weights
		void updateWeights();

	};
}
