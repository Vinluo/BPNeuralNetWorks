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
			
		double learning_rate;

		double accuracy = 0;

		//反向传播误差插值
		std::vector<cv::Mat> delta_err;

		cv::Mat output_error;

		cv::Mat target;

		std::vector<double> loss_vec;

		float fine_tune_factor = 1.01f;

		float loss;
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


		//Train,use loss_threshold
		void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void test(cv::Mat &input, cv::Mat &target_);

		//Predict,just one sample
		int predict_one(cv::Mat &input);

		//Predict,more  than one samples
		std::vector<int> predict(cv::Mat &input);

		//Save model;
		void save(std::string filename);

		//Load model;
		void load(std::string filename);

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
