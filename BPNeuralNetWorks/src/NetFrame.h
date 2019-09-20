#ifndef NETFrame_H
#define NETFrame_H

#endif // !NETFrame_H


#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>


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
		
		//学习率n
		double learning_rate;

		//准确性
		double accuracy = 0;

		//反向传播误差插值
		std::vector<cv::Mat> delta_err;
		
		//误差输出
		cv::Mat output_error;

		//理想目标输出
		cv::Mat target;

		//loss 容器，
		std::vector<double> loss_vec;

		cv::Mat board;

		//
		float fine_tune_factor = 1.01f;

		//loss
		float loss;

		int output_interval = 10;


	public:
		NetFrame();


		//Initialize net:genetate weights matrices、layer matrices and bias matrices
		//initNet()：用来初始化神经网络
		void initNet(std::vector<int> Layer_neuron_num_);


		//初始化权值矩阵，调用initWeight()函数
		void initWeights(int type = 0,double a=0., double b =0.1);


		//初始化偏置项
		void initBias(cv::Scalar Bias_);


		//前向运算，包括线性运算和非线性激活，同时计算误差
		void forward();


		//反向传播，调用updateWeights()函数更新权值
		void backward();


		//Train,use loss_threshold
		void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void test(cv::Mat &input, cv::Mat &target_);


		int predict_one(cv::Mat &input);

		//多样本组预测
		//Predict,more  than one samples
		std::vector<int> predict(cv::Mat &input);

		//模型保存
		//Save model;
		void save(std::string filename);

		//Load model;
		//读取模型数据
		void load(std::string filename);

	protected:
			
		//initialize the weight matrix.if type =0,Gaussian.else uniform.
		void initWeight(cv::Mat &dst, int type, double a, double b);

		//Activation function
		cv::Mat activationFunction(cv::Mat &x, std::string func_type);

		//Compute delta error
		void deltaError();

		//Update weights
		void updateWeights();

	};

	//非成员函数设计

	//绘制损失曲线
	// Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points);

	//数据读取方法
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);


}