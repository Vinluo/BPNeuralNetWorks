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

		//ÿ����Ԫ��
		std::vector<int> Layer_neuron_num;

		//��
		std::vector<cv::Mat> Layer;

		//Ȩ��
		std::vector<cv::Mat> Weights;

		//ƫ����
		std::vector<cv::Mat> Bias;

		//���ʽ
		std::string activation_function = "Sigmoid";


	public:
		NetFrame();
		~NetFrame();


		//Initialize net:genetate weights matrices��layer matrices and bias matrices
		//initNet()��������ʼ��������
		void initNet(std::vector<int> Layer_neuron_num_);


		//��ʼ��Ȩֵ���󣬵���initWeight()����
		void initWeights(int type = 0,double a=0., double b =0.1);


		//��ʼ��ƫ����
		void initBias(cv::Scalar& Bias_);


		//ǰ�����㣬������������ͷ����Լ��ͬʱ�������
		void forward();


		//���򴫲�������updateWeights()��������Ȩֵ
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
