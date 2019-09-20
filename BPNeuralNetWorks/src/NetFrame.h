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
		
		//ѧϰ��n
		double learning_rate;

		//׼ȷ��
		double accuracy = 0;

		//���򴫲�����ֵ
		std::vector<cv::Mat> delta_err;
		
		//������
		cv::Mat output_error;

		//����Ŀ�����
		cv::Mat target;

		//loss ������
		std::vector<double> loss_vec;

		cv::Mat board;

		//
		float fine_tune_factor = 1.01f;

		//loss
		float loss;

		int output_interval = 10;


	public:
		NetFrame();


		//Initialize net:genetate weights matrices��layer matrices and bias matrices
		//initNet()��������ʼ��������
		void initNet(std::vector<int> Layer_neuron_num_);


		//��ʼ��Ȩֵ���󣬵���initWeight()����
		void initWeights(int type = 0,double a=0., double b =0.1);


		//��ʼ��ƫ����
		void initBias(cv::Scalar Bias_);


		//ǰ�����㣬������������ͷ����Լ��ͬʱ�������
		void forward();


		//���򴫲�������updateWeights()��������Ȩֵ
		void backward();


		//Train,use loss_threshold
		void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void test(cv::Mat &input, cv::Mat &target_);


		int predict_one(cv::Mat &input);

		//��������Ԥ��
		//Predict,more  than one samples
		std::vector<int> predict(cv::Mat &input);

		//ģ�ͱ���
		//Save model;
		void save(std::string filename);

		//Load model;
		//��ȡģ������
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

	//�ǳ�Ա�������

	//������ʧ����
	// Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points);

	//���ݶ�ȡ����
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);


}