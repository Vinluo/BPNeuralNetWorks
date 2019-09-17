#include "NetFrame.h"
#include "Function\Function.h"

void BP::NetFrame::initNet(std::vector<int> Layer_neuron_num_)
{
	Layer_neuron_num = Layer_neuron_num_;
	
	//生成每层
	Layer.resize(Layer_neuron_num.size());
	for (int i = 0; i < Layer.size(); i++)
	{
		Layer[i].create(Layer_neuron_num[i], 1, CV_32FC1);
	}
	std::cout << "Generate layers, successfully!" << std::endl;

	//生成权重和偏重项
	Weights.resize(Layer.size() - 1);
	Bias.resize(Layer.size() - 1);
	for (int i=0;i<(Layer.size()-1);i++)
	{
		Weights[i].create(Layer[i + 1].rows, Layer[i].rows, CV_32FC1);

		Bias[i] = cv::Mat::zeros(Layer[i + 1].rows, 1, CV_32FC1);
	}
	
	std::cout << "Generate Weights Matrices and Bias,successfully!" << std::endl;
	std::cout << "Init Net,done" << std::endl;
}


void BP::NetFrame::initWeights(int type /*= 0*/, double a/*=0.*/, double b /*=0.1*/)
{
	for (int i = 0; i < Weights.size(); ++i)
	{
		initWeight(Weights[i], 0, 0., 0.1);
	}
}


//偏置初始化是给所有的偏置赋相同的值。这里用Scalar对象来给矩阵赋值。
void BP::NetFrame::initBias(cv::Scalar& Bias_)
{

	for (int i = 0; i < Bias.size(); i++)
	{
		Bias[i] = Bias_;
	}
}


//前向运算
void BP::NetFrame::forward()
{
	for (int i = 0; i < Layer_neuron_num.size(); i++)
	{
		//线型运算可以用Y = WX + b
		cv::Mat Product = Weights[i] * Layer[i] + Bias[i];
		//非线性激活函数
		Layer[i + 1] = activationFunction(Product, activation_function);
	}
}


//反向传播运算
void BP::NetFrame::backward()
{


}


//这里type区分参数数量，用不同的权重标准，如高斯分布均匀分布就需要两个参数
void BP::NetFrame::initWeight(cv::Mat &dst, int type, double a, double b)
{
	//权重均匀正态分布
	if (type == 0)
	{
		randn(dst, a, b);
	}
	//权重高斯分布
	else
	{
		randu(dst, a, b);
	}
}


cv::Mat BP::NetFrame::activationFunction(cv::Mat &x, std::string func_type)
{

	activation_function = func_type;
	cv::Mat fx;
	if (func_type == "Sigmoid")
	{
		fx = Sigmoid(x);
	}
	if (func_type == "Tanh")
	{
		fx = Tanh(x);
	}
	if (func_type == "ReLU")
	{
		fx = ReLU(x);
	}
	if (func_type == "LeakyReLU")
	{
		//deafult a =0.2
		fx = LeakyRelu(x);
	}
	return fx;

}


void BP::NetFrame::deltaError()
{

}


void BP::NetFrame::updateWeights()
{

}

