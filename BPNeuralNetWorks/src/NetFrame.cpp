#include "NetFrame.h"
#include "Function\Function.h"

void BP::NetFrame::initNet(std::vector<int> Layer_neuron_num_)
{
	Layer_neuron_num = Layer_neuron_num_;
	
	//����ÿ��
	Layer.resize(Layer_neuron_num.size());
	for (int i = 0; i < Layer.size(); i++)
	{
		Layer[i].create(Layer_neuron_num[i], 1, CV_32FC1);
	}
	std::cout << "Generate layers, successfully!" << std::endl;

	//����Ȩ�غ�ƫ����
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


//ƫ�ó�ʼ���Ǹ����е�ƫ�ø���ͬ��ֵ��������Scalar������������ֵ��
void BP::NetFrame::initBias(cv::Scalar& Bias_)
{

	for (int i = 0; i < Bias.size(); i++)
	{
		Bias[i] = Bias_;
	}
}


//ǰ������
void BP::NetFrame::forward()
{
	for (int i = 0; i < Layer_neuron_num.size(); i++)
	{
		//�������������Y = WX + b
		cv::Mat Product = Weights[i] * Layer[i] + Bias[i];
		//�����Լ����
		Layer[i + 1] = activationFunction(Product, activation_function);
	}
}


//���򴫲�����
void BP::NetFrame::backward()
{


}


//����type���ֲ����������ò�ͬ��Ȩ�ر�׼�����˹�ֲ����ȷֲ�����Ҫ��������
void BP::NetFrame::initWeight(cv::Mat &dst, int type, double a, double b)
{
	//Ȩ�ؾ�����̬�ֲ�
	if (type == 0)
	{
		randn(dst, a, b);
	}
	//Ȩ�ظ�˹�ֲ�
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

