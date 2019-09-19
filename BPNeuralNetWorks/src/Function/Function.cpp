#include "Function.h"


cv::Mat Sigmoid(cv::Mat &x)
{
	cv::Mat exp_x, Sig;
	cv::exp(-x, exp_x);
	Sig = 1 / (1 + exp_x);
	return Sig;
}


cv::Mat Tanh(cv::Mat &x)
{
	cv::Mat exp_x, Th;
	cv::exp(-2*x, exp_x);
	Th = (1 - exp_x) / (1 + exp_x);
	return Th;
}


cv::Mat ReLU(cv::Mat &x)
{
	cv::Mat RL = x;
	for (int i = 0; i < RL.rows; i++)
	{
		for (int j = 0; j < RL.cols; j++)
		{
			if (RL.at<float>(i, j) < 0)
			{
				RL.at<float>(i, j) = 0;
			}
		}
	}
	return RL;
}


cv::Mat LeakyRelu(cv::Mat &x,const float a /*= 0.2f*/)
{
	LRL_a = a;
	cv::Mat LRL = x;
	for (int i = 0; i < LRL.rows; i++)
	{
		for (int j = 0; j < LRL.cols; j++)
		{
			if (LRL.at<float>(i, j) < 0)
			{
				LRL.at<float>(i, j) *=a ;
			}
		}
	}
	return LRL;
}


cv::Mat DerivativeFunction(cv::Mat& fx, std::string func_type)
{
	cv::Mat dx;
	if (func_type == "Sigmoid")
	{
		dx = Sigmoid(fx).mul((1 - Sigmoid(fx)));
	}
	if (func_type == "Tanh")
	{
		cv::Mat tanh_2;
		pow(Tanh(fx), 2., tanh_2);
		dx = 1 - tanh_2;
	}
	if (func_type == "ReLU")
	{
		dx = fx;
		for (int i = 0; i < fx.rows; i++)
		{
			for (int j = 0; j < fx.cols; j++)
			{
				if (fx.at<float>(i, j) > 0)
				{
					dx.at<float>(i, j) = 1;
				}
			}
		}
	}
	if (func_type == "LeakyRelu")
	{
		dx = fx;

		for (int i = 0; i < fx.rows; i++)
		{
			for (int j = 0; j < fx.cols; j++)
			{
				if (fx.at<float>(i, j) > 0)
				{
					dx.at<float>(i, j) = 1;
				}
				else
				{
					dx.at<float>(i, j) = LRL_a;
				}
			}
		}
	}

	return dx;
}


void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss)
{
	if (target.empty())
	{
		std::cout << "Can't find the target cv::Matrix" << std::endl;
		return;
	}

	output_error = target - output;
	cv::Mat err_sqrare;
	pow(output_error, 2., err_sqrare);
	cv::Scalar err_sqr_sum = sum(err_sqrare);
	loss = err_sqr_sum[0] / (float)(output.rows);
}
