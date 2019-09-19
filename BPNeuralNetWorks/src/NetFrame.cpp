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


//网络传播计算过程
//http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
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
	calcLoss(Layer[Layer.size() - 1], target, output_error, loss);
	deltaError();
	updateWeights();
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
		//default a =0.2
		fx = LeakyRelu(x);
	}
	return fx;

}


void BP::NetFrame::deltaError()
{
	delta_err.resize(Layer.size() - 1);
	for (int i = delta_err.size() - 1; i >= 0; i--)
	{
		delta_err[i].create(Layer[i + 1].size(), Layer[t + 1].type());
		//cv::Mat dx = layer[i+1].mul(1 - layer[i+1]);
		cv::Mat dx = DerivativeFunction(Layer[i + 1], activation_function);
		//Output layer delta error
		if (i == delta_err.size() - 1)
		{
			delta_err[i] = dx.mul(output_error);
		}
		else
		{
			cv::Mat	weight = Weights[i];
			cv::Mat weight_t = weight[i].t();
			cv::Mat delta_err_1 = delta_err[i];
			delta_err[i] = dx.mul((Weights[i + 1]).t()*delta_err[i + 1]));
		}
	}
}


void BP::NetFrame::updateWeights()
{
	for (int i = 0; i < Weights.size(); i++)
	{
		cv::Mat delta_weights = learning_rate * (delta_err[i] * Layer[i].t());
		Weights[i] = Weights[i] + delta_weights;	
	}
}


//训练流程
void BP::NetFrame::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve /*= false*/)
{

	if (input.empty())
	{
		std::cout << "Input is empty!" << std::endl;
		return;
	}
	
	std::cout << "Train,Begin !" << std::endl;

	cv::Mat sample;

	//接受一个样本（即一个单列矩阵）作为输入，也即神经网络的第一层；
	if (input.rows == (Layer[0].rows) && input.cols == 1)
	{
		//进行前向传播，也即forward()函数做的事情。然后计算loss；
		target = target_;
		sample = input;
		Layer[0] = sample;
		forward();
		//backward();
		int num_of_train = 0;
		while (loss > loss_threshold)
		{
			backward();
			forward();
			num_of_train++;
			if (num_of_train % 500 == 0)
			{
				std::cout << "Train " << num_of_train << " times" << std::endl;
				std::cout << "Loss: " << loss << std::endl;
			}
		}
		std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
		std::cout << "Loss: " << loss << std::endl;
		std::cout << "Train successfully!" << std::endl;
	}
	else if (input.rows == (Layer[0].rows) && input.cols > 1)
	{
		double batch_loss = loss_threshold + 0.01;
		int epoch = 0;
		//如果loss值小于设定的阈值loss_threshold，则进行反向传播更新阈值；
		//重复以上过程直到loss小于等于设定的阈值。
		while (batch_loss > loss_threshold)
		{
			batch_loss = 0.;
			for (int i = 0; i < input.cols; ++i)
			{
				target = target_.col(i);
				sample = input.col(i);
				Layer[0] = sample;

				forward();
				backward();

				batch_loss += loss;
			}

			loss_vec.push_back(batch_loss);

			if (loss_vec.size() >= 2 && draw_loss_curve)
			{
				draw_curve(board, loss_vec);
			}
			epoch++;
			if (epoch % output_interval == 0)
			{
				std::cout << "Number of epoch: " << epoch << std::endl;
				std::cout << "Loss sum: " << batch_loss << std::endl;
			}
			if (epoch % 100 == 0)
			{
				learning_rate *= fine_tune_factor;
			}
		}
		std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
		std::cout << "Loss sum: " << batch_loss << std::endl;
		std::cout << "Train successfully !" << std::endl;
	}
	else
	{
		std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
	}
}


void BP::NetFrame::test(cv::Mat &input, cv::Mat &target_)
{
	if (input.empty())
	{
		std::cout << "Input is empty!" << std::endl;
		return;
	}
	std::cout << std::endl << "Predict,begin!" << std::endl;

	if (input.rows == (Layer[0].rows) && input.cols == 1)
	{
		int predict_number = predict_one(input);

		cv::Point target_maxLoc;
		minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
		int target_number = target_maxLoc.y;

		std::cout << "Predict: " << predict_number << std::endl;
		std::cout << "Target:  " << target_number << std::endl;
		std::cout << "Loss: " << loss << std::endl;
	}
	else if (input.rows == (Layer[0].rows) && input.cols > 1)
	{
		double loss_sum = 0;
		int right_num = 0;
		cv::Mat sample;
		for (int i = 0; i < input.cols; ++i)
		{
			sample = input.col(i);
			int predict_number = predict_one(sample);
			loss_sum += loss;

			target = target_.col(i);
			cv::Point target_maxLoc;
			minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
			int target_number = target_maxLoc.y;

			std::cout << "Test sample: " << i << "   " << "Predict: " << predict_number << std::endl;
			std::cout << "Test sample: " << i << "   " << "Target:  " << target_number << std::endl << std::endl;
			if (predict_number == target_number)
			{
				right_num++;
			}
		}
		accuracy = (double)right_num / input.cols;
		std::cout << "Loss sum: " << loss_sum << std::endl;
		std::cout << "accuracy: " << accuracy << std::endl;
	}
	else
	{
		std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
		return;
	}
}


int BP::NetFrame::predict_one(cv::Mat &input)
{

}


std::vector<int> BP::NetFrame::predict(cv::Mat &input)
{

}


void BP::NetFrame::save(std::string filename)
{

}


void BP::NetFrame::load(std::string filename)
{

}

