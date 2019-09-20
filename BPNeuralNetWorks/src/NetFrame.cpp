#include "NetFrame.h"
#include "Function\Function.h"

BP::NetFrame::NetFrame()
{

}

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
void BP::NetFrame::initBias(cv::Scalar Bias_)
{

	for (int i = 0; i < Bias.size(); i++)
	{
		Bias[i] = Bias_;
	}
}


//���紫���������
//http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
//ǰ������
void BP::NetFrame::forward()
{
	for (int i = 0; i < Layer_neuron_num.size()-1; i++)
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
	calcLoss(Layer[Layer.size() - 1], target, output_error, loss);
	deltaError();
	updateWeights();
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
		delta_err[i].create(Layer[i + 1].size(), Layer[i + 1].type());
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
			cv::Mat weight_t = Weights[i].t();
			cv::Mat delta_err_1 = delta_err[i];
			delta_err[i] = dx.mul((Weights[i + 1]).t()*delta_err[i + 1]);
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


//ѵ������
//ѵ��������õ���Ȩ�ؾ������Ҫ�õ���ѵ��ģ��
void BP::NetFrame::train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve /*= false*/)
{

	if (input.empty())
	{
		std::cout << "Input is empty!" << std::endl;
		return;
	}
	
	std::cout << "Train,Begin !" << std::endl;

	cv::Mat sample;

	//����һ����������һ�����о�����Ϊ���룬Ҳ��������ĵ�һ�㣻
	if (input.rows == (Layer[0].rows) && input.cols == 1)
	{
		//����ǰ�򴫲���Ҳ��forward()�����������顣Ȼ�����loss��
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
		double batch_loss = loss_threshold + 0.01f;
		int epoch = 0;
		//���lossֵС���趨����ֵloss_threshold������з��򴫲�������ֵ��
		//�ظ����Ϲ���ֱ��lossС�ڵ����趨����ֵ��
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
				//ÿһ�ٲ��ѧϰ�ʽ�������
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



//test()��Ŀ�Ľ�train����ִ�к�õ�ģ�ͣ�ͨ��δͨ��ģ��ѵ�������ݽ���һ�β���
//ͨ��������̲��Խ����ʵ�ʽ�����бȶԣ��Ӷ����ж�����ģ�͵���ȷ
//test()�Ĳ���������¼�����
//
//
//
//

void BP::NetFrame::test(cv::Mat &input, cv::Mat &target_)
{
	if (input.empty())
	{
		std::cout << "Input is empty!" << std::endl;
		return;
	}
	std::cout << std::endl << "Predict,begin!" << std::endl;
	//��һ�������������������
	//����ģ��Ԥ��
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
	if (input.rows == (Layer[0].rows) && input.cols == 1)
	{
		Layer[0] = input;
		forward();

		cv::Mat Layer_out = Layer[Layer.size() - 1];
		cv::Point predict_maxLoc;

		minMaxLoc(Layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
		return predict_maxLoc.y;
	}
	else
	{
		return -1;
	}
}


std::vector<int> BP::NetFrame::predict(cv::Mat &input)
{
	std::vector<int> predicted_labels;
	if (input.rows == (Layer[0].rows) && input.cols > 1)
	{
		for (int i = 0; i < input.cols; ++i)
		{
			cv::Mat sample = input.col(i);
			int predicted_label = predict_one(sample);
			predicted_labels.push_back(predicted_label);
		}
	}
	return predicted_labels;
}


void BP::NetFrame::save(std::string filename)
{
	cv::FileStorage model(filename, cv::FileStorage::WRITE);
	model << "layer_neuron_num" << Layer_neuron_num;
	model << "learning_rate" << learning_rate;
	model << "activation_function" << activation_function;

	for (int i = 0; i < Weights.size(); i++)
	{
		std::string weight_name = "weight_" + std::to_string(i);
		model << weight_name << Weights[i];
	}
	model.release();
}


void BP::NetFrame::load(std::string filename)
{
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	cv::Mat input_, target_;

	fs["layer_neuron_num"] >> Layer_neuron_num;
	initNet(Layer_neuron_num);

	for (int i = 0; i < Weights.size(); i++)
	{
		std::string weight_name = "weight_" + std::to_string(i);
		fs[weight_name] >> Weights[i];
	}

	fs["learning_rate"] >> learning_rate;
	fs["activation_function"] >> activation_function;

	fs.release();
}


void BP::draw_curve(cv::Mat& board, std::vector<double> points)
{
	cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
	board = board_;
	cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
	cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

	for (size_t i = 0; i < points.size() - 1; i++)
	{
		cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
		cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
		cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
		if (i >= 1000)
		{
			return;
		}
	}
	cv::imshow("Loss", board);
	cv::waitKey(1);
}


void BP::get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start /*= 0*/)
{
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	cv::Mat input_, target_;
	fs["input"] >> input_;
	fs["target"] >> target_;
	fs.release();
	input = input_(cv::Rect(start, 0, sample_num, input_.rows));
	label = target_(cv::Rect(start, 0, sample_num, target_.rows));
}

