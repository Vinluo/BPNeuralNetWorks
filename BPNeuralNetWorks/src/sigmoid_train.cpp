#include "NetFrame.h"
//<opencv2\opencv.hpp>

using namespace std;
using namespace cv;
using namespace BP;

int main(int argc, char *argv[])
{
	//Set neuron number of every layer
	vector<int> layer_neuron_num = { 784,100,10 };

	// Initialise Net and weights
	NetFrame net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);

	//Get test samples and test samples 
	Mat input, label, test_input, test_label;
	int sample_number = 800;
	get_input_label("../data/input_label_1000.xml", input, label, sample_number);
	get_input_label("../data/input_label_1000.xml", test_input, test_label, 200, 800);

	//Set loss threshold,learning rate and activation function
	float loss_threshold = 0.5;
	net.learning_rate = 0.3;
	net.output_interval = 2;
	net.activation_function = "Sigmoid";

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.train(input, label, loss_threshold, true);
	net.test(test_input, test_label);

	//Save the model
	net.save("models/model_sigmoid_800_200.xml");

	getchar();
	return 0;

}




