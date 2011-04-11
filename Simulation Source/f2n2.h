#ifndef F2N2
#define F2N2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

class nn {
	public:
	
		int backpropagation; /* Whether backpropagation is going to be used to train this network */
		int lmax; /* Maximum number of neurons in a layer */
		int layer_count; /* Total number of layers */
		int *layer_structure; /* The nth element of the array represents the number of neurons in the nth layer */
		float learning_rate; /* Learning rate in backpropagation */
		float momentum; /* Momentum in learning rate */
		float ***weights; /* Neural net weights */
		float ***weight_change; /* Stores weight change (for momentum) */
		float **bias; /* Neural net bias */
		float **layers; /* Neuron values */
		
		
	//public:
		//int backpropagation;
		int change_learning_rate(float var);
		float fitness; /* (Neuroevolution only) fitness of neural network */
		int set_weight(int layer_no, int index_from, int index_to, float weight);
		nn();
		//nn(int layer_count, int *layer_structure, int use_backpropagation = 1, float learning_rate = 0.1, float momentum = 0.9, int max_nodes = 100);
		void create_nn(int layer_count, int *layer_structure, int use_backpropagation = 1, float learning_rate = 0.1, float momentum = 0.9, int max_nodes = 100);
		~nn();
		float *calculate(float *inputs); /* Returns values of output neurons given input neuron values */
		int calculate_layer_output(float *inputs, int layer_no, float *output);
		int calculate_max_output_id(float *inputs); /* Calculate neuron values and return ID of largest output neuron */
		void mutate(float std_deviation = 1.0);
		float get_error(float *desired_output); /* Calculate mean-square error */
		float get_error_converted(float *desired_output); /* Calculate mean-square error */
		void backpropagate(float *desired_output);
		void save(char *filename, int algo_index = -1);/* algo index 0 for dnc, 1 for marchand .Save neural network weights and biases to a file so it can be loaded later or by another program using F2N2 */
		int load(char *filename); /* Load neural network weights and biases from a file created by the save function of F2N2 */
		int get_max_nodes(); /* return max nodes, this is a user defined parameter */
		int get_l_neurons(int layer); /* returns number of neurons for a specific layer */
		
		void add_hidden_node(int number = 1,int  layer_number = 1); /* add new hidden node, helpful for dynamic network creatioin */
		void sub_hidden_node(int number = 1, int layer_number = 1);
		float calculate_for_single(float *input, float *desired); /* Calculate error for single input instance */
		void get_layer_data();
		//void change_random_weight(int layer_number, int node_number);
		void change_random_weight(int layer_number, int node_number);
};

#endif
