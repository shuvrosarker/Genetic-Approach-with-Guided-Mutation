#include "f2n2.h"
#include <iostream>
using namespace std;

/* Constructor */
nn::nn()
{
	
}

void nn::create_nn(int layer_count, int *layer_structure, int use_backpropagation, float learning_rate, float momentum , int max_nodes)
{
	int i, j, k;
	float x;
	
	/* Initialize neural network structure and settings */
	this->layer_count = layer_count;
	this->layer_structure = new int[this->layer_count];
	
	for(int i = 0; i < this->layer_count; i++) this->layer_structure[i] = layer_structure[i];
	
	this->learning_rate = learning_rate;
	this->momentum = momentum;
	this->backpropagation = use_backpropagation;
	fitness = 0.0;
	
	lmax = max_nodes;
	
	/* Allocate enough memory */
	weights = (float ***)malloc(sizeof(float **) * layer_count);
	if (backpropagation) weight_change = (float ***)malloc(sizeof(float **) * layer_count);
	bias = (float **)malloc(sizeof(float *) * layer_count);
	layers = (float **)malloc(sizeof(float *) * layer_count);
	for(i=0; i<layer_count; i++) {
		weights[i] = (float **)malloc(sizeof(float *) * lmax);
		if (backpropagation) weight_change[i] = (float **)malloc(sizeof(float *) * lmax);
		bias[i] = (float *)malloc(sizeof(float) * lmax);
		layers[i] = (float *)malloc(sizeof(float) * lmax);
		for(j=0; j<lmax; j++) {
			weights[i][j] = (float *)malloc(sizeof(float) * lmax);
			if (backpropagation) weight_change[i][j] = (float *)malloc(sizeof(float) * lmax);
		}
	}
	
	/* Randomly weight neural network */
	for(i=0; i<layer_count; i++) {
		for(j=0; j<lmax; j++) {
			for(k=0; k<lmax; k++) {
				/* Values of x should fall between -1 and 1 */
				x = (float)drand48() * 2 - 1;
				weights[i][j][k] = x;
				if (backpropagation) weight_change[i][j][k] = 0;
			}
			x = 2.0 * (float)drand48() - 1.0;
			bias[i][j] = x;
		}
	}
	
}

/* Destructor */
/* Free memory allocated by neural network to store weights, biases, layer state information and such */
nn::~nn() {
	int i, j;
	
	for(i=0; i<layer_count; i++) {
		for(j=0; j<lmax; j++) {
			free(weights[i][j]);
			if (backpropagation) free(weight_change[i][j]);
		}
		free(weights[i]);
		if (backpropagation) free(weight_change[i]);
		free(bias[i]);
		free(layers[i]);
	}
	free(weights);
	//cout << "Backpropagation value : " << backpropagation << endl;
	if (backpropagation) free(weight_change);
	free(bias);
	free(layers);
	delete [] layer_structure;
}

/* add new hidden node, helpful for dynamic network creatioin */
void nn::add_hidden_node(int number, int layer_number)
{
	this->layer_structure[layer_number] += number;
	
}

void nn::sub_hidden_node(int number, int layer_number)
{
	this->layer_structure[layer_number] -= number;
	
}

int nn::get_max_nodes()
{
	return lmax;
}

int nn:: change_learning_rate(float lr)
{
	learning_rate = lr;
	return 0;
}

int nn::get_l_neurons(int layer)
{
	return this->layer_structure[layer];
}

/* both layer_number and node_number starts from zero */
void nn:: change_random_weight(int layer_number, int node_number)
{
	int loop = this->layer_structure[layer_number - 1]; // number of neurons in the layer previous to layer_number
	for( int i = 0; i < loop; i++)
	{
		//srand(time(NULL));
		float var = (float)drand48() * 2.0 - 1.0;
		//cout << "random value is  : " << var << endl;
		//cout << "before change random " << endl;
		weights[layer_number - 1][i][node_number] = var; // set random weights from node_number to all inputs -> nth hidden neurons.
		//cout << "after change random" << endl;
	}
}

int nn:: set_weight(int layer_no, int index_from, int index_to, float weight)
{
	weights[layer_no][index_from][index_to] = weight;
	return 0;
}

/* Calculate neuron values */
/* Returns values of output neurons given input neuron values */
float *nn::calculate(float *inputs) {
	int i, j, l;
	
	/* Load input neurons */
	for(i=0; i<layer_structure[0]; i++) {
		layers[0][i] = inputs[i];
	}
	
	/* Layered processing */
	for(l=1; l<layer_count; l++) {
		for(i=0; i<layer_structure[l]; i++) {
			layers[l][i] = bias[l-1][i];
			for(j=0; j<layer_structure[l-1]; j++) {
				layers[l][i] += layers[l-1][j] * weights[l-1][j][i];
			}
			layers[l][i] = 1.0 / (1.0 + exp(-layers[l][i]));
		}
	}
	
	return layers[layer_count-1];
}

/*watchout this function defination*/
/*layer_no starts from 0. that means 0 is the input layer, 1 first hidden layer*/
int nn:: calculate_layer_output(float *inputs, int layer_no, float *output)
{
	int i, j, l;
	
	/* Load input neurons */
	for(i=0; i<layer_structure[0]; i++) {
		layers[0][i] = inputs[i];
	}
	
	/* Layered processing */
	for(l=1; l <= layer_no; l++) {
		for(i=0; i<layer_structure[l]; i++) {
			layers[l][i] = bias[l-1][i];
			for(j=0; j<layer_structure[l-1]; j++) {
				layers[l][i] += layers[l-1][j] * weights[l-1][j][i];
			}
			layers[l][i] = 1.0 / (1.0 + exp(-layers[l][i]));
		}
	}
	
	int num_neurons = get_l_neurons(layer_no);
	
	for(l = 0; l < num_neurons; l++) output[l] = layers[layer_no][l];
	//return layers[layer_no];
	return 0;
	
}

/* Calculate error for single input instance */
float nn::calculate_for_single( float *inputs, float *desired)
{
	calculate(inputs);
	
	float result = get_error(desired);
	//cout << "error main class: " << result << endl;
	return result;
}

/* Calculate neuron values and return ID of largest output neuron */
int nn::calculate_max_output_id(float *inputs) {
	int i;
	int out;
	float max;
	float *output_neurons;
	
	output_neurons = calculate(inputs);
	
	max = -1000000000.0;
	out = 0;
	for(i=0; i<layer_structure[layer_count-1]; i++) {
		if (output_neurons[i] > max) {
			max = output_neurons[i];
			out = i;
		}
	}
	
	return out;
}

/* Mutate weights of each neuron and bias according to Gaussian distribution with specified standard deviation */
/* Not used in combination with backpropagation; mostly useful for neuroevolution */
void nn::mutate(float std_deviation) {
	int l, i, j;
	//float x;
	float x1, x2, y1, y2, w;
	
	for(l=layer_count-2; l>=0; l--) {
		for(i=0; i<layer_structure[l]; i++) {
			for(j=0; j<layer_structure[l+1]; j++) {
				do {
					x1 = 2.0 * (float)drand48() - 1.0;
					x2 = 2.0 * (float)drand48() - 1.0;
					w = x1 * x1 + x2 * x2;
				} while(w >= 1.0);
				w = sqrt((-2.0 * logf(w)) / w) * std_deviation;
				weights[l][i][j] += x1 * w;
			}
		}
		for(j=0; j<layer_structure[l+1]; j++) {
			do {
				x1 = 2.0 * (float)drand48() - 1.0;
				x2 = 2.0 * (float)drand48() - 1.0;
				w = x1 * x1 + x2 * x2;
			} while(w >= 1.0);
			w = sqrt((-2.0 * logf(w)) / w) * std_deviation;
			bias[i][j] += x1 * w;
		}
	}
}

/* Calculate mean-square error */
float nn::get_error(float *desired_output) {
	int l, i;
	float diff;
	float error;
	
	error = 0;
	l = this->layer_count-1;
	for(i=0; i<layer_structure[l]; i++) {
		diff = (layers[l][i] - desired_output[i]);
		error += diff * diff;
	}
	
	error /= layer_structure[l];
	return error;
}

float nn::get_error_converted(float *desired_output) {
	int l, i;
	float diff;
	float error, layer_data;
	
	error = 0;
	l = this->layer_count-1;
	for(i=0; i<layer_structure[l]; i++) {
		
		if(layers[l][i] <= 0.5) layer_data = 0.0;
		else layer_data = 1.0;
		
		diff = (layer_data - desired_output[i]);
		error += diff * diff;
	}
	error /= layer_structure[l];
	
	return error;
}

/* Train the neural network through backpropagation */
void nn::backpropagate(float *desired_output) {
	int l, i, j;
	float change;
	float sum;
	float **errors;
	
	if (!backpropagation) {
		printf("Error: tried to use backpropagation when object is not initialized with backpropagation enabled\n");
		exit(-1);
	}
	
	/* Allocate memory required to store errors */
	errors = (float **)malloc(sizeof(float *) * layer_count);
	for(i=0; i<layer_count; i++) {
		errors[i] = (float *)malloc(sizeof(float) * lmax);
	}
	
	/* Output layer errors */
	l = layer_count - 1;
	for(i=0; i<layer_structure[l]; i++) {
		errors[l][i] = (desired_output[i] - layers[l][i]) * layers[l][i] * (1 - layers[l][i]);
	}
	
	/* Hidden layer errors */
	for(l=layer_count-2; l>0; l--) {
		for(i=0; i<layer_structure[l]; i++) {
			sum = 0;
			for(j=0; j<layer_structure[l+1]; j++) {
				sum += errors[l+1][j] * weights[l][i][j];
			}
			errors[l][i] = sum * layers[l][i] * (1 - layers[l][i]);
		}
	}
	
	/* Adjust weights and biases */
	for(l=layer_count-2; l>=0; l--) {
		for(i=0; i<layer_structure[l]; i++) {
			for(j=0; j<layer_structure[l+1]; j++) {
				change = learning_rate * errors[l+1][j] * layers[l][i];
				weights[l][i][j] += change + momentum * weight_change[l][i][j];
				weight_change[l][i][j] = change;
			}
		}
		for(j=0; j<layer_structure[l+1]; j++) {
			bias[l][j] += learning_rate * errors[l+1][j];
		}
	}
	
	for(i=0; i<layer_count; i++) {
		free(errors[i]);
	}
	free(errors);
}

/* Save neural network weights and biases to a file so it can be loaded later or by another program using F2N2 */
void nn::save(char *filename, int algo_index) {
	int l, i, j;
	FILE *fp;
	
	fp = fopen(filename, "w");
	
	/* Write headers */
	fprintf(fp, "%d\n", algo_index);
	fprintf(fp, "%d %d\n", layer_count, lmax);
	
	for(l = 0; l < layer_count; l++) fprintf(fp, "%d ", layer_structure[l]);
	fprintf(fp, "\n");
	
	/* Save weights */
	//cout << "layers : " << layer_count << endl;
	for(l=0; l<(layer_count-1); l++) {
		for(i=0; i<layer_structure[l]; i++) {
			for(j=0; j<layer_structure[l+1]; j++) {
				fprintf(fp, "%.15f ",weights[l][i][j]);
			}
			fprintf(fp, "%.15f \n", bias[l][i]);
		}
		fprintf(fp,"\n");
	}
	
	fclose(fp);
}

/* Load neural network weights and biases from a file created by the save function of F2N2 */
int nn::load(char *filename) {
	int l, i, j;
	int loaded_layer_count;
	int loaded_lmax;
	char tmp[10];
	int algo_index;
	FILE *fp;
	
	puts(filename);
	fp = fopen(filename, "r");
	
	if(fp == NULL)
	{
		printf("error opening file\n");
		return 0;
	}
	
	
	/* Read headers */
	fscanf(fp, "%d", &algo_index);
	fscanf(fp, "%d %d ", &loaded_layer_count, &loaded_lmax);
	
	//puts(filename);
	
	/* Check headers for incompatible neural network structure */
	if (loaded_layer_count > layer_count || loaded_lmax > lmax) {
		printf("Error: tried to load weights of incompatible neural network from file %d %d\n",loaded_layer_count, loaded_lmax);
		exit(-1);
	}
	
	layer_count = loaded_layer_count;
	for(l = 0; l < loaded_layer_count; l++) fscanf(fp, "%d ", &layer_structure[l]);
	
	/* Load weights */
	for(l=0; l<(layer_count - 1); l++) {
		for(i=0; i<layer_structure[l]; i++) {
			for(j=0; j<layer_structure[l+1]; j++) {
				fscanf(fp, "%f ", &weights[l][i][j]);
			}
			fscanf(fp, "%f ", &bias[l][i]);
		}
	}
	
	fclose(fp);
	return algo_index;
}

void nn::get_layer_data()
{
	for(int m = 0; m < layer_count; m++) cout <<  layer_structure[m] << " : ";
	cout << endl;
}