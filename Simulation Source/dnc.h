/*
 *  dnc.h
 *  NeuralConstructive
 *
 *  Created by Shuvro Sarker on 1/23/11.
 *  Copyright 2011 BUET. All rights reserved.
 *
 */
 
/* 
 *These variable varies with dataset.
 */
//#ifndef F2N2
//#define F2N2
#include "f2n2.h"
//#endif

#define TRAIN_SIZE 384
#define ATTR_NUM 8
#define NUM_CLASS 2
#define FILE_NAME_SIZE 50

class dnc : public nn{
	private:
		//nn network;
		float error_limit;
		int num_validation;
		int monotone_increament;
		float inputs[TRAIN_SIZE][ATTR_NUM];
		float outputs[TRAIN_SIZE][NUM_CLASS];
		float validation_input[TRAIN_SIZE][ATTR_NUM];
		float validation_output[TRAIN_SIZE][NUM_CLASS];
		
		int num_inputs, max_epoch;
		int allowed_input_index;  // this variable is to store the number of selected input from the input set
		char file_name[50];
		//int layer_max, layer_structure[3];
		//int dummy;
		float time_to_stop;
	public:
		dnc();
		int set_monotone_increament(int n);
		dnc(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, float learning_rate, float momentum, char *f_name, float stop_time);
		int init_dnc(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, float learning_rate, float momentum, char *f_name, float stop_time);
		float execute(float *input, float *output);
		float execute2(float *input, float *output);
		int take_validation_data(float *v_input, float *v_output, int num_v_data); // This should be called before the execute function from main, num_v_data-> number of validation inputs
		float calculate_average_error(int num_inputs);
	
};
