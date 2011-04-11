/*
 *  dnc.cpp
 *  NeuralConstructive
 *
 *  Created by Shuvro Sarker on 1/23/11.
 *  Copyright 2011 BUET. All rights reserved.
 *
 */

#include "dnc.h"
#include <iostream>
#include <cstring>
#include <ctime>
using namespace std;

dnc::dnc()
{
	
}
dnc::dnc(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, float learning_rate, float momentum, char *f_name, float stop_time)
{
	//nn network;
	error_limit = min_error;
	//cout << "output neurons : " << num_out << endl;
	int layer_structure[3] = {num_in, 1, num_out}; /*create a 3-layered neural network */
	create_nn(3, layer_structure, 1, learning_rate, momentum, lmax);
	num_inputs = num_inputs_a;
	max_epoch  = max_epoch_a;
	time_to_stop = stop_time;
	strcpy(file_name, f_name);
}


int dnc::init_dnc(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, float learning_rate, float momentum, char *f_name, float stop_time)
{
	//nn network;
	error_limit = min_error;
	//cout << "output neurons : " << num_out << endl;
	int layer_structure[3] = {num_in, 1, num_out}; /*create a 3-layered neural network */
	create_nn(3, layer_structure, 1, learning_rate, momentum, lmax);
	num_inputs = num_inputs_a;
	max_epoch  = max_epoch_a;
	time_to_stop = stop_time;
	strcpy(file_name, f_name);

	return 0;
}

int dnc:: set_monotone_increament(int n)
{
	monotone_increament = n;
	return 0;
}

//To use this function for multi class we need to modify its code like the first portion of execute function
int dnc:: take_validation_data(float *v_input, float *v_output, int  num_v_data)
{
	//float copy_output;
	num_validation = num_v_data;
	
	for(int i = 0; i < num_v_data; i++)
	{
		for(int j = 0; j < ATTR_NUM; j++) validation_input[i][j]  = *(v_input + i*ATTR_NUM + j);
		for(int k = 0; k < NUM_CLASS; k++) validation_output[i][k] = *(v_output + i*NUM_CLASS + k);
	}
	
	return 0;
}

/* We can filter the inputs to get for only two classes.*/
float dnc::execute(float *input, float *output)
{
		
	int loop_i, loop_j;
	float error;
	float  local_time = 0.0;
	clock_t init, final;
	int num_monotone_increase = 0; // number of montonically increasing errors.
	
	float present_error = 1.0, previous_error;
	allowed_input_index = 0;
	float copy_output;
	
	previous_error =  present_error;
	
	
	for(int i = 0; i < TRAIN_SIZE; i++)
	{
		for(int j = 0; j < ATTR_NUM; j++)
		{
			inputs[allowed_input_index][j]  = *(input + i*ATTR_NUM + j);
		}
		
		for(int k = 0; k < NUM_CLASS; k++)
		{
			copy_output = *(output + i*NUM_CLASS + k);
			outputs[allowed_input_index][k] = copy_output;
			
			if( (k == 0 && copy_output==1) || (k == 1 && copy_output == 1) )
			{
				if(k == 0) outputs[allowed_input_index][1] = 0;
				allowed_input_index++;
				break;
			}
		}
		
	}

	present_error = calculate_average_error(num_validation) ;

	
	init = clock();
	while( (present_error > error_limit) && (get_l_neurons(1) < get_max_nodes() )  && local_time < time_to_stop)
	//while( (get_l_neurons(1) < get_max_nodes() )  && local_time < time_to_stop)
	{
			
		for(loop_j = 0; loop_j < max_epoch; loop_j++)
		{
			//for(loop_i = 0; loop_i < num_inputs; loop_i++)
			for(loop_i = 0; loop_i < allowed_input_index; loop_i++)
			{
				calculate(inputs[loop_i]);
				backpropagate(outputs[loop_i]);
 				error = get_error(outputs[loop_i]);
			}
		
		
			present_error = calculate_average_error(num_validation) ; // calculate error for validation data, not for training data
		
		}
		add_hidden_node(1,1);
		local_time = (double)(clock() - init) / (double)(CLOCKS_PER_SEC);
		save(file_name, 0); // 0 is algo index for dnc
		
		if(present_error > previous_error) num_monotone_increase++;
		else if(present_error < previous_error) num_monotone_increase = 0;
		
		previous_error = present_error;
		
		if(num_monotone_increase == monotone_increament)
		{
			sub_hidden_node(monotone_increament, 1); // decrease monotone_increament neurons from layer 1
			break; // training complete.
		}
		
		
		
	}
	
	printf("Final amount of error %f  epoch number: %d\n", present_error,max_epoch);
	get_layer_data();
	cout << endl;
	
	return present_error;
}

float dnc::execute2(float *input, float *output)
{
		
	int loop_i, loop_j;
	float error;
	float  local_time = 0.0;
	clock_t init, final;
	int num_monotone_increase = 0; // number of montonically increasing errors.
	
	float present_error = 1.0, previous_error;
	allowed_input_index = 0;
	float copy_output;
	
	previous_error =  present_error;
	
	
	for(int i = 0; i < TRAIN_SIZE; i++)
	{
		for(int j = 0; j < ATTR_NUM; j++)
		{
			inputs[allowed_input_index][j]  = *(input + i*ATTR_NUM + j);
			//cout << inputs[allowed_input_index][j] << " " ;
		}
		//cout << endl;
		for(int k = 0; k < NUM_CLASS; k++)
		{
			
			
			copy_output = *(output + i*NUM_CLASS + k);
			outputs[allowed_input_index][k] = copy_output;
			
			//cout << "copy output is : " << copy_output << "  k: " << k << " num class: " << NUM_CLASS << endl;
			if( (k == 0 && copy_output==1) || (k == 1 && copy_output == 1) )
			{
				if(k == 0) outputs[allowed_input_index][1] = 0;
				allowed_input_index++;
				break;
			}
			
		//	cout << outputs[i][k] << " ::" << endl;
		}
		
	}

	
	present_error = calculate_average_error(num_validation) ;
	
	
	init = clock();
	while( (present_error > error_limit) && (get_l_neurons(1) < get_max_nodes() )  && local_time < time_to_stop)
	//while( (get_l_neurons(1) < get_max_nodes() )  && local_time < time_to_stop)
	{
		
		for(loop_j = 0; loop_j < max_epoch; loop_j++)
		{
			//for(loop_i = 0; loop_i < num_inputs; loop_i++)
			for(loop_i = 0; loop_i < allowed_input_index; loop_i++)
			{
				calculate(inputs[loop_i]);
				backpropagate(outputs[loop_i]);
 				error = get_error(outputs[loop_i]);
			}
			present_error = calculate_average_error(num_validation) ; // calculate error for validation data, not for training data
		
		}
		//printf("amount of error %f  epoch number: %d\n", present_error,max_epoch);
		add_hidden_node(1,1);
		local_time = (double)(clock() - init) / (double)(CLOCKS_PER_SEC);
		save(file_name, 0); // 0 is algo index for dnc
		
	}
	
	printf("Final amount of error %f  epoch number: %d\n", present_error,max_epoch);
	get_layer_data();
	cout << endl;
	
	return present_error;
}

float dnc::calculate_average_error(int arg_num_inputs)
{	
	int loop_i;
	float error = 0.0;
	float var = 0.0;
		
	for(loop_i = 0; loop_i < arg_num_inputs; loop_i++)
	{
		var = calculate_for_single(validation_input[loop_i], validation_output[loop_i]);
		error += var;
	}
	return (float)(error/arg_num_inputs);
	
}

