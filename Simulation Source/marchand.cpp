/*************************************************************
* The code is designed to work for multiple classes of data. *
* But now we are working for only two classes.               *
**************************************************************/
  
#include "marchand.h"
#include <iostream>
#include <math.h>
#include <ctime>
#include <cstring>

#define SWITCH_TIME 0.5

using namespace std;

Marchand::Marchand()
{
    
}

Marchand::Marchand(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, char *f_name, float stop_time)
{
    
    error_limit = min_error;
    int layer_structure[3] = {num_in, 1, num_out}; /*create a 3-layered neural network */
    create_nn(3, layer_structure, 0, 0.001, 0.3, lmax); // watch the 3rd parameter, we are not using BackPropagation here.
    num_inputs = num_inputs_a;
    max_epoch  = max_epoch_a;
    time_to_stop = stop_time;
    
    strcpy(file_name, f_name);
}

int Marchand::init_marchand(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, char *f_name, float stop_time)
{
    
    error_limit = min_error;
    int layer_structure[3] = {num_in, 1, num_out}; /*create a 3-layered neural network */
    create_nn(3, layer_structure, 0, 0.001, 0.3, lmax); // watch the 3rd parameter, we are not using BackPropagation here.
    num_inputs = num_inputs_a;
    max_epoch  = max_epoch_a;
    time_to_stop = stop_time;
    strcpy(file_name, f_name);
    return 0;
}



Marchand::~Marchand()
{
}

int Marchand:: set_hidden_to_output_weight()
{
    float weight;
    
    for(int i = 0; i < positive_hidden_indexes.size(); i++)
    {
	weight = 1.0/(float)pow(2,i);
	set_weight(1, i, 0, weight);
	set_weight(1, i, 1, 0.0);
    }
    
    for(int i = 0; i < negative_hidden_indexes.size(); i++)
    {
	weight = 1.0/(float)pow(2,i);
	set_weight(1, i, 0, 0.0);
	set_weight(1, i, 1, weight);
    }
}

bool Marchand:: set_output_check(set< vector<float> > set_data, int output)
{

	set< vector<float> >:: iterator it;
	float local_output[LMAX], data_array[ATTR_NUM] ;
	int hidden_neurons;
	float temp_output;
	
	int var = set_data.size();
	
	for(it = set_data.begin(); !set_data.empty() && it != set_data.end();it++)
	//while(!set_data.empty())
	{
		
		//it = set_data.begin();
		temp = it->begin();
		

		for(int m = 0; temp != it->end(); m++,temp++)
		{
		    data_array[m] = *temp;
		}
		
		
		calculate_layer_output(data_array, 1, local_output);
		hidden_neurons = get_l_neurons(1);
		temp_output = local_output[hidden_neurons - 1];
		
	    
		if(temp_output <= 0.5) temp_output = 0.0;
		else temp_output = 1.0;
		
		if( fabs(temp_output - output) > 0.03)
		{
			return false;
		}
		
	}
	return true;
}

int Marchand:: number_right_classification(set< vector<float> > set_data, int output)
{
	
	set< vector<float> >:: iterator it;
	float local_output[LMAX], data_array[ATTR_NUM] ;
	int hidden_neurons, count = 0;
	float temp_output;
	
	int var =  set_data.size();
	
	//while(!set_data.empty())
	for (it = set_data.begin(); it != set_data.end(); it++) 
	{
	
		//it = set_data.begin();
		temp = it->begin();
		
		for(int m = 0; temp < it->end(); m++,temp++) data_array[m] = *temp;
		
		
		calculate_layer_output(data_array, 1, local_output);
		hidden_neurons = get_l_neurons(1); //gets number of hidden neurons
		
		temp_output = local_output[hidden_neurons - 1];
		
		
		if(temp_output <= 0.5) temp_output = 0.0;
		else temp_output = 1.0;
		
		if( fabs(temp_output - output) > 0.03)
		{
			//cout << "False" << endl;
			//return false;
		}
		else
			count++;
		
		
	}
	
	
	return count;
}

float Marchand:: execute(float *input, float *output)
{
    float temp_var,local_output[LMAX];
    float temp_output;
    int hidden_neurons;
    vector<float> temp_input;
    set< vector<float> > temp_set;
    set< vector<float> > temp_set_two;
    set< vector<float> >:: iterator it;
    float data_array[ATTR_NUM];
	
    vector< float >:: const_iterator temp;
    vector<float> local_vector_temp;
	
    local_vector_temp.push_back(1);
    float local_time = 0.0;
	
	
    
    //for(int i = 0; i < TRAIN_SIZE; i++)
    cout << "Number of Inputs : " << num_inputs << endl;
    for(int i = 0; i < num_inputs; i++)
    {
	for(int j = 0; j < ATTR_NUM; j++) temp_input.push_back(*(input + i*ATTR_NUM + j));

        for(int k = 0; k < NUM_CLASS; k++)
	{
	    temp_var = *(output + i*NUM_CLASS + k);
	    
			
	    /* if this is of class 0*/
	    if(temp_var && !k)
	    {
		temp_set.insert(temp_input);
		
	    }
	    else if(temp_var && k == 1)
	    {
		temp_set_two.insert(temp_input);
	    }
		
	}
	temp_input.erase(temp_input.begin(), temp_input.end());
    }
    
    classes.push_back(temp_set);
    classes.push_back(temp_set_two);
    
	
    int var = 0;
    clock_t init, final, init_temp;
    bool flag1, flag2;
    
    float present_error = 1.0, previous_error;
    int num_monotone_increase = 0;
    
    //previous_error = present_error;
    present_error = calculate_average_error(num_validation);
    
    
    init = clock();
    init_temp = clock();
	//if you need to extend this for multi class you need to change here.
    while( !classes[0].empty() && !classes[1].empty() && ((double)(clock() - init_temp) / (double)CLOCKS_PER_SEC) < time_to_stop)
    {

	//cout << "local time : " << ((double)(clock() - init_temp) / (double)CLOCKS_PER_SEC) << " time to stop: " << time_to_stop << endl;
	
	flag1 = true;
		
	while( !(set_output_check(classes[0], 0) && number_right_classification(classes[1], 1) ) )  // there is a possiblility of infinite loop here. Let's see what happens.x
	{
	    final = clock() - init;
	        
	    if ( ( (double)final / (double) CLOCKS_PER_SEC) > SWITCH_TIME) 
	    {
	        //cout << "I am here " << endl;
		flag1 = false;
		break;
	    }	
	    
	    int l = get_l_neurons(1) - 1; // index of last added neuron of hidden layer
	    change_random_weight(1, l);
	    
	}
	
	
	if (flag1) 
	{
	    
	    int debug_var = 0;
	
	    // Now calculate which new nodes are classified by the added node.
	    for(it = classes[1].begin(); !classes[1].empty() && it != classes[1].end(); )
	    {
		temp = it->begin();
		for(int m = 0; temp < it->end(); m++,temp++) data_array[m] = *temp;
		
		calculate_layer_output(data_array, 1, local_output);
		hidden_neurons = get_l_neurons(1);
		temp_output = local_output[hidden_neurons - 1];
		
		if(temp_output <= 0.5) temp_output = 0.0;
		else temp_output = 1.0;
		
		if( fabs(temp_output - 1.0) < 0.03)  // just to be a bit secure
		{
		    debug_var = 1;
		    negative_hidden_indexes.push_back( get_l_neurons(1) - 1);
		    classes[1].erase(it); // delete the matched element from the set.
		    it = classes[1].begin();
		    var++;
		}
		else
		{
		    it++;
		}
		
	    }
			
	    add_hidden_node(1,1);
	    

	}
		
	
	if (!flag1) 
	{
	    init = clock();
	    flag2 = true;
			
	    
	    while( !(set_output_check(classes[1], 0) && number_right_classification(classes[0], 1) ) )  // there is a possiblility of infinite loop here. Let's see what happens.x
	    {
		final = clock() - init;
		
		//cout << "case Two" << endl;
		//cout << "Time Second : " << ((double)final / (double) CLOCKS_PER_SEC) << endl;
		if ( ((double)final / (double) CLOCKS_PER_SEC) > SWITCH_TIME) 
		{
		    flag2 = false;
		    break;
		}
		int l = get_l_neurons(1) - 1; // index of last added neuron of hidden layer
		change_random_weight(1, l);
		
	    }
	    
	    if (flag2) 
	    {
				
		int debug_var = 0;
				
		// Now calculate which new nodes are classified by the added node.
		for(it = classes[0].begin(); !classes[0].empty() && it != classes[0].end(); )
		{
		    temp = it->begin();
		    for(int m = 0; temp < it->end(); m++,temp++) data_array[m] = *temp;
					
		    calculate_layer_output(data_array, 1, local_output);
		    hidden_neurons = get_l_neurons(1);
					
					
		    temp_output = local_output[hidden_neurons - 1];
					
		    if(temp_output <= 0.5) temp_output = 0.0;
		    else temp_output = 1.0;
		    
		    // cout << "temp output : " << temp_output << endl; 
					
		    if(fabs(temp_output - 1.0) < 0.03)  // just to be a bit secure
		    {
		        
			debug_var = 1;
			positive_hidden_indexes.push_back( get_l_neurons(1) - 1);
			//cout << "half way size " << classes[0].size() << endl;
			classes[0].erase(it); // delete the matched element from the set.
			it = classes[0].begin();
			var++;
		    }
		    else
		    {
			it++;
		    }
					
		}
		    //cout << "I am out of the for loop" << endl;		
		add_hidden_node(1,1);
		    //cout << "adding hidden node second case... " << get_l_neurons(1) << endl;
				
	    }
			
	}
	
    }
	
    
    //cout << "process end" << endl;
    set_hidden_to_output_weight();
    //cout << "complete ... Marchand hidden neurons -> " << get_l_neurons(1) << endl;
    save((char *)file_name, 1); // 1 is algo index for marchand
    local_time = (double)(clock() - init_temp) / (double)CLOCKS_PER_SEC;
    get_layer_data();
    present_error = calculate_average_error(num_validation);
    cout << "Marchand Error : " << present_error << endl;
    //cout << "local time : " << local_time << endl;
    return present_error;
}

/* This method is to debug things. */
void Marchand:: doDecode( vector< set< vector<float> > > data)
{
    set< vector<float> >::iterator it;
    vector<float>::const_iterator vi;
    //set< vector<float> > temp;
    
    for( int i = 0; i < data.size(); i++)
    {
        for( it = data[i].begin(); it != data[i].end(); it++)
        {

            for( vi = it->begin() ; vi != it->end() ; vi++)
            {
                printf("%f ", *vi);
            }
            printf("\n");
        }
        printf("\n");
    }
}


int Marchand::set_monotone_increament(int n)
{
    monotone_increament = n;
}

int Marchand:: take_validation_data(float *v_input, float *v_output, int num_v_data)
{
    num_validation = num_v_data;
	
    for(int i = 0; i < num_v_data; i++)
    {
	for(int j = 0; j < ATTR_NUM; j++) validation_input[i][j]  = *(v_input + i*ATTR_NUM + j);
	for(int k = 0; k < NUM_CLASS; k++) validation_output[i][k] = *(v_output + i*NUM_CLASS + k);
    }
	
    return 0;
}


float Marchand::calculate_average_error(int arg_num_inputs)
{	
	int loop_i;
	float error = 0.0;
	float var = 0.0;
		
	for(loop_i = 0; loop_i < arg_num_inputs; loop_i++)
	{
		//cout << "desired output : " << outputs[loop_i][0] << "  " << outputs[loop_i][1] << " index: " << loop_i << endl;
		var = calculate_for_single(validation_input[loop_i], validation_output[loop_i]);
		error += var;
	}
	
	return (float)(error/arg_num_inputs);
	
}