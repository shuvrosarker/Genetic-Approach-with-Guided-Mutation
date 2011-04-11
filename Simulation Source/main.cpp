#include "dnc.h"
#include "marchand.h"
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cstdio>
//#include <stdlib.h>


/* these variable depends on data set */
#define ALPHA 0.95
#define BETA  0.05
#define TRAIN_SIZE 384
#define ATTR_NUM 8
#define NUM_CLASS 2
#define NUM_CLASS_MARCHAND 2
#define INIT_NET 5
#define TOTAL_NET (2*INIT_NET)
#define NUM_GEN 10
#define EPOCH_DNC 350
#define TRAIN_TIME 1.0


/* number of maximum nodes in hidden layer */
#define MAX_NODES 200
#define VALIDATION_SIZE 192

// For Main Algorithm this parameters are used
#define MAIN_ERROR_INCREASE 3
#define MAIN_ERROR_CHANGE 0.001

float error[TOTAL_NET];
float score[TOTAL_NET];
float cum_score[TOTAL_NET];
float output[TRAIN_SIZE][NUM_CLASS];
float input[TRAIN_SIZE][ATTR_NUM];

float output_temp[TRAIN_SIZE][NUM_CLASS];
float input_temp[TRAIN_SIZE][ATTR_NUM];

float remaining_output[TRAIN_SIZE][NUM_CLASS];
float remaining_input[TRAIN_SIZE][ATTR_NUM];

float validation_input[TRAIN_SIZE][ATTR_NUM];
float validation_output[TRAIN_SIZE][NUM_CLASS];

void normalize_vector(float *vector, int vec_length);

int permutation_next()
{
    vector<int> myVect, copyVect;
    vector<int >:: iterator it;
    int var;
    
    //initialization
    for(int i = 0; i < TRAIN_SIZE; i++) myVect.push_back(i);
    //srand(time(NULL));
    

    copyVect.resize(myVect.size());
    copy(myVect.begin(), myVect.end(), copyVect.begin());

    int index  = 0;
    while(!myVect.empty())
    {
        var = rand() % myVect.size();
        //cout <<  "random index : " << myVect[var] << endl ;
	
	//cout << "index : " << index << "output : ";    
	for(int i = 0; i < NUM_CLASS; i++)
	{
		output_temp[index][i] = output[var][i];
		//cout << output_temp[index][i] << " ";
	}
	//cout << endl;
	
	//cout << "index : " << index << "input : ";    
	for(int i = 0; i < ATTR_NUM; i++)
	{
		input_temp[index][i]  = input[var][i];
		//cout << input_temp[index][i] << " ";
	}
	//cout << endl;
	
        it = find( myVect.begin(), myVect.end(), myVect[var] );
        myVect.erase(it);
	index++;
    }
    //cout << endl;
    myVect.resize(copyVect.size());
    copy(copyVect.begin(), copyVect.end(), myVect.begin());
   
    return 0;
}

int permutation_next_another(float *input, float *output, int size)
{
    vector<int> myVect, copyVect;
    vector<int >:: iterator it;
    int var;
    float local_input[TRAIN_SIZE][ATTR_NUM], local_output[TRAIN_SIZE][NUM_CLASS];
    
    //initialization
    for(int i = 0; i < size; i++) myVect.push_back(i);
    
    copyVect.resize(myVect.size());
    copy(myVect.begin(), myVect.end(), copyVect.begin());

    int index  = 0;
    while(!myVect.empty())
    {
        var = rand() % myVect.size();
        
	for(int i = 0; i < NUM_CLASS; i++) local_output[index][i] = *(output + var*NUM_CLASS + i);	
	for(int i = 0; i < ATTR_NUM; i++) local_input[index][i]  = *(input + var*ATTR_NUM + i);
	
        it = find( myVect.begin(), myVect.end(), myVect[var] );
        myVect.erase(it);
	index++;
    }
    
    myVect.resize(copyVect.size());
    copy(copyVect.begin(), copyVect.end(), myVect.begin());
    
    for(int i = 0; i < size; i++)
    {
	for(int j = 0; j < ATTR_NUM; j++)  *(input + i*ATTR_NUM + j) = local_input[i][j];
	for(int k = 0; k < NUM_CLASS; k++)  *(output + i*NUM_CLASS + k) = local_output[i][k];
    }

    return 0;

}

void normalize_vector(float *vector, int vec_length)
{
	int loop_i;
	float sum = 0.0;
	
	for(loop_i = 0; loop_i < vec_length; loop_i++ )
	{
		sum += vector[loop_i]*vector[loop_i];
	}
	sum = sqrt(sum);
	
	for(loop_i = 0; loop_i < vec_length; loop_i++ )
	{
		vector[loop_i] /= sum;
	}
}

int read_data_from_file(char *file_name, int input_or_validation)   // 1 is for taking input, 2 is for validation data
{
	int loop_i, loop_j;
	FILE *fp  = fopen(file_name,"r");
	
	if(fp == NULL) printf("Error Opening File %s\n", file_name);
	//FILE *fp2 = fopen("output_file", "w");
	char comma;
	float temp;
	
	int num_row = 0;
	while( true )
	{
		if( fscanf(fp, "%f", &temp) == EOF) break; // the index is scanned here
		
		for(loop_i = 0; loop_i < ATTR_NUM ; loop_i++)
		{
			if(input_or_validation == 1 ) fscanf(fp, "%c %f", &comma, &input[num_row][loop_i]);
			if(input_or_validation == 2 ) fscanf(fp, "%c %f", &comma, &validation_input[num_row][loop_i]);
		
		}
		//printf("\n");
		//normalize_vector(input[num_row], ATTR_NUM); // This line should be uncommented for fractional data as input
		
		int value;
		fscanf(fp, "%c %d", &comma, &value);
		//fprintf(fp2, "%d\n",value);
		
		for( loop_j = 0; loop_j < NUM_CLASS; loop_j++ )
		{
			if(loop_j == (value- 1 ) )
			{
				if(input_or_validation == 1) output[num_row][loop_j] = 1;
				else if(input_or_validation == 2) validation_output[num_row][loop_i] = 1;
			}
			else
			{
				if(input_or_validation == 1) output[num_row][loop_j] = 0;
				else if(input_or_validation == 2) validation_output[num_row][loop_i] = 0;
			}
		}
		num_row++;
	}
		
	fclose(fp);
	//fclose(fp2);

	return 0;
	
}

int print_input_data()
{
	int loop_i = 0, loop_j = 0;
	
	while(loop_i < TRAIN_SIZE)
	{
		while( loop_j < (ATTR_NUM -1) )
		{
			printf("%f ", input[loop_i][loop_j++]);
			//loop_j++;
		}
		printf("\n");
		loop_j = 0;
		while( loop_j < (NUM_CLASS -1) )
		{
			printf("%f ", output[loop_i][loop_j++]);
			//loop_j++;
		}
		
		printf("\n\n");
		loop_j = 0;
		loop_i++;
	}
	return 0;
	
}

int print_array(float *input, int length)
{
	for(int i = 0; i < length; i++) printf("%f ", input[i]);
	printf("\n");
	return 0;
}

int hidden_node(char *file_name, int *hidden, int *selected_source)
{
	int tmp_a, tmp_b, tmp_c;
	FILE *fp = fopen(file_name, "r");
	if(fp != NULL)
	{
		fscanf(fp, "%d %d %d %d %d", selected_source, &tmp_a, &tmp_b, &tmp_c, hidden);
		fclose(fp);
		return 0;
	}
	else
	{
		cout << "error opening file ... " << endl;
		return -1;
	}
}

int select_network(int *selected_index, int *selected_source)
{	
	int i,j,hidden_nodes[TOTAL_NET][3]; // 0 is index, 1 is num_hidden_node, 2 is from where(dnc or marchand)
	char file_name[50];
	int sum = 0; // sum of hidden nodes
	float hidden_rate[TOTAL_NET];
	
	for(i = 0, j = 0; i < INIT_NET; i++, j++)
	{
		sprintf(file_name, "dnc%d", i);
		hidden_nodes[j][0] = i;
		hidden_node(file_name, &hidden_nodes[j][1], &hidden_nodes[j][2]);
		strcpy(file_name, "");
	}
	
	for(i = 0; i < INIT_NET; i++, j++)
	{
		sprintf(file_name, "marchand%d", i);
		hidden_nodes[j][0] = i;
		hidden_node(file_name, &hidden_nodes[j][1], &hidden_nodes[j][2]);
		strcpy(file_name, "");
	}
	
	int max_hidden = hidden_nodes[0][1];
	for(i = 0; i < TOTAL_NET; i++)
	{
		printf("Values are : %d %d %d\n",hidden_nodes[i][0], hidden_nodes[i][1], hidden_nodes[i][2]);
		sum += hidden_nodes[i][1];
		if(hidden_nodes[i][1] > max_hidden) max_hidden = hidden_nodes[i][1];
	}
	
	float score_sum = 0.0;
	
	//cout << "Selection Error : " ;
	for(i = 0; i < TOTAL_NET; i++)
	{
		score[i] = 100.0 -  (((BETA *error[i] * max_hidden) + ( ALPHA * hidden_nodes[i][1])) * 100.0)/(max_hidden); //beta barale effeciency, alpha hidden neurons
		score_sum += score[i];
	}

	for(i = 0; i < TOTAL_NET; i++)
	{
		//hidden_rate[i] = 100.0 - (float)(100.0*hidden_nodes[i][1])/(float)(sum);
		cum_score[i] = (float)(100.0)* (float)(score[i])/(float)(score_sum);
		if( i > 0)
			//hidden_rate[i] += hidden_rate[i - 1];
			cum_score[i] += cum_score[i - 1];
	}
	//printf("\n");
	
	int rand_value = rand() % 100 + 1;
	
	for( i = 0; i < TOTAL_NET; i++)
	{
		//if(rand_value <= hidden_rate[i])
		if(rand_value <= cum_score[i])
		{
			*selected_index  = hidden_nodes[i][0];
			*selected_source = hidden_nodes[i][2];
			break;
		}
	}
	//cout << "hidden_rate : " << hidden_rate[TOTAL_NET - 1] << endl;
	
	
	return 0;
}

int train_network_pool(float error_limit)
{
	int i, init = 0; //init is 0 for the first generation of the algorithm
	char index[10],file_name[100], selected_network_file[100];
	dnc* dnc_vars;
	Marchand* marchand_vars;
	float learning_rate, momentum, local_error, threshold  = 0.001;
	int selected_index, selected_source;   // selected_source = 0 -> dnc, 1 -> marchand selected index x-> xth network of selected_source
	int present_gen = 0;
	int count_mc = 0; // count misclassification
	float init_error = 1.0;
	
	srand(time(NULL));
	dnc_vars     =  new dnc[INIT_NET];
	marchand_vars =  new Marchand[INIT_NET];
	nn test_network;
	float present_error, previous_error;
	int increase_count = 0;
	
	present_error = previous_error = 1.0;
	char file_name_to_save[100];
	
	FILE  *fp_output = fopen("graph_input.csv","w");
	
	if(fp_output == NULL) printf("Error Opening File ... %s\n", "graph_input.csv");
	
	fprintf(fp_output, "%d,%f\n", present_gen, 1.0*NUM_GEN);
	
	while(present_gen < NUM_GEN)
	{
		if(present_gen)
		//if(true)
		{
			float local_error = 0.0;
			learning_rate = (double)(rand() % 10 + 2) / (double)(100);
			momentum      = (double)(rand() % 9 + 1) / (double)(10);
		
			cout << "GEN learing rate : " << learning_rate << " momentum : " << momentum << endl;
			
			int layer[3] = {ATTR_NUM, 1, NUM_CLASS};
			
			test_network.create_nn(3, layer, 0,  learning_rate, momentum, MAX_NODES);
			printf("running ..");
			test_network.load(selected_network_file);
			
			//for( int i = 0; i < TRAIN_SIZE; i++)
			for( int i = 0; i < VALIDATION_SIZE; i++)
			{
				test_network.calculate(validation_input[i]);			
				local_error += test_network.get_error(validation_output[i]);
			}
			
			printf("Total error: %f Average error: %f\n", local_error, (float)(local_error/ TRAIN_SIZE));
			fprintf(fp_output, "%d,%f\n", present_gen, NUM_GEN * (float)(local_error/ TRAIN_SIZE));
			
			if( (float)(local_error/ TRAIN_SIZE) < error_limit)
			{
				test_network.save((char *) "genetic_ouptut.txt", 2);
				printf("The network is converged ...\n");
				fclose(fp_output);
				return 0;
			}
			else
			{
				test_network.save((char *) "genetic_ouptut.txt", 2);
				printf("The network is SAVED ...\n");
				
			}
			
			present_error =  (float)(local_error/ TRAIN_SIZE);
			if(fabs(present_error - previous_error) < (error_limit/10))
			{
				printf("Network Converged ... final case\n");
				test_network.save((char *) "genetic_ouptut.txt", 2);
				fclose(fp_output);
				return 0;
			}
			
			if(present_error > previous_error)
			{
				increase_count++;
				sprintf(file_name_to_save, "genetic/genetic_output%d", increase_count);
				test_network.save((char*) file_name_to_save);
				sprintf(file_name_to_save, "");
			}
			else if(present_error < previous_error) increase_count = 0;
			
			if(increase_count == MAIN_ERROR_INCREASE)
			{
				test_network.save((char *) "genetic_ouptut.txt", 2);
				printf("Returning from the main function ... %d\n", increase_count);
				fclose(fp_output);
				return 0;
			}
			else if( (present_error > previous_error) && (present_error - previous_error) < MAIN_ERROR_CHANGE)
			{
		
				sprintf(file_name_to_save, "genetic/genetic_outputLastCase");
				test_network.save((char*) file_name_to_save);
				sprintf(file_name_to_save, "");
				
				printf("Last case, Increase Count : %d\n", increase_count);
				fclose(fp_output);
				return 0;
			}
			printf("The increase count is : %d\n", increase_count);
			previous_error = present_error;
			
			
		}
	
		int m;
		for(i = 0; i < INIT_NET; i++)
		{
			sprintf(file_name, "dnc%d",i);
			//puts(file_name);
			//supply each network a time to stop, a random learning rate and a random permutation
			learning_rate = (double)(rand() % 10 + 2) / (double)(100);
			momentum      = (double)(rand() % 7 + 1) / (double)(10);
		
			cout << "learing rate : " << learning_rate << " momentum : " << momentum << endl;
			dnc_vars[i].init_dnc(ATTR_NUM, NUM_CLASS, TRAIN_SIZE, EPOCH_DNC, 0.001, MAX_NODES, learning_rate, momentum, file_name, TRAIN_TIME);
			dnc_vars[i].set_monotone_increament(5);
			dnc_vars[i].take_validation_data((float *) validation_input, (float *)validation_output, VALIDATION_SIZE); // num of validation data as last argument
			
			if(present_gen) dnc_vars[i].load(selected_network_file);
			
			error[i] = dnc_vars[i].execute((float *)input, (float *)output);
			strcpy(file_name,"");
		}
		
		m = i; // hold the value INIT_NET
		for(i = 0; i < INIT_NET; i++)
		{
			strcpy(file_name,"");
			sprintf(file_name, "marchand%d", i);
			//puts(file_name);
			
			if(!present_gen)
			{
				permutation_next();
				marchand_vars[i].init_marchand(ATTR_NUM, NUM_CLASS, TRAIN_SIZE, 200, 0.05, MAX_NODES, file_name, TRAIN_TIME/3.0);
				marchand_vars[i].take_validation_data((float *) validation_input, (float *)validation_output, VALIDATION_SIZE);
				error[m+i] = marchand_vars[i].execute((float *)input_temp, (float *)output_temp);
			}
			else
			{
				permutation_next_another((float *)remaining_input, (float *)remaining_output, count_mc);
				marchand_vars[i].init_marchand(ATTR_NUM, NUM_CLASS, count_mc, 200, 0.05, MAX_NODES, file_name, TRAIN_TIME/3.0);
				marchand_vars[i].load(selected_network_file);
				marchand_vars[i].take_validation_data((float *) validation_input, (float *)validation_output, VALIDATION_SIZE);
				error[m+i] = marchand_vars[i].execute((float *)remaining_input, (float *)remaining_output);
			}
		}
	
		//best selection
		select_network(&selected_index, &selected_source);
		cout << "selected index: " << selected_index << " selected source: " << selected_source << endl;
	
		count_mc = 0;
	
		if(selected_source == 0)
		{
			for(int m = 0; m < TRAIN_SIZE; m++)
			{
				local_error = dnc_vars[selected_index].calculate_for_single(input[m],output[m]);
				
				if(local_error > threshold)
				{
					for(int k = 0; k < ATTR_NUM; k++) remaining_input[count_mc][k] = input[m][k];
					for(int k = 0; k < NUM_CLASS; k++) remaining_output[count_mc][k] = output[m][k];
					count_mc++;
				}
			}
		}
	
		if(!selected_source) sprintf(selected_network_file, "dnc%d", selected_index);
		else sprintf(selected_network_file, "marchand%d", selected_index);
		
		cout << "Generation is  : " << present_gen << endl;		
		present_gen++;
	}
	
	delete []dnc_vars;
	delete []marchand_vars;
	return 0;
}



/* Now write a function to find the misclassified data */
int main()
{
	read_data_from_file( (char *)"pima_diabetes/pima_diabetes_train.txt", 1);
	read_data_from_file( (char *)"pima_diabetes/pima_diabetes_validation.txt", 2); // copies data into validation_input and validation_output
	
	printf("Code is running ...\n");
	int inputs, count = 0;
	float error, threshold = 0.1;
	
	train_network_pool(0.04);
	
	
	dnc dnc_instance2(ATTR_NUM, NUM_CLASS, TRAIN_SIZE, 540, 0.001, MAX_NODES, 0.05, 0.8, (char *)"dnc_output.txt", 20.0);
	dnc_instance2.set_monotone_increament(5);
	
	dnc_instance2.take_validation_data((float *) validation_input, (float *)validation_output, VALIDATION_SIZE); // num of validation data as last argument
	dnc_instance2.execute2((float *)input, (float *)output);
	
	Marchand marchand_instance(ATTR_NUM, NUM_CLASS_MARCHAND, TRAIN_SIZE, 200, 0.05, MAX_NODES, (char *)"marchand_output.txt", 8.0); //max epoch 200, this call constructs network with 2 output neurons
	marchand_instance.take_validation_data((float *) validation_input, (float *)validation_output, VALIDATION_SIZE);
	marchand_instance.execute((float *)input, (float *)output);
	

	cout << "I am here" << endl;
	
	
	return 0;
	
}